"""Small synchronous client for the public AoE4 Guides build-order API.

Read-only, standard library only. No polling or background refresh: a call
happens only when the caller asks for it.
"""

import concurrent.futures
import datetime
import json
import os
import pathlib
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from aoe4.aoe4_build_order import check_valid_aoe4_build_order

LIST_ENDPOINT = 'https://aoe4-guides-api-7h2vti5ckq-ey.a.run.app/builds'
OVERLAY_ENDPOINT = 'https://aoe4guides.com/api/builds/{build_id}?overlay=true'
SOURCE_URL = 'https://aoe4guides.com/builds/{build_id}'

CIVILIZATION_LABELS = (
    ('ABB', '阿拔斯王朝'), ('AYY', '阿尤布王朝'), ('BYZ', '拜占庭'),
    ('CHI', '中国'), ('DEL', '德里苏丹国'), ('ENG', '英格兰'),
    ('FRE', '法兰西'), ('GOH', '金帐汗国'), ('HOL', '兰开斯特家族'),
    ('HRE', '神圣罗马帝国'), ('JAP', '日本'), ('JDA', '圣女贞德'),
    ('JIN', '金朝'), ('KTE', '圣殿骑士团'), ('MAC', '马其顿王朝'),
    ('MAL', '马里'), ('MON', '蒙古'), ('DRA', '龙之骑士团'),
    ('OTT', '奥斯曼'), ('RUS', '罗斯'), ('SCO', '苏格兰'),
    ('SEN', '战国大名'), ('TUG', '图格拉克王朝'), ('VIK', '维京'),
    ('ZXL', '朱熹遗产'),
)
CIVILIZATION_NAMES_ZH = dict(CIVILIZATION_LABELS)


class OnlineBuildOrderError(Exception):
    """Raised when the AoE4 Guides API or the local file handling fails."""


@dataclass(frozen=True)
class OnlineBuildSummary:
    """Summary of an online build order, as returned by the list endpoint."""

    build_id: str
    title: str
    civilization: str
    author: str
    season: Optional[str] = None
    updated: Optional[datetime.datetime] = None
    views: int = 0


def _parse_updated(value, url: str, index: int) -> Optional[datetime.datetime]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise OnlineBuildOrderError(f'List item {index} from {url} has invalid "timeUpdated".')
    seconds = value.get('_seconds')
    if not isinstance(seconds, int) or isinstance(seconds, bool) or seconds < 0:
        raise OnlineBuildOrderError(
            f'List item {index} from {url} has invalid "timeUpdated._seconds".'
        )
    try:
        return datetime.datetime.fromtimestamp(seconds, tz=datetime.timezone.utc)
    except (OverflowError, OSError, ValueError) as err:
        raise OnlineBuildOrderError(
            f'List item {index} from {url} has invalid "timeUpdated._seconds": {err}'
        ) from err


def _parse_views(value, url: str, index: int) -> int:
    if value is None:
        return 0
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise OnlineBuildOrderError(f'List item {index} from {url} has invalid "views".')
    return value


class Aoe4GuidesClient:
    """Synchronous client for the public read-only AoE4 Guides API."""

    def __init__(
        self,
        opener: Optional[Callable] = None,
        timeout_seconds: float = 10.0,
        bundled_directory=None,
    ):
        self._timeout_seconds = timeout_seconds
        self._opener = (
            opener
            if opener is not None
            else lambda url: urllib.request.urlopen(url, timeout=self._timeout_seconds)
        )
        self._popular_cache = None
        self._popular_cache_time = 0.0
        self._bundled_directory = (
            pathlib.Path(bundled_directory) if bundled_directory is not None else None
        )
        self._bundled_files = None

    def _load_bundled(self):
        """Load the release snapshot from the already validated build-order folder."""
        if getattr(self, '_bundled_files', None) is not None:
            return self._bundled_files
        files = {}
        bundled_directory = getattr(self, '_bundled_directory', None)
        if bundled_directory is not None and bundled_directory.is_dir():
            for path in bundled_directory.glob('aoe4guides-*.json'):
                try:
                    data = json.loads(path.read_text(encoding='utf-8'))
                    metadata = data.get('_aoe4_guides', {})
                    build_id = metadata.get('id')
                    if isinstance(build_id, str) and build_id:
                        files[build_id] = path
                except (OSError, json.JSONDecodeError, AttributeError):
                    continue
        self._bundled_files = files
        return files

    def _list_bundled(self) -> Tuple[OnlineBuildSummary, ...]:
        summaries = []
        for build_id, path in self._load_bundled().items():
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                metadata = data['_aoe4_guides']
                updated_text = metadata.get('updated')
                updated = (
                    datetime.datetime.fromisoformat(updated_text)
                    if isinstance(updated_text, str) else None
                )
                if updated is not None and updated.tzinfo is None:
                    updated = updated.replace(tzinfo=datetime.timezone.utc)
                summaries.append(OnlineBuildSummary(
                    build_id=build_id,
                    title=str(data['name']),
                    civilization=str(metadata['civilization_code']),
                    author='',
                    updated=updated,
                    views=int(metadata.get('views_at_snapshot', 0)),
                ))
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                continue
        return tuple(summaries)

    def _get_json(self, url: str):
        """Fetch and decode a JSON payload, raising OnlineBuildOrderError."""
        for attempt in range(3):
            try:
                response = self._opener(url)
                try:
                    raw = response.read()
                finally:
                    close = getattr(response, 'close', None)
                    if close is not None:
                        close()
                break
            except urllib.error.HTTPError as err:
                if err.code != 429 or attempt == 2:
                    raise OnlineBuildOrderError(
                        f'Network error while fetching {url}: {err}'
                    ) from err
                retry_after = err.headers.get('Retry-After') if err.headers else None
                try:
                    delay = min(max(float(retry_after or 1), 0.5), 5.0)
                except ValueError:
                    delay = 1.0
                time.sleep(delay)
            except OnlineBuildOrderError:
                raise
            except Exception as err:
                raise OnlineBuildOrderError(f'Network error while fetching {url}: {err}') from err

        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError as err:
            raise OnlineBuildOrderError(f'Invalid UTF-8 response from {url}: {err}') from err

        try:
            return json.loads(text)
        except json.JSONDecodeError as err:
            raise OnlineBuildOrderError(f'Invalid JSON response from {url}: {err}') from err

    def list_latest(
        self,
        civilization_code: Optional[str] = None,
        order_by: Optional[str] = None,
    ) -> Tuple[OnlineBuildSummary, ...]:
        """List the latest build orders, optionally filtered by civilization code."""
        url = LIST_ENDPOINT
        query = {}
        if civilization_code is not None:
            query['civ'] = str(civilization_code)
        if order_by is not None:
            if order_by not in ('score', 'scoreAllTime', 'timeCreated', 'views', 'likes'):
                raise ValueError(f'Unsupported AoE4 Guides order: {order_by}')
            query['orderBy'] = order_by
        if query:
            url += '?' + urllib.parse.urlencode(query, quote_via=urllib.parse.quote)

        data = self._get_json(url)
        if not isinstance(data, list):
            raise OnlineBuildOrderError(f'List response from {url} is not a JSON array.')

        summaries = []
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise OnlineBuildOrderError(f'List item {index} from {url} is not a JSON object.')

            build_id = item.get('id')
            title = item.get('title')
            civ = item.get('civ')
            author = item.get('author') or item.get('creatorName') or ''
            season = item.get('season')
            updated = _parse_updated(item.get('timeUpdated'), url, index)
            views = _parse_views(item.get('views'), url, index)

            for field_name, value in (('id', build_id), ('title', title), ('civ', civ)):
                if not isinstance(value, (str, int)) or isinstance(value, bool):
                    raise OnlineBuildOrderError(
                        f'List item {index} from {url} has missing or invalid "{field_name}".'
                    )

            if not isinstance(author, str):
                raise OnlineBuildOrderError(
                    f'List item {index} from {url} has invalid "author".'
                )

            if season is not None and not isinstance(season, str):
                raise OnlineBuildOrderError(f'List item {index} from {url} has invalid "season".')

            summaries.append(
                OnlineBuildSummary(
                    build_id=str(build_id),
                    title=str(title),
                    civilization=str(civ),
                    author=str(author),
                    season=season,
                    updated=updated,
                    views=views,
                )
            )

        return tuple(summaries)

    def list_popular_recent(
        self, now: Optional[datetime.datetime] = None
    ) -> Tuple[OnlineBuildSummary, ...]:
        """Return three popular guides per civilization, preferring the last year."""
        use_cache = now is None
        if use_cache and self._popular_cache is not None:
            if time.monotonic() - self._popular_cache_time < 300:
                return self._popular_cache
        if now is None:
            now = datetime.datetime.now(datetime.timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=datetime.timezone.utc)
        else:
            now = now.astimezone(datetime.timezone.utc)
        cutoff = now - datetime.timedelta(days=365)

        def fetch(code):
            try:
                return self.list_latest(civilization_code=code, order_by='views')
            except Exception as err:
                raise OnlineBuildOrderError(f'获取文明 {code} 的攻略失败：{err}') from err

        bundled_by_civ = {code: [] for code, _ in CIVILIZATION_LABELS}
        for summary in self._list_bundled():
            bundled_by_civ.setdefault(summary.civilization, []).append(summary)

        results = {}
        snapshot_complete = all(
            len(bundled_by_civ.get(code, ())) >= 3 for code, _ in CIVILIZATION_LABELS
        )
        if not snapshot_complete:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    executor.submit(fetch, code): code for code, _ in CIVILIZATION_LABELS
                }
                for future in concurrent.futures.as_completed(futures):
                    code = futures[future]
                    try:
                        results[code] = future.result()
                    except OnlineBuildOrderError:
                        if not bundled_by_civ.get(code):
                            raise
                        results[code] = ()

        selected = []
        for code, _ in CIVILIZATION_LABELS:
            merged = {}
            for summary in tuple(results.get(code, ())) + tuple(bundled_by_civ.get(code, ())):
                merged.setdefault(summary.build_id, summary)
            available = [summary for summary in merged.values() if summary.updated is not None]
            available.sort(
                key=lambda item: (-item.views, -item.updated.timestamp(), item.title)
            )
            recent = [item for item in available if item.updated >= cutoff]
            older = [item for item in available if item.updated < cutoff]
            selected.extend((recent + older)[:3])
        rows = tuple(selected)
        if use_cache:
            self._popular_cache = rows
            self._popular_cache_time = time.monotonic()
        return rows

    def fetch_overlay(self, build_id) -> dict:
        """Fetch the overlay-compatible build order data for a build ID."""
        bundled_path = self._load_bundled().get(str(build_id))
        if bundled_path is not None:
            try:
                data = json.loads(bundled_path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError) as err:
                raise OnlineBuildOrderError(f'Could not read bundled build order: {err}') from err
            valid, error_msg = check_valid_aoe4_build_order(data)
            if not valid:
                raise OnlineBuildOrderError(f'Invalid bundled build order: {error_msg}')
            return data

        url = OVERLAY_ENDPOINT.format(build_id=urllib.parse.quote(str(build_id), safe=''))
        data = self._get_json(url)
        if not isinstance(data, dict):
            raise OnlineBuildOrderError(f'Overlay response from {url} is not a JSON object.')

        valid, error_msg = check_valid_aoe4_build_order(data)
        if not valid:
            raise OnlineBuildOrderError(f'Invalid build order from {url}: {error_msg}')

        if not data.get('source'):
            data['source'] = SOURCE_URL.format(build_id=urllib.parse.quote(str(build_id), safe=''))

        return data

    def save_overlay(self, build_id, destination_directory) -> pathlib.Path:
        """Fetch, validate, and save an overlay build order as JSON."""
        bundled_path = self._load_bundled().get(str(build_id))
        if bundled_path is not None:
            destination_directory = pathlib.Path(destination_directory)
            destination_directory.mkdir(parents=True, exist_ok=True)
            destination = destination_directory / bundled_path.name
            if bundled_path.resolve() != destination.resolve():
                shutil.copy2(bundled_path, destination)
            return destination

        data = self.fetch_overlay(build_id)

        name = data.get('name')
        if not isinstance(name, str) or not name.strip():
            name = f'build-{build_id}'

        safe_name = ''.join(c if (c.isalnum() or c in ' ._-') else '-' for c in name)
        safe_name = safe_name.strip(' .-')
        if not safe_name:
            safe_name = f'build-{build_id}'

        safe_id = ''.join(c if c.isalnum() else '-' for c in str(build_id))

        destination_directory = pathlib.Path(destination_directory)
        destination = destination_directory / f'{safe_name} ({safe_id}).json'

        try:
            destination_directory.mkdir(parents=True, exist_ok=True)
        except OSError as err:
            raise OnlineBuildOrderError(f'Could not create destination directory {destination_directory}: {err}') from err

        payload = json.dumps(data, ensure_ascii=False, indent=2)

        temp_file = None
        try:
            fd, temp_name = tempfile.mkstemp(
                prefix='.' + destination.name + '.', suffix='.tmp', dir=str(destination_directory)
            )
            temp_file = pathlib.Path(temp_name)
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                handle.write(payload)
            temp_file.replace(destination)
            temp_file = None
        except OSError as err:
            raise OnlineBuildOrderError(f'Could not write overlay file {destination}: {err}') from err
        finally:
            if temp_file is not None:
                try:
                    temp_file.unlink()
                except OSError:
                    pass

        return destination
