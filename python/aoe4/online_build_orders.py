"""Small synchronous client for the public AoE4 Guides build-order API.

Read-only, standard library only. No polling or background refresh: a call
happens only when the caller asks for it.
"""

import json
import os
import pathlib
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from aoe4.aoe4_build_order import check_valid_aoe4_build_order

LIST_ENDPOINT = 'https://aoe4-guides-api-7h2vti5ckq-ey.a.run.app/builds'
OVERLAY_ENDPOINT = 'https://aoe4guides.com/api/builds/{build_id}?overlay=true'
SOURCE_URL = 'https://aoe4guides.com/builds/{build_id}'


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


class Aoe4GuidesClient:
    """Synchronous client for the public read-only AoE4 Guides API."""

    def __init__(self, opener: Optional[Callable] = None, timeout_seconds: float = 10.0):
        self._timeout_seconds = timeout_seconds
        self._opener = (
            opener
            if opener is not None
            else lambda url: urllib.request.urlopen(url, timeout=self._timeout_seconds)
        )

    def _get_json(self, url: str):
        """Fetch and decode a JSON payload, raising OnlineBuildOrderError."""
        try:
            response = self._opener(url)
            try:
                raw = response.read()
            finally:
                close = getattr(response, 'close', None)
                if close is not None:
                    close()
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

    def list_latest(self, civilization_code: Optional[str] = None) -> Tuple[OnlineBuildSummary, ...]:
        """List the latest build orders, optionally filtered by civilization code."""
        url = LIST_ENDPOINT
        if civilization_code is not None:
            url = url + '?civ=' + urllib.parse.quote(str(civilization_code), safe='')

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
            author = item.get('author')
            season = item.get('season')

            for field_name, value in (('id', build_id), ('title', title), ('civ', civ), ('author', author)):
                if not isinstance(value, (str, int)) or isinstance(value, bool):
                    raise OnlineBuildOrderError(
                        f'List item {index} from {url} has missing or invalid "{field_name}".'
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
                )
            )

        return tuple(summaries)

    def fetch_overlay(self, build_id) -> dict:
        """Fetch the overlay-compatible build order data for a build ID."""
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
