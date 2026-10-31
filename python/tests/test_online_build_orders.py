import io
import json
import pathlib
import tempfile
import unittest
import urllib.parse
from datetime import datetime, timedelta, timezone

from aoe4.online_build_orders import (
    CIVILIZATION_LABELS,
    LIST_ENDPOINT,
    OVERLAY_ENDPOINT,
    SOURCE_URL,
    Aoe4GuidesClient,
    OnlineBuildOrderError,
    OnlineBuildSummary,
)


def _valid_overlay(name='Test Build'):
    return {
        'name': name,
        'civilization': 'English',
        'build_order': [
            {
                'population_count': 6,
                'villager_count': 6,
                'age': 1,
                'resources': {
                    'food': 0,
                    'wood': 0,
                    'gold': 0,
                    'stone': 0,
                },
                'notes': ['note'],
            }
        ],
    }


class _FakeResponse:
    def __init__(self, payload, encoding='utf-8'):
        if isinstance(payload, str):
            payload = payload.encode(encoding)
        self._buffer = io.BytesIO(payload)
        self.closed = False

    def read(self):
        return self._buffer.read()

    def close(self):
        self.closed = True


class _RecordingOpener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.urls = []

    def __call__(self, url):
        self.urls.append(url)
        if not self.responses:
            raise AssertionError('Unexpected extra call to opener')
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class ListLatestTests(unittest.TestCase):
    def test_list_url_and_summary_mapping(self):
        payload = [
            {'id': 'a1', 'title': 'First', 'civ': 'English', 'author': 'Ann', 'season': 'S1'},
            {'id': 'b2', 'title': 'Second', 'civ': 'French', 'author': 'Bob'},
        ]
        opener = _RecordingOpener([_FakeResponse(json.dumps(payload))])
        client = Aoe4GuidesClient(opener=opener)

        summaries = client.list_latest()

        self.assertEqual(opener.urls, [LIST_ENDPOINT])
        self.assertEqual(
            summaries,
            (
                OnlineBuildSummary('a1', 'First', 'English', 'Ann', 'S1'),
                OnlineBuildSummary('b2', 'Second', 'French', 'Bob', None),
            ),
        )

    def test_civilization_query_encoding(self):
        opener = _RecordingOpener([_FakeResponse('[]')])
        client = Aoe4GuidesClient(opener=opener)

        result = client.list_latest(civilization_code='jeanne d\'arc & co')

        self.assertEqual(result, ())
        expected = LIST_ENDPOINT + '?civ=' + urllib.parse.quote('jeanne d\'arc & co', safe='')
        self.assertEqual(opener.urls, [expected])

    def test_popularity_order_query(self):
        opener = _RecordingOpener([_FakeResponse('[]')])
        client = Aoe4GuidesClient(opener=opener)

        client.list_latest(civilization_code='ENG', order_by='views')

        self.assertEqual(opener.urls, [LIST_ENDPOINT + '?civ=ENG&orderBy=views'])

    def test_timestamp_and_views_mapping(self):
        updated = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        payload = [{
            'id': 'a1', 'title': 'First', 'civ': 'ENG', 'author': 'Ann',
            'timeUpdated': {'_seconds': int(updated.timestamp())}, 'views': 42,
        }]
        client = Aoe4GuidesClient(
            opener=_RecordingOpener([_FakeResponse(json.dumps(payload))])
        )

        summary = client.list_latest()[0]

        self.assertEqual(summary.updated, updated)
        self.assertEqual(summary.views, 42)

    def test_missing_author_uses_creator_name_or_empty_string(self):
        payload = [
            {'id': 'a1', 'title': 'First', 'civ': 'ENG', 'creatorName': 'Creator'},
            {'id': 'a2', 'title': 'Second', 'civ': 'ENG'},
        ]
        client = Aoe4GuidesClient(
            opener=_RecordingOpener([_FakeResponse(json.dumps(payload))])
        )

        summaries = client.list_latest()

        self.assertEqual([item.author for item in summaries], ['Creator', ''])

    def test_malformed_timestamp_raises(self):
        payload = [{
            'id': 'a1', 'title': 'First', 'civ': 'ENG', 'author': 'Ann',
            'timeUpdated': {'_seconds': 'bad'},
        }]
        client = Aoe4GuidesClient(
            opener=_RecordingOpener([_FakeResponse(json.dumps(payload))])
        )
        with self.assertRaises(OnlineBuildOrderError):
            client.list_latest()

    def test_network_error_raises(self):
        opener = _RecordingOpener([ConnectionError('boom')])
        client = Aoe4GuidesClient(opener=opener)
        with self.assertRaises(OnlineBuildOrderError):
            client.list_latest()

    def test_invalid_json_raises(self):
        opener = _RecordingOpener([_FakeResponse('not json')])
        client = Aoe4GuidesClient(opener=opener)
        with self.assertRaises(OnlineBuildOrderError):
            client.list_latest()

    def test_invalid_shape_raises(self):
        opener = _RecordingOpener([_FakeResponse(json.dumps({'not': 'a list'}))])
        client = Aoe4GuidesClient(opener=opener)
        with self.assertRaises(OnlineBuildOrderError):
            client.list_latest()

    def test_invalid_utf8_raises(self):
        opener = _RecordingOpener([_FakeResponse(b'\xff\xfe\x00')])
        client = Aoe4GuidesClient(opener=opener)
        with self.assertRaises(OnlineBuildOrderError):
            client.list_latest()


class _CivilizationClient(Aoe4GuidesClient):
    def __init__(self, values=None, failures=None):
        self.values = values or {}
        self.failures = failures or {}

    def list_latest(self, civilization_code=None, order_by=None):
        self.order_by = order_by
        if civilization_code in self.failures:
            raise self.failures[civilization_code]
        return tuple(self.values.get(civilization_code, ()))


def _summary(build_id, civilization, updated, views=0, title=None):
    return OnlineBuildSummary(
        build_id=build_id,
        title=title or build_id,
        civilization=civilization,
        author='author',
        updated=updated,
        views=views,
    )


class ListPopularRecentTests(unittest.TestCase):
    NOW = datetime(2026, 10, 31, tzinfo=timezone.utc)

    def test_prefers_last_year_then_uses_older_fallback(self):
        code = CIVILIZATION_LABELS[0][0]
        values = {
            code: (
                _summary('boundary', code, self.NOW - timedelta(days=365)),
                _summary('recent', code, self.NOW - timedelta(days=10)),
                _summary('older-high', code, self.NOW - timedelta(days=500), views=100),
                _summary('older-low', code, self.NOW - timedelta(days=600), views=1),
                _summary('future', code, self.NOW + timedelta(seconds=1)),
            )
        }

        client = _CivilizationClient(values)
        result = client.list_popular_recent(now=self.NOW)

        self.assertEqual(
            [item.build_id for item in result],
            ['future', 'recent', 'boundary'],
        )
        self.assertEqual(client.order_by, 'views')

    def test_sorts_by_views_and_caps_three(self):
        code = CIVILIZATION_LABELS[0][0]
        values = {code: tuple(
            _summary(f'item-{views}', code, self.NOW, views=views)
            for views in (1, 50, 3, 100, 20)
        )}

        result = _CivilizationClient(values).list_popular_recent(now=self.NOW)

        self.assertEqual(
            [item.build_id for item in result],
            ['item-100', 'item-50', 'item-20'],
        )

    def test_preserves_civilization_mapping_order(self):
        first, second = CIVILIZATION_LABELS[:2]
        values = {
            second[0]: (_summary('second', second[0], self.NOW),),
            first[0]: (_summary('first', first[0], self.NOW),),
        }

        result = _CivilizationClient(values).list_popular_recent(now=self.NOW)

        self.assertEqual([item.build_id for item in result], ['first', 'second'])

    def test_failure_names_civilization(self):
        code = CIVILIZATION_LABELS[0][0]
        client = _CivilizationClient(failures={code: ConnectionError('offline')})

        with self.assertRaisesRegex(OnlineBuildOrderError, code):
            client.list_popular_recent(now=self.NOW)

    def test_complete_bundled_snapshot_works_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            for code, _ in CIVILIZATION_LABELS:
                for index in range(3):
                    build_id = f'{code}-{index}'
                    data = _valid_overlay(name=build_id)
                    data['_aoe4_guides'] = {
                        'id': build_id,
                        'civilization_code': code,
                        'updated': self.NOW.isoformat(),
                        'views_at_snapshot': index,
                    }
                    (directory / f'aoe4guides-{build_id}.json').write_text(
                        json.dumps(data), encoding='utf-8'
                    )

            client = Aoe4GuidesClient(
                opener=lambda _url: (_ for _ in ()).throw(AssertionError('network used')),
                bundled_directory=directory,
            )
            result = client.list_popular_recent(now=self.NOW)

            self.assertEqual(len(result), len(CIVILIZATION_LABELS) * 3)
            self.assertEqual(client.fetch_overlay('ABB-0')['name'], 'ABB-0')


class FetchOverlayTests(unittest.TestCase):
    def test_overlay_url_and_validation(self):
        overlay = _valid_overlay()
        opener = _RecordingOpener([_FakeResponse(json.dumps(overlay))])
        client = Aoe4GuidesClient(opener=opener)

        result = client.fetch_overlay('abc123')

        self.assertEqual(opener.urls, [OVERLAY_ENDPOINT.format(build_id='abc123')])
        self.assertEqual(result['name'], 'Test Build')

    def test_canonical_source_fallback(self):
        overlay = _valid_overlay()
        opener = _RecordingOpener([_FakeResponse(json.dumps(overlay))])
        client = Aoe4GuidesClient(opener=opener)

        result = client.fetch_overlay('abc123')

        self.assertEqual(result['source'], SOURCE_URL.format(build_id='abc123'))

    def test_existing_source_not_overwritten(self):
        overlay = _valid_overlay()
        overlay['source'] = 'https://example.com/custom'
        opener = _RecordingOpener([_FakeResponse(json.dumps(overlay))])
        client = Aoe4GuidesClient(opener=opener)

        result = client.fetch_overlay('abc123')

        self.assertEqual(result['source'], 'https://example.com/custom')

    def test_invalid_data_raises(self):
        overlay = {'name': 'Broken', 'build_order': []}
        opener = _RecordingOpener([_FakeResponse(json.dumps(overlay))])
        client = Aoe4GuidesClient(opener=opener)
        with self.assertRaises(OnlineBuildOrderError):
            client.fetch_overlay('abc123')

    def test_non_object_response_raises(self):
        opener = _RecordingOpener([_FakeResponse('[]')])
        client = Aoe4GuidesClient(opener=opener)
        with self.assertRaises(OnlineBuildOrderError):
            client.fetch_overlay('abc123')


class SaveOverlayTests(unittest.TestCase):
    def test_safe_filename_and_atomic_save(self):
        overlay = _valid_overlay(name='My / weird: name')
        opener = _RecordingOpener([_FakeResponse(json.dumps(overlay))])
        client = Aoe4GuidesClient(opener=opener)

        with tempfile.TemporaryDirectory() as tmp:
            destination_dir = pathlib.Path(tmp) / 'nested' / 'dir'
            saved = client.save_overlay('abc 123', destination_dir)

            self.assertTrue(saved.exists())
            self.assertEqual(saved.parent, destination_dir)
            self.assertNotIn('/', saved.name)
            self.assertNotIn(':', saved.name)
            self.assertIn('abc-123', saved.stem)
            self.assertEqual(saved.suffix, '.json')

            data = json.loads(saved.read_text(encoding='utf-8'))
            self.assertEqual(data['name'], 'My / weird: name')
            self.assertEqual(data['source'], SOURCE_URL.format(build_id='abc%20123'))

            leftovers = [p for p in destination_dir.iterdir() if p != saved]
            self.assertEqual(leftovers, [])

    def test_failed_replace_preserves_existing_destination(self):
        overlay = _valid_overlay(name='Same Name')
        opener = _RecordingOpener([_FakeResponse(json.dumps(overlay))])
        client = Aoe4GuidesClient(opener=opener)

        with tempfile.TemporaryDirectory() as tmp:
            destination_dir = pathlib.Path(tmp)
            destination = destination_dir / 'Same Name (b1).json'
            destination.write_text('original content', encoding='utf-8')

            original_replace = pathlib.Path.replace

            def _failing_replace(self, target):
                raise OSError('simulated replace failure')

            pathlib.Path.replace = _failing_replace
            try:
                with self.assertRaises(OnlineBuildOrderError):
                    client.save_overlay('b1', destination_dir)
            finally:
                pathlib.Path.replace = original_replace

            self.assertEqual(destination.read_text(encoding='utf-8'), 'original content')
            leftovers = [p for p in destination_dir.iterdir() if p != destination]
            self.assertEqual(leftovers, [])


if __name__ == '__main__':
    unittest.main()
