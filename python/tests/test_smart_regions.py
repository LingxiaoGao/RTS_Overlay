import unittest
from types import MappingProxyType
from typing import Mapping

from aoe4.smart_capture import CapturedFrame, CaptureRegion
from aoe4.smart_regions import HudRegionSet, crop_frame
from aoe4.smart_regions import scale_regions
from aoe4.smart_runtime import FIELD_NAMES


def _pixels(width: int, height: int, base: int = 0) -> bytes:
    """Unique byte per pixel index in BGRA order."""
    out = bytearray()
    for index in range(width * height):
        value = (base + index) % 256
        out.extend((value, value, value, 255))
    return bytes(out)


def _frame(frame_id=1, captured_at=10.5, region=None, base=0) -> CapturedFrame:
    region = region or CaptureRegion(0, 0, 3, 3)
    return CapturedFrame(
        frame_id=frame_id,
        captured_at=captured_at,
        width=region.width,
        height=region.height,
        pixels=_pixels(region.width, region.height, base),
        region=region,
    )


def _expected_rows(frame, region) -> bytes:
    source = frame.region
    stride = frame.width * 4
    x_offset = (region.left - source.left) * 4
    row_bytes = region.width * 4
    y_start = region.top - source.top
    chunks = []
    for row in range(y_start, y_start + region.height):
        start = row * stride + x_offset
        chunks.append(frame.pixels[start:start + row_bytes])
    return b"".join(chunks)


class CropFrameTests(unittest.TestCase):
    def test_full_frame_crop(self):
        frame = _frame(base=7)
        region = CaptureRegion(0, 0, 3, 3)
        result = crop_frame(frame, region)
        self.assertEqual(result.pixels, frame.pixels)
        self.assertEqual(result.width, 3)
        self.assertEqual(result.height, 3)
        self.assertEqual(result.frame_id, frame.frame_id)
        self.assertEqual(result.captured_at, frame.captured_at)
        self.assertEqual(result.region, region)

    def test_one_pixel_crop(self):
        frame = _frame(base=11)
        region = CaptureRegion(2, 1, 1, 1)
        result = crop_frame(frame, region)
        index = 1 * 3 + 2
        expected = frame.pixels[index * 4:index * 4 + 4]
        self.assertEqual(result.pixels, expected)
        self.assertEqual(result.width, 1)
        self.assertEqual(result.height, 1)
        self.assertEqual(result.region, region)
        self.assertEqual(result.frame_id, frame.frame_id)
        self.assertEqual(result.captured_at, frame.captured_at)

    def test_multirow_offset_crop_with_parent_origin(self):
        parent = CaptureRegion(10, 20, 4, 4)
        frame = _frame(frame_id=99, captured_at=3.25, region=parent, base=40)
        region = CaptureRegion(11, 21, 2, 3)
        result = crop_frame(frame, region)
        self.assertEqual(result.pixels, _expected_rows(frame, region))
        self.assertEqual(result.width, 2)
        self.assertEqual(result.height, 3)
        self.assertEqual(result.region, region)
        self.assertEqual(result.frame_id, 99)
        self.assertEqual(result.captured_at, 3.25)
        self.assertEqual(len(result.pixels), 2 * 3 * 4)

    def test_row_order_preserved(self):
        parent = CaptureRegion(5, 5, 4, 4)
        frame = _frame(region=parent, base=0)
        region = CaptureRegion(5, 6, 4, 2)
        result = crop_frame(frame, region)
        first_row = frame.pixels[1 * 16:2 * 16]
        second_row = frame.pixels[2 * 16:3 * 16]
        self.assertEqual(result.pixels, first_row + second_row)

    def test_exact_containment_boundaries(self):
        parent = CaptureRegion(2, 3, 3, 3)
        frame = _frame(region=parent)
        result = crop_frame(frame, CaptureRegion(2, 3, 3, 3))
        self.assertEqual(result.pixels, frame.pixels)
        result = crop_frame(frame, CaptureRegion(4, 5, 1, 1))
        self.assertEqual(result.width, 1)

    def test_out_of_bounds_directions(self):
        parent = CaptureRegion(2, 3, 4, 4)
        frame = _frame(region=parent)
        bad = [
            CaptureRegion(1, 3, 2, 2),   # left
            CaptureRegion(2, 2, 2, 2),   # top
            CaptureRegion(5, 3, 2, 2),   # right
            CaptureRegion(2, 6, 2, 2),   # bottom
            CaptureRegion(1, 1, 10, 10), # all sides
        ]
        for region in bad:
            with self.subTest(region=region):
                with self.assertRaises(ValueError):
                    crop_frame(frame, region)

    def test_wrong_argument_types(self):
        frame = _frame()
        with self.assertRaises(TypeError):
            crop_frame(object(), CaptureRegion(0, 0, 1, 1))
        with self.assertRaises(TypeError):
            crop_frame(frame, (0, 0, 1, 1))

    def test_malformed_short_buffer(self):
        region = CaptureRegion(0, 0, 2, 2)
        frame = CapturedFrame(
            frame_id=1,
            captured_at=1.0,
            width=2,
            height=2,
            pixels=b"\x00" * 8,
            region=region,
        )
        with self.assertRaises(ValueError):
            crop_frame(frame, region)

    def test_malformed_long_buffer(self):
        region = CaptureRegion(0, 0, 2, 2)
        frame = CapturedFrame(
            frame_id=1,
            captured_at=1.0,
            width=2,
            height=2,
            pixels=b"\x00" * 64,
            region=region,
        )
        with self.assertRaises(ValueError):
            crop_frame(frame, region)


class HudRegionSetValidationTests(unittest.TestCase):
    def _regions(self):
        return {
            name: CaptureRegion(0, 0, 1, 1) for name in FIELD_NAMES[:2]
        }

    def test_rejects_non_mapping(self):
        with self.assertRaises(TypeError):
            HudRegionSet([("a", CaptureRegion(0, 0, 1, 1))])

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            HudRegionSet({})

    def test_rejects_non_string_key(self):
        regions = self._regions()
        regions[1] = CaptureRegion(0, 0, 1, 1)
        with self.assertRaises(TypeError):
            HudRegionSet(regions)

    def test_rejects_unknown_key(self):
        regions = self._regions()
        regions["not_a_field"] = CaptureRegion(0, 0, 1, 1)
        with self.assertRaises(ValueError):
            HudRegionSet(regions)

    def test_rejects_non_region_value(self):
        regions = self._regions()
        regions[FIELD_NAMES[0]] = (0, 0, 1, 1)
        with self.assertRaises(TypeError):
            HudRegionSet(regions)

    def test_defensive_copy(self):
        regions = self._regions()
        hud = HudRegionSet(regions)
        regions[FIELD_NAMES[0]] = CaptureRegion(0, 0, 2, 2)
        self.assertEqual(hud.regions[FIELD_NAMES[0]], CaptureRegion(0, 0, 1, 1))

    def test_frozen_instance(self):
        hud = HudRegionSet(self._regions())
        with self.assertRaises(Exception):
            hud._regions = MappingProxyType({})

    def test_deterministic_field_ordering(self):
        names = FIELD_NAMES
        forward = HudRegionSet(
            {name: CaptureRegion(0, 0, 1, 1) for name in names}
        )
        backward = HudRegionSet(
            {name: CaptureRegion(0, 0, 1, 1) for name in reversed(names)}
        )
        self.assertEqual(list(forward.regions), list(names))
        self.assertEqual(list(backward.regions), list(names))
        self.assertEqual(forward.regions, backward.regions)

    def test_regions_mapping_immutable(self):
        hud = HudRegionSet(self._regions())
        with self.assertRaises(TypeError):
            hud.regions[FIELD_NAMES[0]] = CaptureRegion(0, 0, 2, 2)


class HudRegionSetCropTests(unittest.TestCase):
    def test_crop_returns_mapping(self):
        frame = _frame(region=CaptureRegion(0, 0, 3, 3))
        hud = HudRegionSet(
            {
                FIELD_NAMES[0]: CaptureRegion(0, 0, 2, 2),
                FIELD_NAMES[1]: CaptureRegion(1, 1, 2, 2),
            }
        )
        result = hud.crop(frame)
        self.assertIsInstance(result, Mapping)
        self.assertEqual(set(result), {FIELD_NAMES[0], FIELD_NAMES[1]})
        for name, region in hud.regions.items():
            self.assertEqual(result[name].pixels, _expected_rows(frame, region))
            self.assertEqual(result[name].region, region)
            self.assertEqual(result[name].frame_id, frame.frame_id)
            self.assertEqual(result[name].captured_at, frame.captured_at)

    def test_crop_wrong_frame_type(self):
        hud = HudRegionSet({FIELD_NAMES[0]: CaptureRegion(0, 0, 1, 1)})
        with self.assertRaises(TypeError):
            hud.crop("not a frame")

    def test_crop_invalid_region_fails_without_partial_output(self):
        parent = CaptureRegion(0, 0, 3, 3)
        frame = _frame(region=parent)
        hud = HudRegionSet(
            {
                FIELD_NAMES[0]: CaptureRegion(0, 0, 1, 1),
                FIELD_NAMES[1]: CaptureRegion(2, 2, 5, 5),
            }
        )
        with self.assertRaises(ValueError):
            hud.crop(frame)


class ScaleRegionsTests(unittest.TestCase):
    def _hud(self, regions):
        return HudRegionSet(regions)

    def test_identity_scaling(self):
        ref = CaptureRegion(0, 0, 1920, 1080)
        target = CaptureRegion(0, 0, 1920, 1080)
        regions = {
            FIELD_NAMES[0]: CaptureRegion(10, 10, 100, 50),
            FIELD_NAMES[1]: CaptureRegion(500, 200, 200, 100),
        }
        scaled = scale_regions(self._hud(regions), ref, target)
        for name, region in regions.items():
            self.assertEqual(scaled.regions[name], region)

    def test_double_scale_1920_to_3840(self):
        ref = CaptureRegion(0, 0, 1920, 1080)
        target = CaptureRegion(0, 0, 3840, 2160)
        regions = {FIELD_NAMES[0]: CaptureRegion(10, 20, 100, 50)}
        scaled = scale_regions(self._hud(regions), ref, target)
        self.assertEqual(scaled.regions[FIELD_NAMES[0]], CaptureRegion(20, 40, 200, 100))

    def test_nonzero_reference_and_target_origins(self):
        ref = CaptureRegion(100, 200, 800, 600)
        target = CaptureRegion(50, 60, 1600, 1200)
        regions = {FIELD_NAMES[0]: CaptureRegion(300, 500, 100, 50)}
        scaled = scale_regions(self._hud(regions), ref, target)
        # Relative to ref origin: (200, 300) size (100, 50).
        # Doubled to (400, 600) size (200, 100), offset by target origin.
        self.assertEqual(
            scaled.regions[FIELD_NAMES[0]],
            CaptureRegion(450, 660, 200, 100),
        )

    def test_scaled_regions_crop_target_frame(self):
        ref = CaptureRegion(0, 0, 4, 4)
        target = CaptureRegion(0, 0, 8, 8)
        regions = {
            FIELD_NAMES[0]: CaptureRegion(0, 0, 2, 2),
            FIELD_NAMES[1]: CaptureRegion(2, 2, 2, 2),
        }
        hud = self._hud(regions)
        scaled = scale_regions(hud, ref, target)
        frame = _frame(region=CaptureRegion(0, 0, 8, 8))
        cropped = scaled.crop(frame)
        self.assertEqual(
            cropped[FIELD_NAMES[0]].region, CaptureRegion(0, 0, 4, 4)
        )
        self.assertEqual(
            cropped[FIELD_NAMES[1]].region, CaptureRegion(4, 4, 4, 4)
        )

if __name__ == "__main__":
    unittest.main()
