"""
test_ocr_frame_capture.py — Tests unitaires de ocr/vision/frame_capture.py
Phase 7 — Bot Poker Académique

Utilise des images PIL synthétiques (générées en mémoire) plutôt qu'un
vrai écran : MssFrameGrabber lui-même n'est pas testé ici (nécessite un
affichage réel), mais FrameGrabber est un Protocol — un faux grabber
suffit à tester tout le reste (découpage, normalisation, assemblage en
TableFrame) de façon fiable et reproductible.

Placement : tests/
Commande   : depuis la racine du projet
    pytest tests/test_ocr_frame_capture.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from PIL import Image

from ocr.vision.frame_capture import (
    TableFrame, capture_table_frame, crop_region, normalize_for_template_matching,
)
from ocr.vision.region_config import FractionalRegion, RegionConfig
from ocr.vision.window_anchor import BoundingBox, StaticWindowLocator


def solid_image(width: int, height: int, color) -> Image.Image:
    return Image.new("RGB", (width, height), color)


class FakeFrameGrabber:
    """FrameGrabber de test : renvoie une image fournie à l'avance, sans écran réel."""

    def __init__(self, image: Image.Image) -> None:
        self.image = image
        self.calls = []

    def grab(self, bbox: BoundingBox) -> Image.Image:
        self.calls.append(bbox)
        return self.image


# =============================================================================
# crop_region
# =============================================================================

class TestCropRegion:

    def test_crop_at_frame_origin(self):
        frame = solid_image(200, 200, (0, 0, 0))
        origin = BoundingBox(left=0, top=0, width=200, height=200)
        cropped = crop_region(frame, (10, 20, 30, 40), origin)
        assert cropped.size == (30, 40)

    def test_crop_offset_by_frame_origin(self):
        # frame_origin non nul (fenêtre pas en (0,0) de l'écran) : le
        # découpage doit être relatif à l'image, pas à l'écran.
        frame = solid_image(200, 200, (0, 0, 0))
        origin = BoundingBox(left=500, top=300, width=200, height=200)
        pixel_rect = (500 + 10, 300 + 20, 30, 40)   # coordonnées écran absolues
        cropped = crop_region(frame, pixel_rect, origin)
        assert cropped.size == (30, 40)

    def test_crop_content_matches_expected_region(self):
        frame = Image.new("RGB", (100, 100), (0, 0, 0))
        # Peindre un carré blanc à un endroit connu.
        for x in range(40, 60):
            for y in range(40, 60):
                frame.putpixel((x, y), (255, 255, 255))
        origin = BoundingBox(left=0, top=0, width=100, height=100)

        cropped = crop_region(frame, (40, 40, 20, 20), origin)
        arr = np.array(cropped)
        assert arr.mean() > 250   # entièrement blanc

        cropped_elsewhere = crop_region(frame, (0, 0, 20, 20), origin)
        arr2 = np.array(cropped_elsewhere)
        assert arr2.mean() < 5    # entièrement noir


# =============================================================================
# normalize_for_template_matching
# =============================================================================

class TestNormalizeForTemplateMatching:

    def test_output_shape_matches_canonical_size(self):
        region = solid_image(37, 52, (10, 20, 30))   # taille arbitraire, non canonique
        result = normalize_for_template_matching(region, canonical_size=(64, 96))
        assert result.shape == (96, 64, 3)   # numpy : (height, width, channels)

    def test_already_canonical_size_unchanged_dimensions(self):
        region = solid_image(64, 96, (10, 20, 30))
        result = normalize_for_template_matching(region, canonical_size=(64, 96))
        assert result.shape == (96, 64, 3)

    def test_two_different_input_sizes_produce_same_output_shape(self):
        # Le point central de la fonction : une carte lue à 2 tailles de
        # fenêtre différentes doit produire la même taille de sortie,
        # comparable au même jeu de templates.
        small_capture = solid_image(30, 42, (200, 0, 0))
        large_capture = solid_image(48, 67, (200, 0, 0))
        out_small = normalize_for_template_matching(small_capture, canonical_size=(64, 90))
        out_large = normalize_for_template_matching(large_capture, canonical_size=(64, 90))
        assert out_small.shape == out_large.shape


# =============================================================================
# capture_table_frame — assemblage complet
# =============================================================================

class TestCaptureTableFrame:

    def _regions(self) -> RegionConfig:
        return RegionConfig({
            "pot": FractionalRegion(0.45, 0.45, 0.1, 0.05),
            "hero_card_1": FractionalRegion(0.40, 0.85, 0.05, 0.10),
        })

    def test_returns_none_if_window_not_found(self):
        locator = StaticWindowLocator(None)
        grabber = FakeFrameGrabber(solid_image(10, 10, (0, 0, 0)))
        result = capture_table_frame(locator, grabber, self._regions())
        assert result is None
        assert grabber.calls == []   # ne doit même pas essayer de capturer

    def test_produces_a_crop_per_region(self):
        bbox = BoundingBox(left=0, top=0, width=1000, height=1000)
        locator = StaticWindowLocator(bbox)
        grabber = FakeFrameGrabber(solid_image(1000, 1000, (128, 128, 128)))

        frame = capture_table_frame(locator, grabber, self._regions())
        assert isinstance(frame, TableFrame)
        assert set(frame.crops.keys()) == {"pot", "hero_card_1"}

    def test_crop_sizes_match_resolved_pixel_rects(self):
        bbox = BoundingBox(left=0, top=0, width=1000, height=1000)
        locator = StaticWindowLocator(bbox)
        grabber = FakeFrameGrabber(solid_image(1000, 1000, (128, 128, 128)))
        regions = self._regions()

        frame = capture_table_frame(locator, grabber, regions)
        expected = regions.resolve("pot", bbox)
        assert frame.crop("pot").size == (expected[2], expected[3])

    def test_grabber_called_with_current_bbox(self):
        bbox = BoundingBox(left=200, top=100, width=1280, height=720)
        locator = StaticWindowLocator(bbox)
        grabber = FakeFrameGrabber(solid_image(1280, 720, (0, 0, 0)))

        capture_table_frame(locator, grabber, self._regions())
        assert grabber.calls == [bbox]

    def test_survives_window_resize_between_captures(self):
        locator = StaticWindowLocator(BoundingBox(0, 0, 1280, 720))
        regions = self._regions()

        grabber1 = FakeFrameGrabber(solid_image(1280, 720, (0, 0, 0)))
        frame1 = capture_table_frame(locator, grabber1, regions)
        assert frame1 is not None

        locator.set_bbox(BoundingBox(0, 0, 1920, 1080))   # fenêtre redimensionnée
        grabber2 = FakeFrameGrabber(solid_image(1920, 1080, (0, 0, 0)))
        frame2 = capture_table_frame(locator, grabber2, regions)
        assert frame2 is not None
        # La zone "pot" doit être plus grande en pixels après agrandissement.
        assert frame2.crop("pot").size[0] > frame1.crop("pot").size[0]
