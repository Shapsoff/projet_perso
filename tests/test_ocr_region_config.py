"""
test_ocr_region_config.py — Tests unitaires de ocr/vision/region_config.py
Phase 7 — Bot Poker Académique

Placement : tests/
Commande   : depuis la racine du projet
    pytest tests/test_ocr_region_config.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from ocr.vision.region_config import FractionalRegion, RegionConfig
from ocr.vision.window_anchor import BoundingBox


class TestFractionalRegion:

    def test_to_pixels_basic(self):
        region = FractionalRegion(x=0.1, y=0.2, w=0.05, h=0.1)
        bbox = BoundingBox(left=0, top=0, width=1000, height=1000)
        assert region.to_pixels(bbox) == (100, 200, 50, 100)

    def test_to_pixels_accounts_for_bbox_origin(self):
        region = FractionalRegion(x=0.1, y=0.2, w=0.05, h=0.1)
        bbox = BoundingBox(left=500, top=300, width=1000, height=1000)
        left, top, w, h = region.to_pixels(bbox)
        assert left == 500 + 100
        assert top == 300 + 200
        assert (w, h) == (50, 100)

    def test_from_pixels_roundtrip(self):
        bbox = BoundingBox(left=0, top=0, width=1280, height=720)
        original_rect = (128, 144, 64, 72)   # 10%, 20%, 5%, 10%
        region = FractionalRegion.from_pixels(original_rect, bbox)
        assert region.x == pytest.approx(0.1)
        assert region.y == pytest.approx(0.2)
        assert region.w == pytest.approx(0.05)
        assert region.h == pytest.approx(0.1)
        assert region.to_pixels(bbox) == original_rect

    def test_rejects_out_of_range_values(self):
        with pytest.raises(ValueError):
            FractionalRegion(x=1.5, y=0.0, w=0.1, h=0.1)
        with pytest.raises(ValueError):
            FractionalRegion(x=-0.1, y=0.0, w=0.1, h=0.1)

    def test_rejects_region_extending_past_right_edge(self):
        with pytest.raises(ValueError):
            FractionalRegion(x=0.95, y=0.0, w=0.1, h=0.1)

    def test_rejects_region_extending_past_bottom_edge(self):
        with pytest.raises(ValueError):
            FractionalRegion(x=0.0, y=0.95, w=0.1, h=0.1)

    def test_full_window_region_allowed(self):
        FractionalRegion(x=0.0, y=0.0, w=1.0, h=1.0)  # ne doit pas lever

    def test_minimum_one_pixel_even_for_tiny_fraction(self):
        region = FractionalRegion(x=0.0, y=0.0, w=0.0001, h=0.0001)
        bbox = BoundingBox(left=0, top=0, width=100, height=100)
        _, _, w, h = region.to_pixels(bbox)
        assert w >= 1 and h >= 1


# =============================================================================
# Le comportement "responsive" : ce que la calibration fractionnelle apporte
# =============================================================================

class TestResponsiveBehavior:
    """
    Démontre directement la propriété demandée : une zone calibrée UNE
    fois reste correcte quand la fenêtre est déplacée ET/OU redimensionnée
    par la suite, sans recalibration — exactement comme un layout CSS en %
    s'adapte à la taille du viewport.
    """

    def test_region_follows_window_move(self):
        region = FractionalRegion(x=0.5, y=0.5, w=0.1, h=0.1)   # centre de la table, ex: le pot
        at_origin = region.to_pixels(BoundingBox(left=0, top=0, width=1280, height=720))
        moved_bbox = BoundingBox(left=300, top=150, width=1280, height=720)  # fenêtre déplacée, PAS redimensionnée
        after_move = region.to_pixels(moved_bbox)

        assert after_move[0] == at_origin[0] + 300
        assert after_move[1] == at_origin[1] + 150
        assert after_move[2:] == at_origin[2:]   # taille inchangée

    def test_region_scales_with_window_resize(self):
        region = FractionalRegion(x=0.5, y=0.5, w=0.1, h=0.1)
        small = region.to_pixels(BoundingBox(left=0, top=0, width=1280, height=720))
        large = region.to_pixels(BoundingBox(left=0, top=0, width=1920, height=1080))  # 1.5x

        scale = 1920 / 1280
        assert large[0] == pytest.approx(small[0] * scale, abs=2)
        assert large[2] == pytest.approx(small[2] * scale, abs=2)   # largeur de la zone aussi mise à l'échelle

    def test_region_survives_both_move_and_resize_together(self):
        region = FractionalRegion(x=0.2, y=0.3, w=0.05, h=0.08)
        ref_bbox = BoundingBox(left=100, top=50, width=1280, height=720)
        live_bbox = BoundingBox(left=847, top=212, width=1600, height=900)  # déplacée + agrandie

        left, top, w, h = region.to_pixels(live_bbox)
        # La zone doit rester DANS la fenêtre actuelle, à la bonne fraction.
        assert live_bbox.left <= left <= live_bbox.right
        assert live_bbox.top <= top <= live_bbox.bottom
        assert (left - live_bbox.left) / live_bbox.width == pytest.approx(0.2, abs=0.001)
        assert (top - live_bbox.top) / live_bbox.height == pytest.approx(0.3, abs=0.001)


# =============================================================================
# RegionConfig
# =============================================================================

class TestRegionConfig:

    def _sample(self) -> RegionConfig:
        return RegionConfig({
            "pot": FractionalRegion(0.45, 0.45, 0.1, 0.05),
            "hero_card_1": FractionalRegion(0.40, 0.85, 0.05, 0.1),
        })

    def test_names(self):
        cfg = self._sample()
        assert set(cfg.names()) == {"pot", "hero_card_1"}

    def test_len(self):
        assert len(self._sample()) == 2

    def test_contains(self):
        cfg = self._sample()
        assert "pot" in cfg
        assert "board_card_1" not in cfg

    def test_resolve(self):
        cfg = self._sample()
        bbox = BoundingBox(left=0, top=0, width=1000, height=1000)
        assert cfg.resolve("pot", bbox) == (450, 450, 100, 50)

    def test_resolve_unknown_raises(self):
        cfg = self._sample()
        with pytest.raises(KeyError):
            cfg.resolve("unknown_zone", BoundingBox(0, 0, 100, 100))

    def test_resolve_all(self):
        cfg = self._sample()
        bbox = BoundingBox(left=0, top=0, width=1000, height=1000)
        resolved = cfg.resolve_all(bbox)
        assert set(resolved.keys()) == {"pot", "hero_card_1"}

    def test_json_roundtrip(self, tmp_path):
        cfg = self._sample()
        path = tmp_path / "regions.json"
        cfg.to_json(path)
        loaded = RegionConfig.from_json(path)

        bbox = BoundingBox(left=0, top=0, width=1280, height=720)
        assert cfg.resolve_all(bbox) == loaded.resolve_all(bbox)

    def test_json_file_is_human_readable(self, tmp_path):
        cfg = self._sample()
        path = tmp_path / "regions.json"
        cfg.to_json(path)
        content = path.read_text(encoding="utf-8")
        assert "pot" in content
        assert "hero_card_1" in content
