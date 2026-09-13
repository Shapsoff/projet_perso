"""
test_ocr_window_anchor.py — Tests unitaires de ocr/vision/window_anchor.py
Phase 7 — Bot Poker Académique

Win32WindowLocator n'est pas testé ici (nécessite pywin32 + une vraie
fenêtre Windows) — seule sa validation de constructeur, indépendante de
l'OS, l'est. La logique réellement critique (résolution
position/taille) est portée par BoundingBox et FractionalRegion
(cf. test_ocr_region_config.py), testables sans dépendance Windows.

Placement : tests/
Commande   : depuis la racine du projet
    pytest tests/test_ocr_window_anchor.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from ocr.vision.window_anchor import (
    AmbiguousWindowError, BoundingBox, StaticWindowLocator, Win32WindowLocator, select_window,
)


class TestBoundingBox:

    def test_basic_fields(self):
        bbox = BoundingBox(left=100, top=50, width=800, height=600)
        assert bbox.left == 100 and bbox.top == 50
        assert bbox.width == 800 and bbox.height == 600

    def test_right_bottom(self):
        bbox = BoundingBox(left=100, top=50, width=800, height=600)
        assert bbox.right == 900
        assert bbox.bottom == 650

    def test_as_mss_dict(self):
        bbox = BoundingBox(left=10, top=20, width=300, height=400)
        assert bbox.as_mss_dict() == {"left": 10, "top": 20, "width": 300, "height": 400}

    def test_rejects_zero_width(self):
        with pytest.raises(ValueError):
            BoundingBox(left=0, top=0, width=0, height=100)

    def test_rejects_negative_height(self):
        with pytest.raises(ValueError):
            BoundingBox(left=0, top=0, width=100, height=-10)

    def test_immutable(self):
        bbox = BoundingBox(left=0, top=0, width=100, height=100)
        with pytest.raises(Exception):
            bbox.left = 5  # frozen dataclass


class TestStaticWindowLocator:

    def test_returns_configured_bbox(self):
        bbox = BoundingBox(left=0, top=0, width=1280, height=720)
        locator = StaticWindowLocator(bbox)
        assert locator.find_client_bbox() == bbox

    def test_returns_none_by_default_if_unset(self):
        locator = StaticWindowLocator(None)
        assert locator.find_client_bbox() is None

    def test_set_bbox_simulates_move(self):
        locator = StaticWindowLocator(BoundingBox(0, 0, 1280, 720))
        locator.set_bbox(BoundingBox(400, 300, 1280, 720))   # fenêtre déplacée, même taille
        moved = locator.find_client_bbox()
        assert moved.left == 400 and moved.top == 300
        assert moved.width == 1280 and moved.height == 720

    def test_set_bbox_simulates_resize(self):
        locator = StaticWindowLocator(BoundingBox(0, 0, 1280, 720))
        locator.set_bbox(BoundingBox(0, 0, 1920, 1080))      # fenêtre agrandie
        resized = locator.find_client_bbox()
        assert resized.width == 1920 and resized.height == 1080


class TestWin32WindowLocatorConstruction:
    """Seule la validation de constructeur est testable sans Windows/pywin32."""

    def test_rejects_empty_title(self):
        with pytest.raises(ValueError):
            Win32WindowLocator("")

    def test_stores_title_substring(self):
        locator = Win32WindowLocator("CoinPoker")
        assert locator.title_substring == "CoinPoker"


class TestSelectWindow:
    """
    select_window() porte toute la logique de désambiguïsation de
    Win32WindowLocator, séparée de EnumWindows (Windows uniquement) pour
    être testable ici avec de simples listes de (hwnd, titre) simulées —
    y compris le cas clé soulevé par le multi-tabling : plusieurs tables
    ouvertes en même temps, avec des titres qui se ressemblent.
    """

    def test_single_match(self):
        candidates = [(111, "CoinPoker - Table 1")]
        assert select_window(candidates, "CoinPoker") == 111

    def test_no_match_returns_none(self):
        candidates = [(111, "Une autre fenêtre")]
        assert select_window(candidates, "CoinPoker") is None

    def test_case_insensitive(self):
        candidates = [(111, "COINPOKER - Table 1")]
        assert select_window(candidates, "coinpoker") == 111

    def test_ignores_non_matching_windows(self):
        candidates = [
            (111, "CoinPoker - Table 1"),
            (222, "Visual Studio Code"),
            (333, "Chrome"),
        ]
        assert select_window(candidates, "CoinPoker") == 111

    def test_new_window_instance_each_call_still_found(self):
        # Simule une table fermée puis rouverte : hwnd DIFFÉRENT, même
        # sous-chaîne de titre -> toujours retrouvée normalement, tant
        # qu'un seul candidat correspond à la fois.
        first_session = [(111, "CoinPoker - Table 1")]
        assert select_window(first_session, "CoinPoker") == 111
        second_session = [(999, "CoinPoker - Table 1")]  # nouveau hwnd
        assert select_window(second_session, "CoinPoker") == 999

    def test_multiple_tables_raise_ambiguous_error(self):
        # Le cas soulevé par le multi-tabling : 2 tables ouvertes en même
        # temps avec des titres qui matchent tous les deux -> jamais de
        # choix arbitraire, une erreur explicite.
        candidates = [
            (111, "CoinPoker - Table 1"),
            (222, "CoinPoker - Table 2"),
        ]
        with pytest.raises(AmbiguousWindowError):
            select_window(candidates, "CoinPoker")

    def test_ambiguous_error_message_mentions_count(self):
        candidates = [(1, "Table A"), (2, "Table B"), (3, "Table C")]
        with pytest.raises(AmbiguousWindowError, match="3"):
            select_window(candidates, "Table")

    def test_specific_enough_substring_disambiguates(self):
        # Si le client inclut un identifiant de table dans le titre, une
        # sous-chaîne assez précise résout l'ambiguïté sans changer le code.
        candidates = [
            (111, "CoinPoker - Table 1 - $1/$2 NLHE"),
            (222, "CoinPoker - Table 2 - $1/$2 NLHE"),
        ]
        assert select_window(candidates, "Table 1 -") == 111
