"""
test_ocr_template_collector.py — Tests unitaires de ocr/vision/template_collector.py
Phase 7 — Bot Poker Académique

Seules images_differ() et card_region_names() sont testées ici : ce sont
les deux seules fonctions pures du module, indépendantes de toute
fenêtre/écran réel. run() (la boucle de collecte) nécessite un vrai
client de poker ouvert et n'est pas exécutable en environnement de test.

Placement : tests/
Commande   : depuis la racine du projet
    pytest tests/test_ocr_template_collector.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from ocr.vision.region_config import FractionalRegion, RegionConfig
from ocr.vision.template_collector import card_region_names, images_differ


def solid(width, height, color):
    return Image.new("RGB", (width, height), color)


# =============================================================================
# images_differ
# =============================================================================

class TestImagesDiffer:

    def test_identical_images_do_not_differ(self):
        a = solid(40, 60, (200, 50, 50))
        b = solid(40, 60, (200, 50, 50))
        assert not images_differ(a, b)

    def test_clearly_different_colors_differ(self):
        a = solid(40, 60, (255, 255, 255))
        b = solid(40, 60, (0, 0, 0))
        assert images_differ(a, b)

    def test_tiny_noise_within_threshold_does_not_differ(self):
        a = solid(40, 60, (100, 100, 100))
        b = solid(40, 60, (102, 100, 100))   # bruit de compression plausible
        assert not images_differ(a, b, threshold=8.0)

    def test_moderate_change_above_threshold_differs(self):
        a = solid(40, 60, (100, 100, 100))
        b = solid(40, 60, (130, 100, 100))
        assert images_differ(a, b, threshold=8.0)

    def test_different_sizes_always_differ(self):
        a = solid(40, 60, (100, 100, 100))
        b = solid(50, 70, (100, 100, 100))
        assert images_differ(a, b)

    def test_threshold_is_configurable(self):
        a = solid(40, 60, (100, 100, 100))
        b = solid(40, 60, (130, 100, 100))   # diff moyenne (30+0+0)/3 = 10
        assert not images_differ(a, b, threshold=20.0)
        assert images_differ(a, b, threshold=5.0)

    def test_partial_region_change_detected(self):
        # Simule une vraie carte : seule une partie de l'image change
        # (le rang/la couleur dans un coin), le reste (fond) reste
        # identique. La zone modifiée doit être assez grande par rapport
        # à l'image entière pour faire dépasser le seuil (la moyenne se
        # calcule sur TOUS les pixels, pas seulement la zone changée).
        a = Image.new("RGB", (40, 60), (255, 255, 255))
        b = a.copy()
        for x in range(2, 17):
            for y in range(2, 17):
                b.putpixel((x, y), (0, 0, 0))
        assert images_differ(a, b, threshold=8.0)


# =============================================================================
# card_region_names
# =============================================================================

class TestCardRegionNames:

    def test_filters_card_regions_only(self):
        regions = RegionConfig({
            "hero_card_1": FractionalRegion(0.1, 0.1, 0.05, 0.08),
            "hero_card_2": FractionalRegion(0.16, 0.1, 0.05, 0.08),
            "board_card_1": FractionalRegion(0.3, 0.3, 0.05, 0.08),
            "seat_0_stack": FractionalRegion(0.05, 0.9, 0.1, 0.03),
            "pot": FractionalRegion(0.45, 0.45, 0.1, 0.05),
        })
        names = card_region_names(regions)
        assert set(names) == {"hero_card_1", "hero_card_2", "board_card_1"}

    def test_empty_when_no_card_regions(self):
        regions = RegionConfig({"pot": FractionalRegion(0.45, 0.45, 0.1, 0.05)})
        assert card_region_names(regions) == []

    def test_empty_config(self):
        assert card_region_names(RegionConfig({})) == []

    def test_all_five_board_cards_and_both_hero_cards(self):
        regions = RegionConfig({
            **{f"hero_card_{i}": FractionalRegion(0.1 * i, 0.1, 0.05, 0.08) for i in (1, 2)},
            **{f"board_card_{i}": FractionalRegion(0.1 * i, 0.3, 0.05, 0.08) for i in range(1, 6)},
        })
        assert len(card_region_names(regions)) == 7
