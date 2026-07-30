"""
test_selfplay_variance.py — Diagnostic : biais systématique ou simple variance ?

Lance plusieurs matchs self-play avec des seeds différentes pour voir si
l'exploitabilité mesurée est stable (même signe, même ordre de grandeur à
chaque fois -> biais réel) ou si elle varie fortement voire change de signe
(-> beaucoup de variance sur 500 mains, pas nécessairement un vrai edge
d'un bot sur l'autre).

Usage :
    python tests/test_selfplay_variance.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.WARNING)  # pas besoin du détail ici

from tests.validate_phase5 import validate_self_play, ValidationConfig

SEEDS = [1, 2, 3, 4, 5, 42, 123, 999, 7777, 31415]

results = []
for seed in SEEDS:
    config = ValidationConfig(seed=seed)
    result = validate_self_play(config, n_hands=5000)
    results.append((seed, result.bb_100))

print("\n" + "=" * 50)
print("Récapitulatif")
print("=" * 50)
for seed, bb100 in results:
    print(f"  seed={seed:<6} BB/100={bb100:+8.2f}")

values = [v for _, v in results]
mean = sum(values) / len(values)
print(f"\nMoyenne   : {mean:+.2f} BB/100")
print(f"Min / Max : {min(values):+.2f} / {max(values):+.2f}")
same_sign = all(v < 0 for v in values) or all(v > 0 for v in values)
print(f"Toujours le même signe : {same_sign}")
