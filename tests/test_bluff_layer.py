"""
test_bluff_layer.py — Tests unitaires Phase 5 / P3
Bot Poker Académique

Couvre la couche bluff minimale :
  Section 1 — Calcul de la fréquence minimale (_compute_min_bluff_freq)
  Section 2 — Calcul de l'EV du bluff (_compute_ev_bluff)
  Section 3 — evaluate() : décisions complètes par situation

Usage :
    python tests/test_bluff_layer.py
    python -m pytest tests/test_bluff_layer.py -v
"""

import sys
import os
import math
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.best_response.bluff_layer import (
    BluffLayer,
    BluffLayerConfig,
    BluffEvalResult,
)
from core.best_response.ev_calculator import EVResult


# =============================================================================
# Helper — créer un EVResult de bet minimal
# =============================================================================

def _make_bet_ev(
    sizing_pct: float = 0.75,
    pot:        float = 100.0,
    p_fold:     float = 0.45,
    p_call:     float = 0.40,
    p_raise:    float = 0.15,
    equity:     float = 0.60,
) -> EVResult:
    amount = sizing_pct * pot
    return EVResult(
        action='bet',
        sizing_pct=sizing_pct,
        amount=amount,
        ev=p_fold * pot + p_call * (equity * (pot + 2 * amount) - amount) + p_raise * (-amount),
        p_fold=p_fold,
        p_call=p_call,
        p_raise=p_raise,
        equity_call=equity,
    )


# =============================================================================
# Section 1 — _compute_min_bluff_freq
# =============================================================================

class TestComputeMinBluffFreq(unittest.TestCase):
    """Tests du calcul de la fréquence de bluff minimale théorique."""

    def test_bet_half_pot(self):
        """bet 50% pot → pot_odds = 50/(100+50) = 33%."""
        freq = BluffLayer._compute_min_bluff_freq(50.0, 100.0)
        self.assertAlmostEqual(freq, 50 / 150, places=6)
        self.assertAlmostEqual(freq, 0.3333, places=3)

    def test_bet_75_pot(self):
        """bet 75% pot → pot_odds = 75/(100+75) ≈ 42.9%."""
        freq = BluffLayer._compute_min_bluff_freq(75.0, 100.0)
        self.assertAlmostEqual(freq, 75 / 175, places=6)
        self.assertAlmostEqual(freq, 0.4286, places=3)

    def test_bet_pot(self):
        """bet 100% pot → pot_odds = 100/200 = 50%."""
        freq = BluffLayer._compute_min_bluff_freq(100.0, 100.0)
        self.assertAlmostEqual(freq, 0.50, places=6)

    def test_bet_overbet(self):
        """overbet 200% pot → pot_odds = 200/300 ≈ 66.7%."""
        freq = BluffLayer._compute_min_bluff_freq(200.0, 100.0)
        self.assertAlmostEqual(freq, 200 / 300, places=6)

    def test_zero_pot(self):
        """pot = 0 → fréquence = 0 (pas de division par zéro)."""
        freq = BluffLayer._compute_min_bluff_freq(50.0, 0.0)
        self.assertEqual(freq, 0.0)

    def test_zero_bet(self):
        """bet = 0 → fréquence = 0 (pas de bet, pas de bluff)."""
        freq = BluffLayer._compute_min_bluff_freq(0.0, 100.0)
        self.assertEqual(freq, 0.0)

    def test_monotone_increasing(self):
        """Plus le bet est grand, plus la fréquence minimale est haute."""
        freqs = [BluffLayer._compute_min_bluff_freq(b, 100.0)
                 for b in [25, 50, 75, 100, 150, 200]]
        self.assertEqual(freqs, sorted(freqs))


# =============================================================================
# Section 2 — _compute_ev_bluff
# =============================================================================

class TestComputeEvBluff(unittest.TestCase):
    """Tests du calcul d'EV du bluff."""

    def test_pure_bluff_positive_ev(self):
        """Bluff pur avec p_fold élevé → EV positive."""
        # EV = p_fold × pot - p_call × bet
        # = 0.6 × 100 - 0.4 × 75 = 60 - 30 = +30
        ev = BluffLayer._compute_ev_bluff(75.0, 100.0, p_fold=0.60, equity_if_called=0.0)
        self.assertAlmostEqual(ev, 0.60 * 100 - 0.40 * 75, places=4)
        self.assertGreater(ev, 0)

    def test_pure_bluff_negative_ev(self):
        """Bluff pur avec p_fold faible → EV négative."""
        # EV = 0.2 × 100 - 0.8 × 75 = 20 - 60 = -40
        ev = BluffLayer._compute_ev_bluff(75.0, 100.0, p_fold=0.20, equity_if_called=0.0)
        self.assertLess(ev, 0)

    def test_semi_bluff_better_than_pure_bluff(self):
        """Semi-bluff (PPot > 0) toujours meilleur que bluff pur."""
        ev_pure = BluffLayer._compute_ev_bluff(75.0, 100.0, p_fold=0.40, equity_if_called=0.0)
        ev_semi = BluffLayer._compute_ev_bluff(75.0, 100.0, p_fold=0.40, equity_if_called=0.30)
        self.assertGreater(ev_semi, ev_pure)

    def test_break_even_fold_freq(self):
        """À la fréquence de fold exacte de break-even, EV ≈ 0 pour bluff pur."""
        # Pour un bluff pur : EV = 0 ⟺ p_fold = bet/(pot+bet) = 75/175 ≈ 0.4286
        p_fold_be = 75.0 / (100.0 + 75.0)
        ev = BluffLayer._compute_ev_bluff(75.0, 100.0, p_fold=p_fold_be, equity_if_called=0.0)
        self.assertAlmostEqual(ev, 0.0, places=3)

    def test_ev_formula_components(self):
        """Vérifier la formule composante par composante."""
        bet, pot, p_fold, eq = 60.0, 100.0, 0.50, 0.25
        p_call = 1.0 - p_fold
        expected = p_fold * pot + p_call * (eq * (pot + 2 * bet) - bet)
        result   = BluffLayer._compute_ev_bluff(bet, pot, p_fold, eq)
        self.assertAlmostEqual(result, expected, places=6)


# =============================================================================
# Section 3 — evaluate() : décisions complètes
# =============================================================================

class TestBluffLayerEvaluate(unittest.TestCase):
    """Tests de la méthode evaluate() complète."""

    def setUp(self):
        self.layer = BluffLayer()

    def test_value_hand_no_bluff(self):
        """Main de valeur (EHS élevé) → pas de bluff."""
        ev = _make_bet_ev(sizing_pct=0.75, p_fold=0.45)
        result = self.layer.evaluate(
            best_ev=ev, ehs=0.75, ppot=0.05,
            pot=100, street='flop', archetype='TAG',
        )
        self.assertFalse(result.should_bluff)
        self.assertEqual(result.bluff_type, 'none')
        self.assertIn('Value hand', result.reason)

    def test_semi_bluff_with_strong_draw(self):
        """Draw fort (PPot ≥ 0.18) + fold equity → semi-bluff."""
        ev = _make_bet_ev(sizing_pct=0.75, p_fold=0.45)
        result = self.layer.evaluate(
            best_ev=ev, ehs=0.35, ppot=0.30,
            pot=100, street='flop', archetype='LAG',
        )
        self.assertTrue(result.should_bluff)
        self.assertEqual(result.bluff_type, 'semi_bluff')
        self.assertGreater(result.ev_bluff, 0)

    def test_no_bluff_insufficient_fold_equity(self):
        """Fold equity insuffisante → pas de bluff même avec draw."""
        ev = _make_bet_ev(sizing_pct=0.75, p_fold=0.10)  # CS ne fold presque pas
        result = self.layer.evaluate(
            best_ev=ev, ehs=0.35, ppot=0.35,
            pot=100, street='flop', archetype='CALLING_STATION',
        )
        self.assertFalse(result.should_bluff)
        self.assertIn('Fold equity insuffisante', result.reason)

    def test_no_bluff_min_freq_too_low(self):
        """Sizing très petit → fréquence minimale < seuil actionnable."""
        ev = _make_bet_ev(sizing_pct=0.05, p_fold=0.50)  # bet 5% pot
        result = self.layer.evaluate(
            best_ev=ev, ehs=0.35, ppot=0.30,
            pot=100, street='flop', archetype='TAG',
        )
        self.assertFalse(result.should_bluff)

    def test_river_no_semi_bluff_by_default(self):
        """Sur la river, pas de semi-bluff (config par défaut)."""
        ev = _make_bet_ev(sizing_pct=0.75, p_fold=0.45)
        result = self.layer.evaluate(
            best_ev=ev, ehs=0.35, ppot=0.35,
            pot=100, street='river', archetype='TAG',
        )
        # Sur river, river_semi_bluff=False → semi-bluff désactivé
        # Peut être pure_bluff si PPot >= min_ppot_pure_bluff
        if result.should_bluff:
            self.assertEqual(result.bluff_type, 'pure_bluff')

    def test_pure_bluff_positive_ev(self):
        """Bluff pur avec bonne fold equity → EV positive."""
        ev = _make_bet_ev(sizing_pct=0.75, p_fold=0.55)
        config = BluffLayerConfig(
            min_ppot_pure_bluff=0.0,  # pas de seuil PPot pour test
            allow_pure_bluff=True,
        )
        layer = BluffLayer(config=config)
        result = layer.evaluate(
            best_ev=ev, ehs=0.35, ppot=0.10,
            pot=100, street='river', archetype='TAG',
        )
        if result.should_bluff:
            self.assertEqual(result.bluff_type, 'pure_bluff')
            self.assertGreater(result.ev_bluff, 0)

    def test_pure_bluff_disabled(self):
        """allow_pure_bluff=False → jamais de bluff pur."""
        config = BluffLayerConfig(allow_pure_bluff=False)
        layer  = BluffLayer(config=config)
        ev     = _make_bet_ev(sizing_pct=0.75, p_fold=0.60)
        result = layer.evaluate(
            best_ev=ev, ehs=0.35, ppot=0.05,
            pot=100, street='river', archetype='TAG',
        )
        self.assertNotEqual(result.bluff_type, 'pure_bluff')

    def test_min_bluff_freq_correct(self):
        """La fréquence minimale théorique est correctement calculée."""
        sizing_pct = 0.75
        pot        = 100.0
        ev         = _make_bet_ev(sizing_pct=sizing_pct, p_fold=0.45)
        result     = self.layer.evaluate(
            best_ev=ev, ehs=0.35, ppot=0.30,
            pot=pot, street='flop', archetype='LAG',
        )
        expected_min = (sizing_pct * pot) / (pot + sizing_pct * pot)
        self.assertAlmostEqual(result.min_bluff_freq, expected_min, places=4)

    def test_deficit_positive_when_underbluffing(self):
        """Déficit positif quand la fréquence actuelle < minimale."""
        ev = _make_bet_ev(sizing_pct=0.75, p_fold=0.45)
        result = self.layer.evaluate(
            best_ev=ev, ehs=0.40, ppot=0.05,  # ppot faible = peu de bluffs
            pot=100, street='flop', archetype='TAG',
        )
        # min_bluff_freq ≈ 0.43, current ≈ ppot=0.05 → déficit ≈ 0.38
        if result.min_bluff_freq > result.bluff_freq:
            self.assertGreater(result.deficit, 0)

    def test_result_has_all_fields(self):
        """BluffEvalResult contient tous les champs attendus."""
        ev     = _make_bet_ev()
        result = self.layer.evaluate(
            best_ev=ev, ehs=0.40, ppot=0.25,
            pot=100, street='flop', archetype='TAG',
        )
        self.assertIsInstance(result.should_bluff,    bool)
        self.assertIsInstance(result.bluff_type,      str)
        self.assertIsInstance(result.bluff_freq,      float)
        self.assertIsInstance(result.min_bluff_freq,  float)
        self.assertIsInstance(result.ev_bluff,        float)
        self.assertIsInstance(result.deficit,         float)
        self.assertIsInstance(result.reason,          str)
        self.assertFalse(math.isnan(result.ev_bluff))

    def test_summary_not_empty(self):
        """summary() retourne une chaîne non vide."""
        ev     = _make_bet_ev()
        result = self.layer.evaluate(
            best_ev=ev, ehs=0.40, ppot=0.25,
            pot=100, street='flop', archetype='TAG',
        )
        self.assertGreater(len(result.summary()), 0)

    def test_no_bluff_on_check(self):
        """Pas de bluff si sizing = 0 (check)."""
        ev_check = EVResult('check', 0.0, 0.0, 50.0, p_fold=0.0)
        result   = self.layer.evaluate(
            best_ev=ev_check, ehs=0.40, ppot=0.30,
            pot=100, street='flop', archetype='TAG',
        )
        self.assertFalse(result.should_bluff)


# =============================================================================
# Runner standalone
# =============================================================================

def run_tests():
    loader = unittest.TestLoader()
    sections = [
        ("Section 1 — _compute_min_bluff_freq", TestComputeMinBluffFreq),
        ("Section 2 — _compute_ev_bluff",        TestComputeEvBluff),
        ("Section 3 — evaluate()",               TestBluffLayerEvaluate),
    ]

    total_passed = total_failed = total_errors = 0

    for title, cls in sections:
        print(f"\n{'─' * 60}")
        print(f"  {title}")
        print(f"{'─' * 60}")
        s      = loader.loadTestsFromTestCase(cls)
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(s)
        total_passed += result.testsRun - len(result.failures) - len(result.errors)
        total_failed += len(result.failures)
        total_errors += len(result.errors)

    print(f"\n{'═' * 60}")
    total = total_passed + total_failed + total_errors
    print(f"  TOTAL : {total_passed}/{total} tests passés")
    if total_failed == 0 and total_errors == 0:
        print("  ✅ Tous les tests BluffLayer (P3) passent.")
    else:
        print(f"  ⚠  {total_failed} échec(s), {total_errors} erreur(s)")
    print(f"{'═' * 60}\n")
    return total_failed + total_errors == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
