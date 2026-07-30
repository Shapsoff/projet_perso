"""
test_sizing_optimizer.py — Tests unitaires Phase 5 / P2
Bot Poker Académique

Couvre le SizingOptimizer :
  Section 1 — Interpolation quadratique (_quadratic_peak)
  Section 2 — find_optimal_sizing (cas nominaux et cas limites)
  Section 3 — get_ev_curve (cohérence de la courbe)

Usage :
    python tests/test_sizing_optimizer.py
    python -m pytest tests/test_sizing_optimizer.py -v
"""

import sys
import os
import math
import unittest
from typing import Dict, List
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.best_response.frequency_model import FrequencyModel
from core.best_response.ev_calculator import EVCalculator, EVResult
from core.best_response.sizing_optimizer import SizingOptimizer, OptimalSizingResult


# =============================================================================
# Stubs
# =============================================================================

class _MockMCEngine:
    """Monte Carlo stub avec equity variable selon le sizing."""
    def __init__(self, equity: float = 0.60):
        self._equity = equity

    def compute(self, our_hand_str, board_str, opp_ranges, n_sims=5000):
        result = MagicMock()
        result.ev   = self._equity
        result.win  = self._equity - 0.05
        result.tie  = 0.10
        result.loss = 1.0 - self._equity - 0.05
        result.n    = n_sims
        return result


class _MockEHSCalc:
    def calculate_multiway(self, hand, board, n_opp, n_sims):
        result = MagicMock()
        result.EHS  = 0.65
        result.HS   = 0.65
        result.PPot = 0.10
        result.NPot = 0.05
        result.elapsed_ms = 0.0
        return result


def _make_optimizer(equity: float = 0.60) -> SizingOptimizer:
    """Construit un SizingOptimizer avec stubs."""
    freq_model = FrequencyModel(
        ehs_calculator=_MockEHSCalc(), n_sims=100
    )
    ev_calc = EVCalculator(
        mc_engine=_MockMCEngine(equity=equity),
        frequency_model=freq_model,
        n_mc_sims=200,
    )
    return SizingOptimizer(
        ev_calculator=ev_calc,
        curve_sizings=[0.25, 0.33, 0.50, 0.75, 1.00, 1.50, 2.00],
    )


_DISTRIBUTION = {
    'AhKd': 0.10, 'AhKc': 0.10, 'AhKs': 0.10,
    'AdKh': 0.10, 'AdKc': 0.10, 'AdKs': 0.10,
    'KhQd': 0.10, 'KhQc': 0.10, 'KhQs': 0.10,
    '9h8h': 0.10,
}
_OUR_HAND = ['Ah', 'Kd']
_BOARD    = ['7c', '2h', 'Js']


# =============================================================================
# Section 1 — Interpolation quadratique
# =============================================================================

class TestQuadraticPeak(unittest.TestCase):
    """Tests de la méthode _quadratic_peak."""

    def test_parabola_concave_finds_peak(self):
        """Parabole concave — sommet entre les deux bornes."""
        # y = -(x-1)^2 + 5 → sommet à x=1, y=5
        x0, y0 = 0.0, 4.0   # -(0-1)^2 + 5 = 4
        x1, y1 = 1.0, 5.0   # -(1-1)^2 + 5 = 5
        x2, y2 = 2.0, 4.0   # -(2-1)^2 + 5 = 4

        x_peak, y_peak = SizingOptimizer._quadratic_peak(x0, y0, x1, y1, x2, y2)
        self.assertAlmostEqual(x_peak, 1.0, places=6)
        self.assertAlmostEqual(y_peak, 5.0, places=6)

    def test_asymmetric_parabola(self):
        """Parabole asymétrique — sommet décalé."""
        # y = -(x-0.7)^2 + 3 → sommet à x=0.7
        def f(x): return -(x - 0.7)**2 + 3
        x0, y0 = 0.33, f(0.33)
        x1, y1 = 0.75, f(0.75)
        x2, y2 = 1.00, f(1.00)

        x_peak, y_peak = SizingOptimizer._quadratic_peak(x0, y0, x1, y1, x2, y2)
        self.assertAlmostEqual(x_peak, 0.7, places=5)
        self.assertAlmostEqual(y_peak, 3.0, places=5)

    def test_convex_parabola_raises(self):
        """Parabole convexe (minimum) → ValueError."""
        # y = (x-1)^2 → minimum à x=1
        x0, y0 = 0.0, 1.0
        x1, y1 = 1.0, 0.0
        x2, y2 = 2.0, 1.0

        with self.assertRaises(ValueError):
            SizingOptimizer._quadratic_peak(x0, y0, x1, y1, x2, y2)

    def test_colinear_points_raises(self):
        """Points colinéaires → ZeroDivisionError."""
        # y = x → ligne droite
        x0, y0 = 0.0, 0.0
        x1, y1 = 1.0, 1.0
        x2, y2 = 2.0, 2.0

        with self.assertRaises(ZeroDivisionError):
            SizingOptimizer._quadratic_peak(x0, y0, x1, y1, x2, y2)

    def test_poker_realistic_values(self):
        """Valeurs réalistes poker — sizings en fraction du pot."""
        # Courbe EV typique avec maximum autour de 75%
        def ev_curve(s):
            # Courbe concave avec sommet à s=0.75
            return -(s - 0.75)**2 * 20 + 55  # EV max ≈ 55 à s=0.75

        x0, y0 = 0.50, ev_curve(0.50)
        x1, y1 = 0.75, ev_curve(0.75)
        x2, y2 = 1.00, ev_curve(1.00)

        x_peak, y_peak = SizingOptimizer._quadratic_peak(x0, y0, x1, y1, x2, y2)
        self.assertAlmostEqual(x_peak, 0.75, places=5)
        self.assertAlmostEqual(y_peak, ev_curve(0.75), places=3)


# =============================================================================
# Section 2 — find_optimal_sizing
# =============================================================================

class TestFindOptimalSizing(unittest.TestCase):
    """Tests de SizingOptimizer.find_optimal_sizing()."""

    def setUp(self):
        self.optimizer = _make_optimizer(equity=0.65)

    def test_returns_optimal_sizing_result(self):
        """Retourne un OptimalSizingResult valide."""
        result = self.optimizer.find_optimal_sizing(
            our_hand=_OUR_HAND,
            board=_BOARD,
            distribution=_DISTRIBUTION,
            archetype='LAG',
            pot=100,
            stack=1000,
        )
        self.assertIsInstance(result, OptimalSizingResult)
        self.assertIsNotNone(result.optimal_sizing)
        self.assertIsNotNone(result.optimal_ev)
        self.assertIn(result.method,
                      ['quadratic', 'discrete', 'single', 'no_bet'])

    def test_optimal_sizing_within_bounds(self):
        """Le sizing optimal est dans [0.20, 2.50]."""
        result = self.optimizer.find_optimal_sizing(
            our_hand=_OUR_HAND,
            board=_BOARD,
            distribution=_DISTRIBUTION,
            archetype='TAG',
            pot=100,
            stack=1000,
        )
        if result.method != 'no_bet':
            self.assertGreaterEqual(result.optimal_sizing, 0.20)
            self.assertLessEqual(result.optimal_sizing, 2.50)

    def test_optimal_ev_not_nan(self):
        """L'EV optimale n'est jamais NaN."""
        for archetype in ['TAG', 'LAG', 'CALLING_STATION']:
            result = self.optimizer.find_optimal_sizing(
                our_hand=_OUR_HAND,
                board=_BOARD,
                distribution=_DISTRIBUTION,
                archetype=archetype,
                pot=100,
                stack=1000,
            )
            self.assertFalse(math.isnan(result.optimal_ev),
                f"EV NaN pour archetype={archetype}")

    def test_optimal_ev_geq_discrete_best(self):
        """L'EV optimale est >= l'EV du meilleur point discret."""
        result = self.optimizer.find_optimal_sizing(
            our_hand=_OUR_HAND,
            board=_BOARD,
            distribution=_DISTRIBUTION,
            archetype='LAG',
            pot=100,
            stack=1000,
        )
        if result.method != 'no_bet':
            self.assertGreaterEqual(
                result.optimal_ev,
                result.discrete_best.ev - 1e-6,
                "L'EV optimale ne peut pas être inférieure au discret"
            )

    def test_stack_constraint_respected(self):
        """Le montant optimal ne dépasse pas le stack."""
        stack = 80.0  # stack serré
        result = self.optimizer.find_optimal_sizing(
            our_hand=_OUR_HAND,
            board=_BOARD,
            distribution=_DISTRIBUTION,
            archetype='LAG',
            pot=100,
            stack=stack,
        )
        self.assertLessEqual(result.optimal_amount, stack + 1e-6,
            f"Montant {result.optimal_amount} dépasse le stack {stack}")

    def test_single_sizing_available(self):
        """Avec un seul sizing valide, method='single' ou 'discrete'."""
        optimizer = SizingOptimizer(
            ev_calculator=_make_optimizer()._ev_calc,
            curve_sizings=[0.75],  # un seul sizing
        )
        result = optimizer.find_optimal_sizing(
            our_hand=_OUR_HAND,
            board=_BOARD,
            distribution=_DISTRIBUTION,
            archetype='TAG',
            pot=100,
            stack=1000,
        )
        self.assertIn(result.method, ['single', 'discrete', 'no_bet'])

    def test_empty_distribution_safe(self):
        """Distribution vide → pas d'erreur, retourne un résultat."""
        result = self.optimizer.find_optimal_sizing(
            our_hand=_OUR_HAND,
            board=_BOARD,
            distribution={},
            archetype='TAG',
            pot=100,
            stack=1000,
        )
        self.assertIsNotNone(result)
        self.assertFalse(math.isnan(result.optimal_ev))

    def test_all_discrete_populated(self):
        """all_discrete contient les EVResults de tous les sizings évalués."""
        result = self.optimizer.find_optimal_sizing(
            our_hand=_OUR_HAND,
            board=_BOARD,
            distribution=_DISTRIBUTION,
            archetype='LAG',
            pot=100,
            stack=1000,
        )
        self.assertGreater(len(result.all_discrete), 0)


# =============================================================================
# Section 3 — get_ev_curve
# =============================================================================

class TestGetEvCurve(unittest.TestCase):
    """Tests de SizingOptimizer.get_ev_curve()."""

    def setUp(self):
        self.optimizer = _make_optimizer(equity=0.60)

    def test_curve_sorted_by_sizing(self):
        """La courbe est triée par sizing croissant."""
        curve = self.optimizer.get_ev_curve(
            our_hand=_OUR_HAND,
            board=_BOARD,
            distribution=_DISTRIBUTION,
            archetype='TAG',
            pot=100,
            stack=1000,
        )
        sizings = [s for s, _ in curve]
        self.assertEqual(sizings, sorted(sizings),
            "La courbe n'est pas triée par sizing croissant")

    def test_curve_non_empty(self):
        """La courbe contient au moins un point."""
        curve = self.optimizer.get_ev_curve(
            our_hand=_OUR_HAND,
            board=_BOARD,
            distribution=_DISTRIBUTION,
            archetype='LAG',
            pot=100,
            stack=1000,
        )
        self.assertGreater(len(curve), 0)

    def test_curve_ev_values_finite(self):
        """Toutes les valeurs EV de la courbe sont finies."""
        curve = self.optimizer.get_ev_curve(
            our_hand=_OUR_HAND,
            board=_BOARD,
            distribution=_DISTRIBUTION,
            archetype='CALLING_STATION',
            pot=100,
            stack=1000,
        )
        for sizing, ev in curve:
            self.assertFalse(math.isnan(ev),
                f"EV NaN pour sizing={sizing}")
            self.assertFalse(math.isinf(ev),
                f"EV infinie pour sizing={sizing}")

    def test_curve_sizings_match_input(self):
        """Les sizings de la courbe correspondent aux sizings demandés."""
        custom_sizings = [0.33, 0.75, 1.50]
        optimizer = SizingOptimizer(
            ev_calculator=self.optimizer._ev_calc,
            curve_sizings=custom_sizings,
        )
        curve = optimizer.get_ev_curve(
            our_hand=_OUR_HAND,
            board=_BOARD,
            distribution=_DISTRIBUTION,
            archetype='TAG',
            pot=100,
            stack=1000,
        )
        curve_sizings = [s for s, _ in curve]
        for s in curve_sizings:
            self.assertIn(round(s, 2), [round(x, 2) for x in custom_sizings],
                f"Sizing {s} inattendu dans la courbe")


# =============================================================================
# Runner standalone
# =============================================================================

def run_tests():
    loader = unittest.TestLoader()
    sections = [
        ("Section 1 — Interpolation quadratique", TestQuadraticPeak),
        ("Section 2 — find_optimal_sizing",        TestFindOptimalSizing),
        ("Section 3 — get_ev_curve",               TestGetEvCurve),
    ]

    total_passed = total_failed = total_errors = 0

    for title, cls in sections:
        print(f"\n{'─' * 60}")
        print(f"  {title}")
        print(f"{'─' * 60}")
        s = loader.loadTestsFromTestCase(cls)
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(s)
        total_passed += result.testsRun - len(result.failures) - len(result.errors)
        total_failed += len(result.failures)
        total_errors += len(result.errors)

    print(f"\n{'═' * 60}")
    total = total_passed + total_failed + total_errors
    print(f"  TOTAL : {total_passed}/{total} tests passés")
    if total_failed == 0 and total_errors == 0:
        print("  ✅ Tous les tests SizingOptimizer (P2) passent.")
    else:
        print(f"  ⚠  {total_failed} échec(s), {total_errors} erreur(s)")
    print(f"{'═' * 60}\n")
    return total_failed + total_errors == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
