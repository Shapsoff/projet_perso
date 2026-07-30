"""
sizing_optimizer.py — Sizing Optimal Continu
Phase 5 — P2 — Bot Poker Académique

Trouve le sizing de bet qui maximise l'EV en continu, en interpolant
entre les points discrets calculés par l'EVCalculator.

Principe :
    Le calcul d'EV par action (P0) évalue 5 sizings discrets :
    [33%, 50%, 75%, 100%, 200%]. La courbe EV(sizing) est continue
    et généralement unimodale (un seul maximum). On peut trouver un
    meilleur sizing qu'un des 5 points discrets en interpolant.

    Méthode : interpolation quadratique locale autour du maximum discret.
    Si le maximum discret est au point (s*, EV*), on fit une parabole
    sur (s*-1, s*, s*+1) et on cherche le sommet analytiquement.
    Le sommet de ax² + bx + c est à x = -b / 2a.

    Si le sommet est en dehors de l'intervalle [s*-1, s*+1], on reste
    sur le maximum discret — la courbe est trop plate ou monotone.

    Cas spéciaux :
    - Un seul point disponible → on retourne ce point.
    - Deux points → interpolation linéaire, pas de sommet quadratique.
    - Le maximum discret est aux bornes (premier ou dernier sizing) →
      on retourne le point de bord sans extrapoler.

Limites assumées :
    L'interpolation quadratique est une approximation. La vraie courbe
    EV(sizing) n'est pas nécessairement quadratique — elle dépend de la
    forme de la distribution de folds/calls/raises qui change de façon
    non-linéaire avec le sizing. L'approximation est suffisante pour
    trouver un sizing ±5% du vrai optimum, ce qui est plus que suffisant
    en pratique (le bot ne peut pas bet 87.3% exactement de toute façon —
    les tailles de bet sont arrondies à l'entier near).

    Une version future pourrait utiliser golden-section search si on
    voulait une précision théorique maximale, mais ce n'est pas justifié
    dans ce contexte.

Interface publique :
    optimizer = SizingOptimizer(ev_calculator)

    # Trouver le sizing optimal pour un contexte donné
    result = optimizer.find_optimal_sizing(
        our_hand, board, distribution, archetype,
        pot, to_call, stack, street, n_opponents
    )
    # result.optimal_sizing  : fraction du pot (ex: 0.87)
    # result.optimal_ev      : EV estimée à ce sizing
    # result.optimal_amount  : montant en chips
    # result.all_discrete    : List[EVResult] des 5 points discrets
    # result.method          : 'quadratic' | 'discrete' | 'single'
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.best_response.ev_calculator import EVCalculator, EVResult, DEFAULT_SIZINGS

logger = logging.getLogger(__name__)

# Sizings discrets évalués pour la courbe EV(sizing)
# Plus dense autour des valeurs courantes pour une meilleure interpolation
_CURVE_SIZINGS = [0.25, 0.33, 0.50, 0.67, 0.75, 1.00, 1.50, 2.00]

# Sizing minimum et maximum autorisés (fraction du pot)
_SIZING_MIN = 0.20
_SIZING_MAX = 2.50

# Tolérance pour considérer une amélioration comme significative
# (évite de changer le sizing pour un gain marginal de 0.01 chips)
_MIN_EV_IMPROVEMENT = 0.5  # chips


# =============================================================================
# Résultat de l'optimisation
# =============================================================================

@dataclass
class OptimalSizingResult:
    """
    Résultat de l'optimisation du sizing.

    Attributes:
        optimal_sizing  : sizing optimal en fraction du pot
        optimal_ev      : EV estimée à ce sizing (en chips)
        optimal_amount  : montant en chips (= optimal_sizing × pot)
        discrete_best   : meilleur EVResult parmi les points discrets
        all_discrete    : tous les EVResults discrets calculés
        method          : méthode utilisée ('quadratic'|'discrete'|'single')
        improvement     : gain d'EV vs le meilleur point discret (chips)
    """
    optimal_sizing:  float
    optimal_ev:      float
    optimal_amount:  float
    discrete_best:   EVResult
    all_discrete:    List[EVResult]
    method:          str
    improvement:     float = 0.0

    def summary(self) -> str:
        return (
            f"OptimalSizing: {self.optimal_sizing:.0%} pot "
            f"(EV={self.optimal_ev:+.2f}, method={self.method}, "
            f"improvement={self.improvement:+.2f} vs discret={self.discrete_best.sizing_pct:.0%})"
        )


# =============================================================================
# Sizing Optimizer
# =============================================================================

class SizingOptimizer:
    """
    Trouve le sizing de bet optimal en continu par interpolation quadratique.

    Utilise les EVResults déjà calculés par EVCalculator pour construire
    la courbe EV(sizing) sans recalcul Monte Carlo supplémentaire.

    Usage typique :
        optimizer = SizingOptimizer(ev_calculator)
        result = optimizer.find_optimal_sizing(
            our_hand, board, distribution, archetype,
            pot, to_call=0, stack=1000, street='flop'
        )
        best_action = Action('bet', result.optimal_amount, result.optimal_sizing)
    """

    def __init__(
        self,
        ev_calculator:  EVCalculator,
        curve_sizings:  List[float] = None,
    ):
        """
        Args:
            ev_calculator : EVCalculator (P0) déjà initialisé
            curve_sizings : sizings discrets pour la courbe (défaut: _CURVE_SIZINGS)
        """
        self._ev_calc      = ev_calculator
        self._curve_sizings = curve_sizings or _CURVE_SIZINGS

        logger.debug(
            "SizingOptimizer initialisé (curve_sizings=%s)",
            self._curve_sizings,
        )

    # =========================================================================
    # Interface principale
    # =========================================================================

    def find_optimal_sizing(
        self,
        our_hand:     List[str],
        board:        List[str],
        distribution: Dict[str, float],
        archetype:    str,
        pot:          float,
        to_call:      float   = 0.0,
        stack:        float   = 1000.0,
        street:       str     = 'flop',
        n_opponents:  int     = 1,
    ) -> OptimalSizingResult:
        """
        Trouve le sizing optimal en évaluant une courbe de points discrets
        puis en interpolant autour du maximum.

        Args:
            our_hand     : nos deux cartes
            board        : cartes du board
            distribution : Dict[combo, proba] depuis RangeEstimator
            archetype    : archétype estimé de l'adversaire
            pot          : taille du pot
            to_call      : montant à suivre (0 si on a l'initiative)
            stack        : notre stack effectif
            street       : street courante
            n_opponents  : adversaires actifs

        Returns:
            OptimalSizingResult avec le sizing et l'EV optimaux.
        """
        # Filtrer les sizings qui dépassent le stack
        max_sizing = stack / pot if pot > 0 else _SIZING_MAX
        valid_sizings = [s for s in self._curve_sizings
                         if _SIZING_MIN <= s <= min(max_sizing, _SIZING_MAX)]

        if not valid_sizings:
            valid_sizings = [min(self._curve_sizings[0], max_sizing)]

        # Calculer l'EV pour tous les sizings discrets
        all_results = self._ev_calc.compute_all_actions(
            our_hand=our_hand,
            board=board,
            distribution=distribution,
            archetype=archetype,
            pot=pot,
            to_call=to_call,
            stack=stack,
            street=street,
            sizings=valid_sizings,
            n_opponents=n_opponents,
        )

        # Filtrer uniquement les actions bet/raise/allin
        bet_results = [r for r in all_results
                       if r.action in ('bet', 'raise', 'allin') and r.sizing_pct > 0]

        if not bet_results:
            # Pas de bet possible → retourner la meilleure action disponible
            best = all_results[0] if all_results else EVResult('check', 0, 0, 0)
            return OptimalSizingResult(
                optimal_sizing=0.0,
                optimal_ev=best.ev,
                optimal_amount=0.0,
                discrete_best=best,
                all_discrete=all_results,
                method='no_bet',
            )

        # Trier par sizing croissant pour l'interpolation
        bet_results.sort(key=lambda r: r.sizing_pct)

        # Cas trivial : un seul point
        if len(bet_results) == 1:
            best = bet_results[0]
            return OptimalSizingResult(
                optimal_sizing=best.sizing_pct,
                optimal_ev=best.ev,
                optimal_amount=best.amount,
                discrete_best=best,
                all_discrete=all_results,
                method='single',
            )

        # Trouver le maximum discret
        discrete_best = max(bet_results, key=lambda r: r.ev)

        # Interpolation quadratique autour du maximum
        optimal_sizing, optimal_ev, method = self._interpolate_optimum(
            bet_results, discrete_best, pot
        )

        # Vérifier que l'amélioration est significative
        improvement = optimal_ev - discrete_best.ev
        if improvement < _MIN_EV_IMPROVEMENT:
            # Pas d'amélioration significative → rester sur le discret
            optimal_sizing = discrete_best.sizing_pct
            optimal_ev     = discrete_best.ev
            method         = 'discrete'
            improvement    = 0.0

        optimal_amount = min(optimal_sizing * pot, stack)

        result = OptimalSizingResult(
            optimal_sizing=optimal_sizing,
            optimal_ev=optimal_ev,
            optimal_amount=optimal_amount,
            discrete_best=discrete_best,
            all_discrete=all_results,
            method=method,
            improvement=improvement,
        )

        logger.debug("[SizingOptimizer] %s", result.summary())
        return result

    def get_ev_curve(
        self,
        our_hand:     List[str],
        board:        List[str],
        distribution: Dict[str, float],
        archetype:    str,
        pot:          float,
        to_call:      float = 0.0,
        stack:        float = 1000.0,
        street:       str   = 'flop',
        n_opponents:  int   = 1,
    ) -> List[Tuple[float, float]]:
        """
        Retourne la courbe EV(sizing) comme liste de (sizing, ev).

        Utile pour le debug, la visualisation, et les benchmarks P4.

        Returns:
            List[(sizing_pct, ev)] triée par sizing croissant.
        """
        all_results = self._ev_calc.compute_all_actions(
            our_hand=our_hand,
            board=board,
            distribution=distribution,
            archetype=archetype,
            pot=pot,
            to_call=to_call,
            stack=stack,
            street=street,
            sizings=self._curve_sizings,
            n_opponents=n_opponents,
        )
        curve = [(r.sizing_pct, r.ev)
                 for r in all_results
                 if r.action in ('bet', 'raise', 'allin') and r.sizing_pct > 0]
        curve.sort(key=lambda x: x[0])
        return curve

    # =========================================================================
    # Interpolation quadratique
    # =========================================================================

    def _interpolate_optimum(
        self,
        bet_results:   List[EVResult],
        discrete_best: EVResult,
        pot:           float,
    ) -> Tuple[float, float, str]:
        """
        Interpolation quadratique locale autour du maximum discret.

        Fit une parabole sur les 3 points autour du max :
        (s_left, EV_left), (s_best, EV_best), (s_right, EV_right)
        et trouve le sommet analytiquement.

        Le sommet d'une parabole ax² + bx + c est à x* = -b / (2a).
        Si a >= 0, la parabole est convexe (minimum) → pas de maximum.
        Si x* est hors de [s_left, s_right] → rester sur le discret.

        Returns:
            (optimal_sizing, optimal_ev, method)
        """
        idx = next(
            (i for i, r in enumerate(bet_results)
             if r.sizing_pct == discrete_best.sizing_pct),
            None
        )

        if idx is None:
            return discrete_best.sizing_pct, discrete_best.ev, 'discrete'

        # Besoin de 3 points pour la quadratique
        if idx == 0 or idx == len(bet_results) - 1:
            # Maximum aux bornes → pas d'interpolation (évite l'extrapolation)
            return discrete_best.sizing_pct, discrete_best.ev, 'discrete'

        left  = bet_results[idx - 1]
        mid   = bet_results[idx]
        right = bet_results[idx + 1]

        x0, y0 = left.sizing_pct,  left.ev
        x1, y1 = mid.sizing_pct,   mid.ev
        x2, y2 = right.sizing_pct, right.ev

        # Fit quadratique : résoudre le système 3×3
        # y = a×x² + b×x + c
        # On utilise les différences pour éviter les problèmes numériques
        try:
            optimal_s, optimal_ev = self._quadratic_peak(x0, y0, x1, y1, x2, y2)
        except (ZeroDivisionError, ValueError):
            return discrete_best.sizing_pct, discrete_best.ev, 'discrete'

        # Vérifier que le sommet est dans l'intervalle
        if not (x0 <= optimal_s <= x2):
            return discrete_best.sizing_pct, discrete_best.ev, 'discrete'

        # Vérifier que l'EV interpolée est supérieure au discret
        if optimal_ev < discrete_best.ev:
            return discrete_best.sizing_pct, discrete_best.ev, 'discrete'

        return optimal_s, optimal_ev, 'quadratic'

    @staticmethod
    def _quadratic_peak(
        x0: float, y0: float,
        x1: float, y1: float,
        x2: float, y2: float,
    ) -> Tuple[float, float]:
        """
        Trouve le sommet de la parabole passant par (x0,y0), (x1,y1), (x2,y2).

        Utilise la formule de Lagrange pour éviter la résolution matricielle.

        Le sommet est à : x* = 0.5 × (x0²(y1-y2) + x1²(y2-y0) + x2²(y0-y1))
                                   / (x0(y1-y2)  + x1(y2-y0)  + x2(y0-y1))

        Raises:
            ZeroDivisionError : si les trois points sont colinéaires
            ValueError        : si la parabole est convexe (pas de maximum)
        """
        denom = x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1)

        if abs(denom) < 1e-10:
            raise ZeroDivisionError("Points colinéaires — pas de parabole unique")

        numer = (x0**2 * (y1 - y2)
               + x1**2 * (y2 - y0)
               + x2**2 * (y0 - y1))

        x_peak = 0.5 * numer / denom

        # Calculer y_peak par interpolation quadratique de Lagrange
        L0 = ((x_peak - x1) * (x_peak - x2)) / ((x0 - x1) * (x0 - x2))
        L1 = ((x_peak - x0) * (x_peak - x2)) / ((x1 - x0) * (x1 - x2))
        L2 = ((x_peak - x0) * (x_peak - x1)) / ((x2 - x0) * (x2 - x1))

        y_peak = y0 * L0 + y1 * L1 + y2 * L2

        # Vérifier que c'est bien un maximum (parabole concave)
        # Si y_peak < max(y0, y1, y2), la parabole est convexe
        if y_peak < max(y0, y1, y2) - 1e-6:
            raise ValueError("Parabole convexe — pas de maximum")

        return x_peak, y_peak
