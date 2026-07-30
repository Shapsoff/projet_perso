"""
ev_calculator.py — Calcul d'EV par action
Phase 5 — Bot Poker Académique

Calcule l'Expected Value (EV) de chaque action possible (fold, check, call,
bet à différents sizings, allin) en combinant :
  - Le FrequencyModel (fréquences fold/call/raise de l'adversaire)
  - Le MonteCarloEngine existant (équité vs la range qui call/raise)

Formule d'EV pour un bet de taille B dans un pot P :

  EV(bet_B) = p_fold  × P
            + p_call  × [equity_vs_call_range × (P + 2B) - B]
            + p_raise × EV_face_à_raise

où :
  - P            = pot avant notre bet
  - B            = montant du bet (B = bet_sizing × P)
  - equity_vs_call_range = notre equity MC vs les combos qui callent
  - EV_face_à_raise     = approximation conservative (fold notre main)

EV d'un call face à une mise adverse :
  EV(call) = equity × (pot + to_call) - to_call

EV d'un fold :
  EV(fold) = 0  (par convention, on perd ce qu'on a déjà investi)

EV d'un check :
  EV(check) = equity × pot  (approximation — pas de bet adverse garanti)

Interface publique :
    calc = EVCalculator(monte_carlo_engine, frequency_model)
    results = calc.compute_all_actions(
        our_hand,           # ['Ah', 'Kd']
        board,              # ['7c', '2h', 'Js']
        distribution,       # Dict[combo, proba] depuis RangeEstimator
        archetype,          # 'TAG' | 'LAG' | 'CALLING_STATION'
        pot,                # taille du pot
        to_call,            # montant à suivre (0 si on a l'initiative)
        stack,              # notre stack effectif
        street,             # 'flop' | 'turn' | 'river'
        sizings,            # list de bet sizings à évaluer (fraction du pot)
        n_opponents,        # adversaires actifs
    )
    → List[EVResult] trié par EV décroissante

Approximation EV face à raise :
    Modéliser récursivement l'EV face à un raise nécessiterait un arbre de
    jeu complet (solver). On utilise une approximation conservative :
    EV_face_à_raise ≈ -B (on fold notre bet, perte nette = montant investit).
    Cette approximation pénalise légèrement les sizings qui induisent beaucoup
    de raises, ce qui est cohérent avec l'approche exploitative.

    Une version future (phase 6+) pourrait calculer récursivement l'EV du
    call/reraise face au raise adverse.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.best_response.frequency_model import (
    FrequencyModel,
    FoldCallRaiseResult,
    PlayerRange,
)

logger = logging.getLogger(__name__)

# Nombre de simulations Monte Carlo pour le calcul d'equity
_MC_SIMS_EV = 5000   # compromis vitesse/précision pour le calcul d'EV
                     # (vs 10 000 pour la décision finale de l'EHSBot)

# Sizings de bet évalués par défaut (fraction du pot)
DEFAULT_SIZINGS = [0.0, 0.33, 0.50, 0.75, 1.00, 2.00]
# 0.0 = check, 2.0 = overbet (allin approximé si stack < 2×pot)


# =============================================================================
# Résultat d'EV pour une action
# =============================================================================

@dataclass
class EVResult:
    """
    EV calculée pour une action spécifique.

    Attributes:
        action      : type d'action ('fold', 'check', 'call', 'bet', 'raise', 'allin')
        sizing_pct  : sizing en fraction du pot (0 pour fold/check/call)
        amount      : montant en chips
        ev          : Expected Value en chips (peut être négative)
        ev_fold     : composante EV du fold adverse (p_fold × pot)
        ev_call     : composante EV du call adverse
        ev_raise    : composante EV du raise adverse (approx conservative)
        equity_call : notre equity vs la call_range (0 si pas de call possible)
        p_fold      : probabilité de fold adverse
        p_call      : probabilité de call adverse
        p_raise     : probabilité de raise adverse
        n_mc_sims   : simulations Monte Carlo effectuées
    """
    action:      str
    sizing_pct:  float
    amount:      float
    ev:          float
    ev_fold:     float   = 0.0
    ev_call:     float   = 0.0
    ev_raise:    float   = 0.0
    equity_call: float   = 0.0
    p_fold:      float   = 0.0
    p_call:      float   = 0.0
    p_raise:     float   = 0.0
    n_mc_sims:   int     = 0

    def __str__(self) -> str:
        if self.action in ('fold', 'check'):
            return f"{self.action:6s}           EV={self.ev:+8.2f}"
        if self.action == 'call':
            return (f"call            EV={self.ev:+8.2f} "
                    f"(equity={self.equity_call:.1%})")
        return (
            f"{self.action:5s} {self.sizing_pct:5.0%} pot  "
            f"EV={self.ev:+8.2f}  "
            f"(fold={self.p_fold:.1%} call={self.p_call:.1%} "
            f"raise={self.p_raise:.1%} eq={self.equity_call:.1%})"
        )


# =============================================================================
# EV Calculator
# =============================================================================

class EVCalculator:
    """
    Calcule l'EV de chaque action possible dans une situation de jeu donnée.

    Utilise le FrequencyModel pour les fréquences de réponse adverse et
    le MonteCarloEngine pour l'equity vs les ranges qui callent et raisent.

    Usage :
        calc = EVCalculator(mc_engine, freq_model)
        results = calc.compute_all_actions(
            our_hand, board, distribution, archetype,
            pot, to_call, stack, street
        )
        best = results[0]  # meilleure action (tri décroissant par EV)
    """

    def __init__(
        self,
        mc_engine,              # MonteCarloEngine (poker_engine C++)
        frequency_model: FrequencyModel,
        n_mc_sims: int = _MC_SIMS_EV,
    ):
        """
        Args:
            mc_engine       : instance de MonteCarloEngine pour le calcul d'equity
            frequency_model : FrequencyModel pré-initialisé
            n_mc_sims       : simulations Monte Carlo par calcul d'equity
        """
        self._mc       = mc_engine
        self._freq     = frequency_model
        self._n_sims   = n_mc_sims

        logger.debug(
            "EVCalculator initialisé (mc_sims=%d)", n_mc_sims
        )

    # =========================================================================
    # Interface principale
    # =========================================================================

    def compute_all_actions(
        self,
        our_hand:     List[str],
        board:        List[str],
        distribution: Dict[str, float],
        archetype:    str,
        pot:          float,
        to_call:      float,
        stack:        float,
        street:       str       = 'flop',
        sizings:      List[float] = None,
        n_opponents:  int       = 1,
    ) -> List[EVResult]:
        """
        Calcule l'EV de toutes les actions possibles et les trie par EV.

        Si to_call > 0 (on fait face à un bet adverse) :
            Actions évaluées : fold, call, raise (sizings × pot)
        Si to_call == 0 (on a l'initiative) :
            Actions évaluées : check, bet (sizings × pot), allin

        Args:
            our_hand     : nos deux cartes (['Ah', 'Kd'])
            board        : cartes du board (3 à 5 cartes)
            distribution : Dict[combo, proba] depuis RangeEstimator
            archetype    : archétype estimé de l'adversaire
            pot          : pot avant notre action
            to_call      : montant à suivre (0 si on a l'initiative)
            stack        : notre stack effectif
            street       : 'flop' | 'turn' | 'river'
            sizings      : sizings à évaluer (défaut: DEFAULT_SIZINGS)
            n_opponents  : adversaires actifs

        Returns:
            List[EVResult] triée par EV décroissante.
        """
        if sizings is None:
            sizings = DEFAULT_SIZINGS

        results: List[EVResult] = []

        # ── Cas 1 : on fait face à un bet adverse (to_call > 0) ───────────────
        if to_call > 0:
            results.append(self._ev_fold())
            results.append(self._ev_call(
                our_hand, board, distribution, pot, to_call, n_opponents
            ))
            # Raises : sizings × pot (après avoir callé)
            for sizing in sizings:
                if sizing <= 0:
                    continue
                raise_total = to_call + sizing * (pot + to_call)
                if raise_total > stack:
                    raise_total = stack  # allin
                results.append(self._ev_raise(
                    our_hand, board, distribution, archetype,
                    pot, to_call, raise_total, stack, n_opponents, street
                ))

        # ── Cas 2 : on a l'initiative (to_call == 0) ──────────────────────────
        else:
            results.append(self._ev_check(
                our_hand, board, distribution, pot, n_opponents
            ))
            for sizing in sizings:
                if sizing <= 0:
                    continue
                bet_amount = min(sizing * pot, stack)
                if bet_amount <= 0:
                    continue
                # Allin si sizing dépasse le stack
                is_allin = (bet_amount >= stack)
                results.append(self._ev_bet(
                    our_hand, board, distribution, archetype,
                    pot, bet_amount, sizing if not is_allin else (stack / pot),
                    stack, n_opponents, street,
                    action_label='allin' if is_allin else 'bet',
                ))

        # Trier par EV décroissante
        results.sort(key=lambda r: r.ev, reverse=True)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "EVCalculator résultats (%s, arch=%s, pot=%.0f, to_call=%.0f):",
                street, archetype, pot, to_call,
            )
            for r in results:
                logger.debug("  %s", r)

        return results

    # =========================================================================
    # Calcul d'EV par type d'action
    # =========================================================================

    def _ev_fold(self) -> EVResult:
        """EV du fold = 0 (on perd uniquement ce qu'on a déjà investi)."""
        return EVResult(
            action='fold', sizing_pct=0.0, amount=0.0, ev=0.0
        )

    def _ev_check(
        self,
        our_hand:     List[str],
        board:        List[str],
        distribution: Dict[str, float],
        pot:          float,
        n_opponents:  int,
    ) -> EVResult:
        """
        EV d'un check.

        Approximation : EV(check) = equity_totale × pot
        C'est une borne inférieure conservative — l'adversaire peut bet après
        notre check, mais on traite ça comme s'il checkait aussi (street suivante).
        """
        equity = self._compute_equity(our_hand, board, distribution, n_opponents)
        ev_check = equity * pot

        return EVResult(
            action='check',
            sizing_pct=0.0,
            amount=0.0,
            ev=ev_check,
            equity_call=equity,
            n_mc_sims=self._n_sims,
        )

    def _ev_call(
        self,
        our_hand:     List[str],
        board:        List[str],
        distribution: Dict[str, float],
        pot:          float,
        to_call:      float,
        n_opponents:  int,
    ) -> EVResult:
        """
        EV d'un call face à un bet adverse.

        EV(call) = equity × (pot + to_call) - to_call

        L'equity est calculée vs toute la distribution adverse (on ne filtre
        pas ici — si l'adversaire bet, on l'a déjà pris en compte dans le
        contexte de la situation).
        """
        equity  = self._compute_equity(our_hand, board, distribution, n_opponents)
        ev_call = equity * (pot + to_call) - to_call

        return EVResult(
            action='call',
            sizing_pct=0.0,
            amount=to_call,
            ev=ev_call,
            equity_call=equity,
            p_call=1.0,
            n_mc_sims=self._n_sims,
        )

    def _ev_bet(
        self,
        our_hand:     List[str],
        board:        List[str],
        distribution: Dict[str, float],
        archetype:    str,
        pot:          float,
        bet_amount:   float,
        bet_sizing:   float,
        stack:        float,
        n_opponents:  int,
        street:       str,
        action_label: str = 'bet',
    ) -> EVResult:
        """
        EV d'un bet de taille bet_amount dans un pot de taille pot.

        EV(bet_B) = p_fold  × pot
                  + p_call  × [equity_vs_call_range × (pot + 2B) - B]
                  + p_raise × (-B)   ← approximation conservative
        """
        # Fréquences de réponse adverse
        freq = self._freq.compute(
            distribution=distribution,
            archetype=archetype,
            board=board,
            bet_sizing=bet_sizing,
            pot=pot,
            n_opponents=n_opponents,
            street=street,
        )

        # Composante fold : on gagne le pot immédiatement
        ev_fold_component = freq.p_fold * pot

        # Composante call : equity vs la range qui calle
        ev_call_component = 0.0
        equity_call       = 0.0
        if freq.p_call > 0 and freq.call_range:
            equity_call = self._compute_equity(
                our_hand, board, freq.call_range, n_opponents
            )
            pot_after_call    = pot + 2 * bet_amount
            ev_call_component = freq.p_call * (
                equity_call * pot_after_call - bet_amount
            )

        # Composante raise : approximation conservative (on fold = -bet_amount)
        ev_raise_component = freq.p_raise * (-bet_amount)

        ev_total = ev_fold_component + ev_call_component + ev_raise_component

        return EVResult(
            action=action_label,
            sizing_pct=bet_sizing,
            amount=bet_amount,
            ev=ev_total,
            ev_fold=ev_fold_component,
            ev_call=ev_call_component,
            ev_raise=ev_raise_component,
            equity_call=equity_call,
            p_fold=freq.p_fold,
            p_call=freq.p_call,
            p_raise=freq.p_raise,
            n_mc_sims=self._n_sims if freq.p_call > 0 else 0,
        )

    def _ev_raise(
        self,
        our_hand:     List[str],
        board:        List[str],
        distribution: Dict[str, float],
        archetype:    str,
        pot:          float,
        to_call:      float,
        raise_amount: float,
        stack:        float,
        n_opponents:  int,
        street:       str,
    ) -> EVResult:
        """
        EV d'un raise face à un bet adverse.

        On modélise le raise comme un bet de raise_amount dans le pot actuel
        (pot + to_call) après avoir callé la mise adverse.
        Les fréquences de réponse sont calculées pour ce sizing effectif.
        """
        pot_after_call = pot + to_call
        raise_sizing   = raise_amount / pot_after_call if pot_after_call > 0 else 1.0

        freq = self._freq.compute(
            distribution=distribution,
            archetype=archetype,
            board=board,
            bet_sizing=raise_sizing,
            pot=pot_after_call,
            n_opponents=n_opponents,
            street=street,
        )

        ev_fold_component  = freq.p_fold  * pot_after_call
        ev_call_component  = 0.0
        equity_call        = 0.0

        if freq.p_call > 0 and freq.call_range:
            equity_call = self._compute_equity(
                our_hand, board, freq.call_range, n_opponents
            )
            pot_total         = pot_after_call + 2 * raise_amount
            ev_call_component = freq.p_call * (
                equity_call * pot_total - raise_amount
            )

        ev_raise_component = freq.p_raise * (-raise_amount)
        ev_total           = ev_fold_component + ev_call_component + ev_raise_component

        return EVResult(
            action='raise',
            sizing_pct=raise_sizing,
            amount=raise_amount,
            ev=ev_total,
            ev_fold=ev_fold_component,
            ev_call=ev_call_component,
            ev_raise=ev_raise_component,
            equity_call=equity_call,
            p_fold=freq.p_fold,
            p_call=freq.p_call,
            p_raise=freq.p_raise,
            n_mc_sims=self._n_sims if freq.p_call > 0 else 0,
        )

    # =========================================================================
    # Helper Monte Carlo
    # =========================================================================

    def _compute_equity(
        self,
        our_hand:     List[str],
        board:        List[str],
        opp_range:    Dict[str, float],
        n_opponents:  int,
    ) -> float:
        """
        Calcule notre equity via MonteCarloEngine vs une range adverse pondérée.

        Args:
            our_hand  : nos deux cartes
            board     : board courant
            opp_range : PlayerRange (combo → poids relatif)
            n_opponents : pour le calcul multiway (généralement 1 en HU)

        Returns:
            equity ∈ [0, 1]
        """
        if not opp_range or not our_hand or len(our_hand) < 2:
            return 0.5

        if self._mc is None:
            # Stub : equity neutre
            return 0.5

        try:
            result = self._mc.compute(
                our_hand,
                board,
                [opp_range],
                self._n_sims,
            )
            return result.ev
        except Exception as e:
            logger.warning("MonteCarloEngine error: %s", e)
            return 0.5

    def __repr__(self) -> str:
        return (
            f"EVCalculator("
            f"mc={'disponible' if self._mc else 'stub'}, "
            f"n_sims={self._n_sims})"
        )
