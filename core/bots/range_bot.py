"""
range_bot.py — RangeBots : bots à range fixe pour validation du Range Estimator
Phase 3 — Bot Poker Académique

Trois archétypes × trois mutations = 9 RangeBots distincts.
Ces bots jouent de façon déterministe et prévisible :
  - Preflop : joue uniquement les mains dans sa range, fold le reste
  - Postflop : bet/raise si EHS > seuil_bet, call si EHS > seuil_call, sinon fold/check
  - SPR très bas : push si la main est dans la range preflop

Le comportement déterministe est voulu : il fournit une vérité terrain
claire pour valider la précision du Range Estimator en phase 4.

Limite connue et acceptée :
  Un Range Estimator validé contre ces bots l'est en contexte contrôlé.
  La robustesse contre des joueurs réels (mixed strategies, adaptation)
  est une problématique de phase ultérieure.

Usage:
    from core.bots.range_bot import RangeBot, make_range_bot

    bot = make_range_bot('TAG', mutation=0)
    action = bot.decide(game_state)

    # Ou directement via la factory
    all_bots = make_all_range_bots()
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Callable
import logging

try:
    import poker_engine
    _POKER_ENGINE_AVAILABLE = True
except ImportError:
    poker_engine = None
    _POKER_ENGINE_AVAILABLE = False

from core.bots.range_definitions import (
    ArchetypeConfig,
    get_range,
    combo_in_range,
    hand_to_class,
    ARCHETYPES,
    MUTATIONS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Action type (réutilise le même que EHSBot pour compatibilité simulateur)
# =============================================================================

@dataclass
class Action:
    """Action retournée par le RangeBot."""
    action_type: str    # 'fold', 'check', 'call', 'bet', 'raise', 'allin'
    amount:      float = 0.0
    sizing_pct:  float = 0.0

    def __str__(self):
        if self.action_type in ('fold', 'check'):
            return self.action_type
        if self.action_type == 'call':
            return f"call {self.amount:.0f}"
        if self.action_type == 'allin':
            return f"allin {self.amount:.0f}"
        return f"{self.action_type} {self.amount:.0f} ({self.sizing_pct*100:.0f}% pot)"


# =============================================================================
# RangeBot
# =============================================================================

class RangeBot:
    """
    Bot à range fixe déterministe.

    Logique de décision :
      PREFLOP :
        si main ∈ range preflop → raise standard (3x BB)
        sinon → fold
        si SPR < 2 et main ∈ range → allin

      POSTFLOP (TAG / LAG — aggressive) :
        si SPR < 2 → allin
        si EHS > seuil_bet → bet 50% pot (ou raise 2.5x)
        si EHS > seuil_call et face à une mise → call
        sinon → check ou fold

      POSTFLOP (CALLING_STATION — passive) :
        si SPR < 2 → call ou allin
        si EHS > seuil_bet → bet 33% pot (rare — seuil élevé)
        si EHS > seuil_call → call
        sinon → check ou fold
    """

    def __init__(self, config: ArchetypeConfig, n_sims: int = 1000):
        """
        Args:
            config : ArchetypeConfig depuis range_definitions.get_range()
            n_sims : simulations Monte Carlo pour le calcul EHS postflop
        """
        self.config   = config
        self.n_sims   = n_sims
        self.name     = f"RangeBot_{config.archetype}_mut{config.mutation}"

        if _POKER_ENGINE_AVAILABLE:
            self._ehs_calc = poker_engine.EHSCalculator(seed=42, use_openmp=False)
        else:
            self._ehs_calc = None
            logger.warning("%s : poker_engine non disponible, EHS stub actif", self.name)

        # Sizing postflop selon le profil
        self._bet_sizing = 0.33 if config.aggression == 'passive' else 0.50

        logger.debug(
            "%s initialisé : %d combos (%.1f%%) seuil_bet=%.2f seuil_call=%.2f",
            self.name, len(config.preflop_combos), config.range_pct * 100,
            config.postflop_ehs_threshold, config.postflop_ehs_call
        )

    # =========================================================================
    # Point d'entrée principal
    # =========================================================================

    def decide(self, game_state: dict) -> Action:
        """
        Prend une décision pour l'état de jeu courant.

        Args:
            game_state : dict au format normalisé (section 4.4)

        Returns:
            Action déterministe selon la range et les seuils EHS.
        """
        hand    = game_state.get('hand', [])
        board   = game_state.get('board', [])
        street  = game_state.get('street', 'preflop')
        pot     = float(game_state.get('pot', 10))
        to_call = float(game_state.get('to_call', 0))
        stack   = float(game_state.get('stack', 1000))
        players = game_state.get('players', [])
        bb      = float(game_state.get('big_blind', 10))

        n_opponents = max(len(players), 1)

        # SPR
        spr = self._compute_spr(stack, pot, players)

        if street == 'preflop':
            return self._decide_preflop(hand, pot, to_call, stack, bb, spr)
        else:
            return self._decide_postflop(
                hand, board, pot, to_call, stack, spr, n_opponents
            )

    # =========================================================================
    # Décision preflop
    # =========================================================================

    def _decide_preflop(
        self,
        hand:    List[str],
        pot:     float,
        to_call: float,
        stack:   float,
        bb:      float,
        spr:     float,
    ) -> Action:
        """
        Preflop : raise si main dans la range, fold sinon.
        SPR < 2 : allin directement.
        """
        if len(hand) < 2:
            return Action(action_type='fold')

        in_range = combo_in_range(hand[0], hand[1], self.config.preflop_combos)

        if not in_range:
            return Action(action_type='fold')

        # Main dans la range
        if spr < 2.0:
            # Commitment forcé
            return Action(action_type='allin', amount=min(stack, pot * 3))

        # Raise standard : 3x BB ou 2.5x la mise adverse
        if to_call > 0:
            raise_amount = min(to_call * 2.5 + pot * 0.5, stack)
            return Action(action_type='raise', amount=raise_amount, sizing_pct=0.5)
        else:
            open_amount = min(3 * bb, stack)
            return Action(action_type='raise', amount=open_amount, sizing_pct=0.0)

    # =========================================================================
    # Décision postflop
    # =========================================================================

    def _decide_postflop(
        self,
        hand:        List[str],
        board:       List[str],
        pot:         float,
        to_call:     float,
        stack:       float,
        spr:         float,
        n_opponents: int,
    ) -> Action:
        """
        Postflop : bet/raise/call selon EHS vs seuils de l'archétype.
        """
        # SPR très bas : logique de commitment (section 3.4 du document)
        if spr < 2.0:
            if self.config.aggression == 'aggressive':
                return Action(action_type='allin', amount=min(stack, pot * 3))
            else:
                # Calling Station : call ou allin selon si face à une mise
                if to_call > 0:
                    return Action(action_type='call', amount=min(to_call, stack))
                return Action(action_type='allin', amount=min(stack, pot * 3))

        # Calcul EHS
        ehs = self._compute_ehs(hand, board, n_opponents)

        # ── Aggressive (TAG / LAG) ────────────────────────────────────────────
        if self.config.aggression == 'aggressive':
            if ehs > self.config.postflop_ehs_threshold:
                if to_call > 0:
                    raise_amount = min(to_call * 2.5 + pot * self._bet_sizing, stack)
                    return Action(action_type='raise', amount=raise_amount,
                                  sizing_pct=self._bet_sizing)
                bet_amount = min(self._bet_sizing * pot, stack)
                return Action(action_type='bet', amount=bet_amount,
                              sizing_pct=self._bet_sizing)

            if ehs > self.config.postflop_ehs_call:
                if to_call > 0:
                    # Vérifier pot odds
                    pot_odds = to_call / (pot + to_call)
                    if ehs > pot_odds:
                        return Action(action_type='call',
                                      amount=min(to_call, stack))
                    return Action(action_type='fold')
                return Action(action_type='check')

            # EHS faible
            if to_call > 0:
                return Action(action_type='fold')
            return Action(action_type='check')

        # ── Passive (Calling Station) ─────────────────────────────────────────
        else:
            # Bet uniquement avec mains très fortes (seuil élevé = 0.75)
            if ehs > self.config.postflop_ehs_threshold and to_call == 0:
                bet_amount = min(self._bet_sizing * pot, stack)
                return Action(action_type='bet', amount=bet_amount,
                              sizing_pct=self._bet_sizing)

            # Call avec EHS > seuil_call (0.30) — call très large
            if ehs > self.config.postflop_ehs_call:
                if to_call > 0:
                    return Action(action_type='call', amount=min(to_call, stack))
                return Action(action_type='check')

            # EHS très faible : fold ou check
            if to_call > 0:
                return Action(action_type='fold')
            return Action(action_type='check')

    # =========================================================================
    # Helpers
    # =========================================================================

    def _compute_ehs(
        self,
        hand:        List[str],
        board:       List[str],
        n_opponents: int,
    ) -> float:
        """Calcule la Hand Strength via poker_engine ou stub."""
        if self._ehs_calc is not None and len(hand) == 2 and len(board) >= 3:
            try:
                result = self._ehs_calc.calculate_multiway(
                    hand, board, n_opponents, self.n_sims
                )
                return result.EHS
            except Exception as e:
                logger.warning("%s EHS error: %s", self.name, e)
                return 0.5
        # Stub : equity neutre
        return 0.5

    def _compute_spr(
        self,
        our_stack: float,
        pot:       float,
        players:   List[dict],
    ) -> float:
        """Calcule le SPR (Stack-to-Pot Ratio)."""
        if pot <= 0:
            return float('inf')
        opp_stacks = [p.get('stack', our_stack) for p in players
                      if p.get('stack', 0) > 0]
        eff_stack = min(our_stack, min(opp_stacks)) if opp_stacks else our_stack
        return eff_stack / pot

    def _is_hand_in_range(self, hand: List[str]) -> bool:
        """Vérifie si une main est dans la range preflop."""
        if len(hand) < 2:
            return False
        return combo_in_range(hand[0], hand[1], self.config.preflop_combos)

    # =========================================================================
    # Interface simulateur
    # =========================================================================

    def get_action(self, game_state: dict) -> dict:
        """Interface JSON-compatible pour le simulateur."""
        action = self.decide(game_state)
        return {'action': action.action_type, 'amount': action.amount}

    def reset(self):
        """Réinitialise l'état entre les mains (stateless)."""
        pass

    def __repr__(self):
        return (f"RangeBot(archetype={self.config.archetype}, "
                f"mutation={self.config.mutation}, "
                f"combos={len(self.config.preflop_combos)}, "
                f"range={self.config.range_pct:.1%})")


# =============================================================================
# Factory
# =============================================================================

def make_range_bot(
    archetype: str,
    mutation:  int = 0,
    n_sims:    int = 1000,
) -> RangeBot:
    """
    Crée un RangeBot depuis un archétype et une mutation.

    Args:
        archetype : 'TAG', 'LAG', ou 'CALLING_STATION'
        mutation  : 0 (base), 1, ou 2
        n_sims    : simulations Monte Carlo EHS (défaut: 1000)

    Returns:
        RangeBot configuré et prêt à jouer.

    Exemple:
        bot = make_range_bot('TAG', mutation=0)
        bot = make_range_bot('LAG', mutation=2, n_sims=500)
    """
    config = get_range(archetype, mutation)
    return RangeBot(config, n_sims=n_sims)


def make_all_range_bots(n_sims: int = 1000) -> Dict[str, RangeBot]:
    """
    Crée les 9 RangeBots d'un coup.

    Returns:
        Dict avec clés 'TAG_0', 'TAG_1', ..., 'CALLING_STATION_2'
    """
    bots = {}
    for archetype in ARCHETYPES:
        for mutation in MUTATIONS:
            key = f"{archetype}_{mutation}"
            bots[key] = make_range_bot(archetype, mutation, n_sims)
    return bots
