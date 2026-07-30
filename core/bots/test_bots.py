"""
bots/ — Bots de test pour valider le simulateur

Niveau 0 : RandomBot  — action aléatoire parmi les légales
Niveau 1 : RuleBot    — call si équité > seuil, fold sinon
"""
from __future__ import annotations
import random
from ..game_state import GameState, Action, ActionType


# ── Niveau 0 : RandomBot ──────────────────────────────────────────────────────

class RandomBot:
    """
    Choisit aléatoirement parmi les actions légales.
    Utile comme baseline de sanité — tout bot doit battre le RandomBot.
    """
    def __init__(self, player_id: int, seed: int | None = None):
        self.player_id = player_id
        self.rng = random.Random(seed)

    def __call__(self, state: GameState) -> Action:
        player = state.get_player(self.player_id)
        can_check = state.to_call == 0

        if can_check:
            # Check ou bet aléatoire (33% du pot)
            if self.rng.random() < 0.5:
                return Action(self.player_id, ActionType.CHECK, 0)
            else:
                bet = max(1, state.pot // 3)
                bet = min(bet, player.stack)
                return Action(self.player_id, ActionType.RAISE, bet)
        else:
            choice = self.rng.choices(
                ["fold", "call", "raise"],
                weights=[0.4, 0.4, 0.2]
            )[0]
            if choice == "fold":
                return Action(self.player_id, ActionType.FOLD, 0)
            elif choice == "call":
                return Action(self.player_id, ActionType.CALL, state.to_call)
            else:
                raise_amount = state.to_call + max(state.big_blind, state.pot // 2)
                raise_amount = min(raise_amount, player.stack)
                return Action(self.player_id, ActionType.RAISE, raise_amount)


# ── Niveau 1 : RuleBot ────────────────────────────────────────────────────────

class RuleBot:
    """
    Prend des décisions basées sur la force de la main (pot odds simplifiés).

    Logique :
      - Calcule les pot odds implicites
      - Call si pot_odds > threshold (défaut 0.3)
      - Fold sinon
      - Bet/raise si main très forte (top pair+)
    """

    # Force de main approximative (préflop uniquement, à améliorer phase 2)
    STRONG_HANDS = {
        frozenset(["A", "A"]), frozenset(["K", "K"]), frozenset(["Q", "Q"]),
        frozenset(["J", "J"]), frozenset(["T", "T"]), frozenset(["A", "K"]),
    }

    def __init__(self, player_id: int, call_threshold: float = 0.30):
        self.player_id       = player_id
        self.call_threshold  = call_threshold

    def __call__(self, state: GameState) -> Action:
        player = state.get_player(self.player_id)

        # Pot odds = montant à payer / (pot + montant à payer)
        if state.to_call == 0:
            pot_odds = 0.0
        else:
            pot_odds = state.to_call / (state.pot + state.to_call)

        hand_strength = self._estimate_hand_strength(state)

        if state.to_call == 0:
            # Check ou value bet
            if hand_strength > 0.7:
                bet = min(player.stack, max(state.big_blind, state.pot // 2))
                return Action(self.player_id, ActionType.RAISE, bet)
            return Action(self.player_id, ActionType.CHECK, 0)

        # Décision face à une mise
        if hand_strength > pot_odds + self.call_threshold:
            # Main assez forte pour suivre
            if hand_strength > 0.85:
                # Très forte : relancer
                raise_size = state.to_call * 3
                raise_size = min(raise_size, player.stack)
                return Action(self.player_id, ActionType.RAISE, raise_size)
            return Action(self.player_id, ActionType.CALL, state.to_call)
        else:
            return Action(self.player_id, ActionType.FOLD, 0)

    def _estimate_hand_strength(self, state: GameState) -> float:
        """
        Estimation simplifiée de la force de la main.
        Phase 2 : remplacée par le vrai Monte Carlo C++.
        """
        player = state.get_player(self.player_id)
        if not player.hole_cards:
            return 0.5

        r1, r2 = player.hole_cards[0].rank, player.hole_cards[1].rank
        hand_set = frozenset([r1, r2])

        # Préflop
        if not state.board:
            if hand_set in self.STRONG_HANDS:
                return 0.80
            rank_vals = [player.hole_cards[0].rank_val, player.hole_cards[1].rank_val]
            avg_rank = sum(rank_vals) / 2
            return 0.3 + (avg_rank / 12) * 0.4  # 0.3 – 0.7

        # Postflop : force basique selon la paire
        board_ranks = [c.rank for c in state.board]
        if r1 in board_ranks or r2 in board_ranks:
            return 0.65   # paire
        return 0.35        # rien


# ── Niveau 2 : CallBot (toujours call/check) ─────────────────────────────────

class CallBot:
    """Call tout le temps — utile pour tester que le simulateur gère les mises."""
    def __init__(self, player_id: int):
        self.player_id = player_id

    def __call__(self, state: GameState) -> Action:
        if state.to_call == 0:
            return Action(self.player_id, ActionType.CHECK, 0)
        return Action(self.player_id, ActionType.CALL, state.to_call)
