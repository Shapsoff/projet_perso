"""
game_state.py — Structures de données du game_state normalisé

C'est le format unique qui circule entre tous les modules :
OCR → game_state → moteur de décision → action
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from .deck import Card


# ── Enums ─────────────────────────────────────────────────────────────────────

class Street(Enum):
    PREFLOP = "preflop"
    FLOP    = "flop"
    TURN    = "turn"
    RIVER   = "river"

class Position(Enum):
    UTG  = "UTG"
    UTG1 = "UTG+1"
    MP   = "MP"
    CO   = "CO"
    BTN  = "BTN"
    SB   = "SB"
    BB   = "BB"

class ActionType(Enum):
    FOLD       = "fold"
    CHECK      = "check"
    CALL       = "call"
    RAISE      = "raise"
    ALLIN      = "allin"
    POST_BLIND = "post_blind"   # mise forcée (SB/BB) — jamais une décision
                                 # volontaire du joueur, à distinguer de RAISE
                                 # pour tous les calculs VPIP/PFR/3bet/4bet en
                                 # aval (hand_recorder.py, action_history.py,
                                 # range_estimator.py) — cf. session 6

class PlayerStatus(Enum):
    ACTIVE   = "active"    # encore en jeu cette main
    FOLDED   = "folded"    # a foldé
    ALLIN    = "allin"     # all-in, ne peut plus agir
    SITOUT   = "sitout"    # absent


# ── Action ────────────────────────────────────────────────────────────────────

@dataclass
class Action:
    player_id: int
    action_type: ActionType
    amount: int = 0          # montant mis (0 pour fold/check)
    street: Street = Street.PREFLOP

    def __str__(self) -> str:
        if self.action_type in (ActionType.FOLD, ActionType.CHECK):
            return f"P{self.player_id} {self.action_type.value}"
        return f"P{self.player_id} {self.action_type.value} {self.amount}"


# ── Statistiques de profil (vides pour l'instant, remplies phase 3) ───────────

@dataclass
class PlayerStats:
    vpip:          float = 0.0   # Voluntarily Put In Pot
    pfr:           float = 0.0   # Pre-Flop Raise
    three_bet_pct: float = 0.0
    fold_to_cbet:  float = 0.0
    aggression_factor: float = 0.0
    sample_size:   int   = 0

    @property
    def is_reliable(self) -> bool:
        """Stats fiables à partir de 30 mains observées."""
        return self.sample_size >= 30


# ── Joueur ────────────────────────────────────────────────────────────────────

@dataclass
class Player:
    player_id:  int
    name:       str
    stack:      int
    position:   Position
    status:     PlayerStatus = PlayerStatus.ACTIVE
    hole_cards: list[Card]   = field(default_factory=list)
    bet_this_street: int     = 0   # montant misé sur la rue en cours
    stats:      PlayerStats  = field(default_factory=PlayerStats)

    # Range estimée (remplie par le RangeEstimator, phase 3)
    # Format : {combo_str: probabilité}, ex {"AsKd": 0.08}
    estimated_range: dict[str, float] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.status == PlayerStatus.ACTIVE

    @property
    def is_allin(self) -> bool:
        return self.status == PlayerStatus.ALLIN

    @property
    def can_act(self) -> bool:
        return self.status in (PlayerStatus.ACTIVE,)

    def __str__(self) -> str:
        cards = " ".join(str(c) for c in self.hole_cards) if self.hole_cards else "??"
        return f"{self.name}({self.position.value}) stack={self.stack} [{cards}]"


# ── Game State ────────────────────────────────────────────────────────────────

@dataclass
class GameState:
    """
    État complet d'une main à un instant T.
    Format normalisé utilisé par tous les modules du bot.
    """
    # Identification
    hand_id:    int = 0
    session_id: int = 0

    # Joueurs
    players:    list[Player] = field(default_factory=list)
    our_id:     int = 0       # player_id de notre bot

    # Board
    board:      list[Card]   = field(default_factory=list)
    street:     Street       = Street.PREFLOP

    # Pot et mises
    pot:        int = 0
    main_pot:   int = 0
    side_pots:  list[dict]   = field(default_factory=list)

    # Action en cours
    to_call:    int = 0       # montant à suivre (0 si check possible)
    min_raise:  int = 0       # raise minimum
    aggressor_id: Optional[int] = None  # qui a misé en dernier

    # Blindes
    small_blind: int = 1
    big_blind:   int = 2

    # Historique complet de la main
    action_history: list[Action] = field(default_factory=list)

    # ── Accesseurs pratiques ──────────────────────────────────────────────────

    @property
    def our_player(self) -> Player:
        return self.get_player(self.our_id)

    @property
    def active_players(self) -> list[Player]:
        return [p for p in self.players if p.can_act or p.is_allin]

    @property
    def active_opponents(self) -> list[Player]:
        return [p for p in self.active_players if p.player_id != self.our_id]

    @property
    def n_active(self) -> int:
        return len(self.active_players)

    def get_player(self, player_id: int) -> Player:
        for p in self.players:
            if p.player_id == player_id:
                return p
        raise KeyError(f"Joueur {player_id} introuvable")

    def board_str(self) -> list[str]:
        return [str(c) for c in self.board]

    def our_hand_str(self) -> list[str]:
        return [str(c) for c in self.our_player.hole_cards]

    @property
    def effective_stack(self) -> int:
        """Stack effectif = min(notre stack, max stack adverse)."""
        opp_stacks = [p.stack for p in self.active_opponents]
        if not opp_stacks:
            return self.our_player.stack
        return min(self.our_player.stack, max(opp_stacks))

    @property
    def spr(self) -> float:
        """Stack-to-Pot Ratio."""
        if self.pot == 0:
            return float('inf')
        return self.effective_stack / self.pot

    def history_for_street(self, street: Street) -> list[Action]:
        return [a for a in self.action_history if a.street == street]

    def to_dict(self) -> dict:
        """Sérialisation en dict (pour logs, BDD, debug)."""
        return {
            "hand_id":   self.hand_id,
            "street":    self.street.value,
            "board":     self.board_str(),
            "pot":       self.pot,
            "to_call":   self.to_call,
            "our_hand":  self.our_hand_str(),
            "our_stack": self.our_player.stack,
            "spr":       round(self.spr, 2),
            "n_active":  self.n_active,
            "players": [
                {
                    "id":       p.player_id,
                    "name":     p.name,
                    "stack":    p.stack,
                    "position": p.position.value,
                    "status":   p.status.value,
                    "bet":      p.bet_this_street,
                }
                for p in self.players
            ],
            "action_history": [str(a) for a in self.action_history],
        }
