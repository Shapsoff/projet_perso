"""
vision_types.py — Contrat de données Vision -> OCR
Phase 7 — Bot Poker Académique

Ce module ne contient AUCUNE logique de capture d'écran ni de template
matching. Il définit uniquement le format que la couche vision (à
construire séparément, une fois les assets du client cible disponibles :
screenshots, templates de cartes, calibration des zones par résolution)
doit produire à chaque frame.

Séparer ce contrat de son implémentation permet de développer et tester
tout le reste du pipeline OCR (validation, inférence d'actions,
construction du GameState, cycle de vie de la main) sans dépendre de
pixels réels — exactement comme core/game_state.py sert de contrat entre
best_response_engine.py et core/simulator.py.

Convention numérique : tous les montants ici sont des `float`, pas des
`int` comme dans core/game_state.py. Une lecture de texte brute peut
produire des valeurs non entières (arrondis d'affichage, devises à
centimes) ; c'est à state_builder.py de trancher et de caster en `int`
au moment de remplir un core.game_state.GameState, jamais à la vision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from core.deck import RANKS, SUITS


@dataclass(frozen=True)
class CardRead:
    """
    Lecture d'une carte à un instant t.

    rank/suit = None signifie "carte non visible" (dos de carte, main
    adverse non showdownée) — un état normal et attendu, à distinguer
    d'une lecture ratée (carte censée être visible mais confidence
    basse), qui elle doit être filtrée par validation.py en amont.
    """
    rank: Optional[str] = None
    suit: Optional[str] = None
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.rank is not None and self.rank not in RANKS:
            raise ValueError(f"rank invalide : {self.rank!r}")
        if self.suit is not None and self.suit not in SUITS:
            raise ValueError(f"suit invalide : {self.suit!r}")
        if (self.rank is None) != (self.suit is None):
            raise ValueError(
                "rank et suit doivent être tous les deux None ou tous les "
                "deux définis (carte partiellement lue = carte non lue)"
            )

    def is_visible(self) -> bool:
        return self.rank is not None and self.suit is not None

    def as_str(self) -> Optional[str]:
        return f"{self.rank}{self.suit}" if self.is_visible() else None


@dataclass
class SeatRead:
    """Lecture d'un siège physique de la table à un instant t."""

    seat_index: int                                  # position physique 0..N-1, stable tant que le joueur reste assis
    is_occupied: bool = True                          # False = siège vide
    pseudo: Optional[str] = None
    stack: Optional[float] = None
    bet_this_street: float = 0.0
    hole_cards: List[CardRead] = field(default_factory=list)  # 0 (caché), 2 (hero ou showdown)
    is_active: bool = True                            # encore dans le coup (pas foldé)
    is_allin: bool = False
    is_dealer: bool = False                           # porte le bouton
    is_to_act: bool = False                           # surbrillance/timer visible sur ce siège
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if len(self.hole_cards) not in (0, 2):
            raise ValueError(
                f"hole_cards doit contenir 0 ou 2 cartes, reçu {len(self.hole_cards)}"
            )


@dataclass
class TableRead:
    """Lecture complète de la table à un instant t (une frame)."""

    timestamp: float
    seats: List[SeatRead] = field(default_factory=list)
    board_cards: List[CardRead] = field(default_factory=list)
    pot: Optional[float] = None
    hero_seat_index: Optional[int] = None
    small_blind: Optional[float] = None
    big_blind: Optional[float] = None

    def seat(self, seat_index: int) -> Optional[SeatRead]:
        return next((s for s in self.seats if s.seat_index == seat_index), None)

    def hero(self) -> Optional[SeatRead]:
        if self.hero_seat_index is None:
            return None
        return self.seat(self.hero_seat_index)

    def occupied_seats(self) -> List[SeatRead]:
        return [s for s in self.seats if s.is_occupied]

    def active_seats(self) -> List[SeatRead]:
        """Sièges occupés encore en jeu (pas foldé) — inclut les all-in."""
        return [s for s in self.seats if s.is_occupied and s.is_active]

    def dealer_seat_index(self) -> Optional[int]:
        for s in self.seats:
            if s.is_occupied and s.is_dealer:
                return s.seat_index
        return None
