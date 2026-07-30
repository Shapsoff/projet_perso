"""
deck.py — Représentation et manipulation d'un deck de cartes
"""
from __future__ import annotations
import random
from dataclasses import dataclass

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_VAL = {r: i for i, r in enumerate(RANKS)}  # '2'=0, 'A'=12


@dataclass(frozen=True)
class Card:
    rank: str   # '2'..'A'
    suit: str   # 'c','d','h','s'

    def __str__(self) -> str:
        return self.rank + self.suit

    def __repr__(self) -> str:
        return f"Card({self})"

    @property
    def rank_val(self) -> int:
        return RANK_VAL[self.rank]

    @classmethod
    def from_str(cls, s: str) -> "Card":
        if len(s) != 2 or s[0] not in RANKS or s[1] not in SUITS:
            raise ValueError(f"Carte invalide : {s!r}")
        return cls(s[0], s[1])


class Deck:
    """Deck standard de 52 cartes, mélangeable."""

    def __init__(self) -> None:
        self._cards: list[Card] = [
            Card(r, s) for r in RANKS for s in SUITS
        ]
        self._index = 0

    def shuffle(self, seed: int | None = None) -> None:
        rng = random.Random(seed)
        rng.shuffle(self._cards)
        self._index = 0

    def deal(self, n: int = 1) -> list[Card]:
        if self._index + n > len(self._cards):
            raise RuntimeError("Plus de cartes dans le deck")
        cards = self._cards[self._index: self._index + n]
        self._index += n
        return cards

    def deal_one(self) -> Card:
        return self.deal(1)[0]

    def remove(self, cards: list[Card]) -> None:
        """Retire des cartes spécifiques du deck (pour tests avec mains fixées)."""
        card_set = set(cards)
        self._cards = [c for c in self._cards[self._index:] if c not in card_set]
        self._index = 0

    def remaining(self) -> int:
        return len(self._cards) - self._index

    def __len__(self) -> int:
        return self.remaining()
