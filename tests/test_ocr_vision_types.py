"""
test_ocr_vision_types.py — Tests unitaires du contrat de données ocr/vision_types.py
Phase 7 — Bot Poker Académique

Placement : tests/
Commande   : depuis la racine du projet
    pytest tests/test_ocr_vision_types.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from ocr.vision_types import CardRead, SeatRead, TableRead


# =============================================================================
# CardRead
# =============================================================================

class TestCardRead:

    def test_visible_card(self):
        card = CardRead("A", "s")
        assert card.is_visible()
        assert card.as_str() == "As"

    def test_hidden_card(self):
        card = CardRead()
        assert not card.is_visible()
        assert card.as_str() is None

    def test_rejects_invalid_rank(self):
        with pytest.raises(ValueError):
            CardRead("1", "s")

    def test_rejects_invalid_suit(self):
        with pytest.raises(ValueError):
            CardRead("A", "x")

    def test_rejects_partial_card_rank_only(self):
        with pytest.raises(ValueError):
            CardRead("A", None)

    def test_rejects_partial_card_suit_only(self):
        with pytest.raises(ValueError):
            CardRead(None, "s")

    def test_default_confidence(self):
        assert CardRead("A", "s").confidence == 1.0

    def test_all_ranks_accepted(self):
        for r in "23456789TJQKA":
            CardRead(r, "c")  # ne doit pas lever

    def test_all_suits_accepted(self):
        for s in "cdhs":
            CardRead("A", s)  # ne doit pas lever


# =============================================================================
# SeatRead
# =============================================================================

class TestSeatRead:

    def test_zero_hole_cards_ok(self):
        seat = SeatRead(seat_index=0, hole_cards=[])
        assert seat.hole_cards == []

    def test_two_hole_cards_ok(self):
        seat = SeatRead(seat_index=0, hole_cards=[CardRead("A", "s"), CardRead("K", "d")])
        assert len(seat.hole_cards) == 2

    def test_rejects_one_hole_card(self):
        with pytest.raises(ValueError):
            SeatRead(seat_index=0, hole_cards=[CardRead("A", "s")])

    def test_rejects_three_hole_cards(self):
        with pytest.raises(ValueError):
            SeatRead(seat_index=0, hole_cards=[CardRead("A", "s")] * 3)

    def test_defaults(self):
        seat = SeatRead(seat_index=2)
        assert seat.is_occupied is True
        assert seat.is_active is True
        assert seat.is_allin is False
        assert seat.is_dealer is False
        assert seat.is_to_act is False
        assert seat.bet_this_street == 0.0


# =============================================================================
# TableRead
# =============================================================================

class TestTableRead:

    def _seats(self):
        return [
            SeatRead(seat_index=0, hole_cards=[CardRead("A", "s"), CardRead("K", "d")], is_dealer=True),
            SeatRead(seat_index=1),
            SeatRead(seat_index=2, is_occupied=False),
            SeatRead(seat_index=3, is_active=False),
        ]

    def test_seat_lookup(self):
        read = TableRead(timestamp=0.0, seats=self._seats())
        assert read.seat(1).seat_index == 1
        assert read.seat(99) is None

    def test_hero_lookup(self):
        read = TableRead(timestamp=0.0, seats=self._seats(), hero_seat_index=0)
        assert read.hero() is not None
        assert read.hero().seat_index == 0

    def test_hero_lookup_none_when_unset(self):
        read = TableRead(timestamp=0.0, seats=self._seats())
        assert read.hero() is None

    def test_occupied_seats(self):
        read = TableRead(timestamp=0.0, seats=self._seats())
        occupied = {s.seat_index for s in read.occupied_seats()}
        assert occupied == {0, 1, 3}   # seat 2 non occupé

    def test_active_seats_excludes_folded_and_unoccupied(self):
        read = TableRead(timestamp=0.0, seats=self._seats())
        active = {s.seat_index for s in read.active_seats()}
        assert active == {0, 1}        # seat 2 non occupé, seat 3 foldé

    def test_dealer_seat_index(self):
        read = TableRead(timestamp=0.0, seats=self._seats())
        assert read.dealer_seat_index() == 0

    def test_dealer_seat_index_none_when_no_dealer(self):
        seats = [SeatRead(seat_index=i) for i in range(3)]
        read = TableRead(timestamp=0.0, seats=seats)
        assert read.dealer_seat_index() is None

    def test_dealer_must_be_occupied(self):
        seats = [SeatRead(seat_index=0, is_occupied=False, is_dealer=True)]
        read = TableRead(timestamp=0.0, seats=seats)
        assert read.dealer_seat_index() is None
