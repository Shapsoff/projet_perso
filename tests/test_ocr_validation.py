"""
test_ocr_validation.py — Tests unitaires de ocr/validation.py
Phase 7 — Bot Poker Académique

Placement : tests/
Commande   : depuis la racine du projet
    pytest tests/test_ocr_validation.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr.vision_types import CardRead, SeatRead, TableRead
from ocr.validation import (
    FrameStabilizer,
    validate_board_length,
    validate_cards_unique,
    validate_confidence,
    validate_stacks,
    validate_table_read,
)


def _read(seats=None, board=None, pot=0.0):
    return TableRead(timestamp=0.0, seats=seats or [], board_cards=board or [], pot=pot)


# =============================================================================
# validate_cards_unique
# =============================================================================

class TestValidateCardsUnique:

    def test_no_duplicate_ok(self):
        read = _read(
            seats=[SeatRead(seat_index=0, hole_cards=[CardRead("A", "s"), CardRead("K", "d")])],
            board=[CardRead("2", "c"), CardRead("7", "d")],
        )
        assert validate_cards_unique(read).ok

    def test_duplicate_between_board_and_hand(self):
        read = _read(
            seats=[SeatRead(seat_index=0, hole_cards=[CardRead("A", "s"), CardRead("K", "d")])],
            board=[CardRead("A", "s")],   # doublon avec la main du siège 0
        )
        result = validate_cards_unique(read)
        assert not result.ok
        assert any("As" in e for e in result.errors)

    def test_duplicate_between_two_seats(self):
        read = _read(seats=[
            SeatRead(seat_index=0, hole_cards=[CardRead("A", "s"), CardRead("K", "d")]),
            SeatRead(seat_index=1, hole_cards=[CardRead("A", "s"), CardRead("Q", "h")]),
        ])
        assert not validate_cards_unique(read).ok

    def test_hidden_cards_never_conflict(self):
        read = _read(seats=[
            SeatRead(seat_index=0, hole_cards=[]),
            SeatRead(seat_index=1, hole_cards=[]),
        ])
        assert validate_cards_unique(read).ok


# =============================================================================
# validate_stacks
# =============================================================================

class TestValidateStacks:

    def test_positive_amounts_ok(self):
        read = _read(seats=[SeatRead(seat_index=0, stack=100.0, bet_this_street=10.0)], pot=10.0)
        assert validate_stacks(read).ok

    def test_negative_pot_rejected(self):
        read = _read(pot=-5.0)
        assert not validate_stacks(read).ok

    def test_negative_stack_rejected(self):
        read = _read(seats=[SeatRead(seat_index=0, stack=-10.0)])
        assert not validate_stacks(read).ok

    def test_negative_bet_rejected(self):
        read = _read(seats=[SeatRead(seat_index=0, bet_this_street=-1.0)])
        assert not validate_stacks(read).ok

    def test_unoccupied_seat_ignored(self):
        read = _read(seats=[SeatRead(seat_index=0, is_occupied=False, stack=-999.0)])
        assert validate_stacks(read).ok


# =============================================================================
# validate_confidence
# =============================================================================

class TestValidateConfidence:

    def test_high_confidence_ok(self):
        read = _read(seats=[SeatRead(seat_index=0, confidence=0.95)])
        assert validate_confidence(read, min_confidence=0.6).ok

    def test_low_seat_confidence_rejected(self):
        read = _read(seats=[SeatRead(seat_index=0, confidence=0.3)])
        assert not validate_confidence(read, min_confidence=0.6).ok

    def test_low_card_confidence_rejected(self):
        read = _read(seats=[SeatRead(
            seat_index=0,
            hole_cards=[CardRead("A", "s", confidence=0.2), CardRead("K", "d")],
        )])
        assert not validate_confidence(read, min_confidence=0.6).ok

    def test_low_board_card_confidence_rejected(self):
        read = _read(board=[CardRead("2", "c", confidence=0.1)])
        assert not validate_confidence(read, min_confidence=0.6).ok

    def test_unoccupied_seat_confidence_ignored(self):
        read = _read(seats=[SeatRead(seat_index=0, is_occupied=False, confidence=0.0)])
        assert validate_confidence(read, min_confidence=0.6).ok


# =============================================================================
# validate_board_length
# =============================================================================

class TestValidateBoardLength:

    def test_valid_lengths(self):
        for n in (0, 3, 4, 5):
            board = [CardRead("2", "c")] * n if n else []
            # éviter les doublons de carte pour ce test ciblé sur la longueur
            board = [CardRead(r, "c") for r in "23456789TJQKA"[:n]]
            assert validate_board_length(_read(board=board)).ok

    def test_invalid_lengths(self):
        for n in (1, 2, 6):
            board = [CardRead(r, "c") for r in "23456789TJQKA"[:n]]
            assert not validate_board_length(_read(board=board)).ok


# =============================================================================
# validate_table_read (agrégateur)
# =============================================================================

class TestValidateTableRead:

    def test_clean_read_passes(self):
        read = _read(
            seats=[SeatRead(seat_index=0, stack=100.0, confidence=0.9,
                             hole_cards=[CardRead("A", "s"), CardRead("K", "d")])],
            board=[CardRead("2", "c"), CardRead("7", "d"), CardRead("J", "h")],
            pot=20.0,
        )
        assert validate_table_read(read).ok

    def test_can_disable_confidence_check(self):
        read = _read(seats=[SeatRead(seat_index=0, confidence=0.01)])
        assert not validate_table_read(read, check_confidence=True).ok
        assert validate_table_read(read, check_confidence=False).ok

    def test_aggregates_multiple_errors(self):
        read = _read(
            seats=[SeatRead(seat_index=0, stack=-5.0,
                             hole_cards=[CardRead("A", "s"), CardRead("A", "s")])],
        )
        # carte dupliquée dans la MÊME main -> déjà invalide côté SeatRead ?
        # non : SeatRead n'interdit pas les doublons en interne, c'est le rôle
        # de validate_cards_unique. On vérifie juste qu'on cumule plusieurs erreurs.
        result = validate_table_read(read)
        assert not result.ok
        assert len(result.errors) >= 2


# =============================================================================
# FrameStabilizer
# =============================================================================

class TestFrameStabilizer:

    def test_first_value_confirmed_immediately(self):
        s = FrameStabilizer(required_repeats=2)
        value, changed = s.update("pot", 100)
        assert value == 100 and changed

    def test_none_returns_last_confirmed(self):
        s = FrameStabilizer(required_repeats=2)
        s.update("pot", 100)
        value, changed = s.update("pot", None)
        assert value == 100 and not changed

    def test_same_value_no_change_flag(self):
        s = FrameStabilizer(required_repeats=2)
        s.update("pot", 100)
        value, changed = s.update("pot", 100)
        assert value == 100 and not changed

    def test_single_noisy_frame_ignored(self):
        s = FrameStabilizer(required_repeats=2)
        s.update("pot", 100)
        value, changed = s.update("pot", 999)   # bruit ponctuel
        assert value == 100 and not changed     # pas encore confirmé

    def test_repeated_new_value_confirmed(self):
        s = FrameStabilizer(required_repeats=2)
        s.update("pot", 100)
        s.update("pot", 150)   # 1er vote
        value, changed = s.update("pot", 150)   # 2e vote -> confirmé
        assert value == 150 and changed

    def test_flip_flopping_never_confirms(self):
        s = FrameStabilizer(required_repeats=2)
        s.update("pot", 100)
        s.update("pot", 150)
        value, changed = s.update("pot", 200)   # candidat différent -> vote repart à 1
        assert value == 100 and not changed

    def test_required_repeats_higher(self):
        s = FrameStabilizer(required_repeats=3)
        s.update("pot", 100)
        s.update("pot", 150)
        v1, c1 = s.update("pot", 150)
        assert v1 == 100 and not c1
        v2, c2 = s.update("pot", 150)
        assert v2 == 150 and c2

    def test_independent_keys(self):
        s = FrameStabilizer(required_repeats=2)
        s.update("pot", 100)
        s.update(("stack", 0), 500)
        assert s.update("pot", 100) == (100, False)
        assert s.update(("stack", 0), 500) == (500, False)

    def test_reset_single_key(self):
        s = FrameStabilizer(required_repeats=2)
        s.update("pot", 100)
        s.reset("pot")
        value, changed = s.update("pot", 999)
        assert value == 999 and changed   # traité comme une première valeur

    def test_reset_all(self):
        s = FrameStabilizer(required_repeats=2)
        s.update("pot", 100)
        s.update(("stack", 0), 500)
        s.reset()
        assert s.update("pot", 999) == (999, True)
        assert s.update(("stack", 0), 42) == (42, True)

    def test_rejects_zero_required_repeats(self):
        import pytest
        with pytest.raises(ValueError):
            FrameStabilizer(required_repeats=0)
