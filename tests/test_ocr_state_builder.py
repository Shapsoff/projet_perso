"""
test_ocr_state_builder.py — Tests unitaires de ocr/state_builder.py
Phase 7 — Bot Poker Académique

Placement : tests/
Commande   : depuis la racine du projet
    pytest tests/test_ocr_state_builder.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from core.game_state import ActionType, PlayerStatus, Position, Street
from ocr.state_builder import StateBuilderError, TableStateBuilder
from ocr.vision_types import CardRead, SeatRead, TableRead


def c(s: str) -> CardRead:
    return CardRead(s[0], s[1])


def seat(idx, stack, bet=0.0, cards=None, active=True, allin=False,
         dealer=False, to_act=False, occupied=True, confidence=1.0):
    return SeatRead(
        seat_index=idx, is_occupied=occupied, pseudo=f"P{idx}", stack=stack,
        bet_this_street=bet, hole_cards=cards or [], is_active=active,
        is_allin=allin, is_dealer=dealer, is_to_act=to_act, confidence=confidence,
    )


def three_handed_start(hero_bet=0.0, sb_bet=1.0, bb_bet=2.0, hero_to_act=True):
    """3-max : seat0=hero=BTN, seat1=SB, seat2=BB, blindes 1/2, stacks 1000."""
    return TableRead(
        timestamp=0.0,
        seats=[
            seat(0, 1000 - hero_bet, hero_bet, [c("As"), c("Kd")], dealer=True, to_act=hero_to_act),
            seat(1, 999, sb_bet),
            seat(2, 998, bb_bet),
        ],
        board_cards=[], pot=sb_bet + bb_bet,
        hero_seat_index=0, small_blind=1.0, big_blind=2.0,
    )


# =============================================================================
# start_new_hand
# =============================================================================

class TestStartNewHand:

    def test_builds_correct_players_and_positions(self):
        builder = TableStateBuilder()
        state = builder.start_new_hand(three_handed_start())
        assert {p.player_id for p in state.players} == {0, 1, 2}
        assert state.get_player(0).position == Position.BTN
        assert state.get_player(1).position == Position.SB
        assert state.get_player(2).position == Position.BB

    def test_our_id_matches_hero_seat(self):
        builder = TableStateBuilder()
        state = builder.start_new_hand(three_handed_start())
        assert state.our_id == 0
        assert state.our_player.hole_cards[0].rank == "A"

    def test_blinds_posted_correctly(self):
        builder = TableStateBuilder()
        state = builder.start_new_hand(three_handed_start())
        assert state.pot == 3
        assert state.get_player(1).bet_this_street == 1
        assert state.get_player(2).bet_this_street == 2
        assert state.get_player(1).stack == 999
        assert state.get_player(2).stack == 998
        types = [a.action_type for a in state.action_history]
        assert types == [ActionType.POST_BLIND, ActionType.POST_BLIND]

    def test_min_raise_and_to_call_after_blinds(self):
        builder = TableStateBuilder()
        state = builder.start_new_hand(three_handed_start())
        assert state.min_raise == 4      # 2x BB
        assert state.to_call == 2        # hero (BTN) doit suivre la BB

    def test_street_is_preflop(self):
        builder = TableStateBuilder()
        state = builder.start_new_hand(three_handed_start())
        assert state.street == Street.PREFLOP
        assert state.board == []

    def test_hand_id_auto_increments(self):
        builder = TableStateBuilder()
        s1 = builder.start_new_hand(three_handed_start())
        s2 = builder.start_new_hand(three_handed_start())
        assert s2.hand_id == s1.hand_id + 1

    def test_explicit_hand_id_respected(self):
        builder = TableStateBuilder()
        state = builder.start_new_hand(three_handed_start(), hand_id=42)
        assert state.hand_id == 42

    def test_heads_up_button_posts_small_blind(self):
        read = TableRead(
            timestamp=0.0,
            seats=[
                seat(0, 999, 1.0, [c("As"), c("Kd")], dealer=True, to_act=True),
                seat(1, 998, 2.0),
            ],
            board_cards=[], pot=3.0, hero_seat_index=0, small_blind=1.0, big_blind=2.0,
        )
        builder = TableStateBuilder()
        state = builder.start_new_hand(read)
        assert state.get_player(0).position == Position.BTN
        assert state.get_player(0).bet_this_street == 1   # BTN=SB en heads-up
        assert state.get_player(1).bet_this_street == 2

    def test_rejects_non_empty_board(self):
        read = three_handed_start()
        read.board_cards = [c("2c"), c("7d"), c("Jh")]
        builder = TableStateBuilder()
        with pytest.raises(StateBuilderError):
            builder.start_new_hand(read)

    def test_rejects_missing_hero_cards(self):
        read = three_handed_start()
        read.seats[0] = seat(0, 1000, 0.0, cards=[], dealer=True, to_act=True)
        builder = TableStateBuilder()
        with pytest.raises(StateBuilderError):
            builder.start_new_hand(read)

    def test_rejects_missing_blinds(self):
        read = three_handed_start()
        read.small_blind = None
        builder = TableStateBuilder()
        with pytest.raises(StateBuilderError):
            builder.start_new_hand(read)

    def test_rejects_missing_dealer(self):
        read = three_handed_start()
        for s in read.seats:
            s.is_dealer = False
        builder = TableStateBuilder()
        with pytest.raises(StateBuilderError):
            builder.start_new_hand(read)

    def test_rejects_too_few_active_players(self):
        read = TableRead(
            timestamp=0.0,
            seats=[seat(0, 1000, 0.0, [c("As"), c("Kd")], dealer=True, to_act=True)],
            board_cards=[], pot=0.0, hero_seat_index=0, small_blind=1.0, big_blind=2.0,
        )
        builder = TableStateBuilder()
        with pytest.raises(StateBuilderError):
            builder.start_new_hand(read)

    def test_failed_start_keeps_previous_state_intact(self):
        builder = TableStateBuilder()
        state1 = builder.start_new_hand(three_handed_start())
        bad_read = three_handed_start()
        bad_read.small_blind = None
        with pytest.raises(StateBuilderError):
            builder.start_new_hand(bad_read)
        assert builder.current_state is state1

    def test_duplicate_cards_rejected(self):
        read = three_handed_start()
        read.seats[0] = seat(0, 1000, 0.0, [c("As"), c("As")], dealer=True, to_act=True)
        builder = TableStateBuilder()
        with pytest.raises(StateBuilderError):
            builder.start_new_hand(read)


# =============================================================================
# ingest — actions preflop
# =============================================================================

class TestIngestPreflop:

    def _started(self):
        builder = TableStateBuilder()
        builder.start_new_hand(three_handed_start())
        return builder

    def test_ingest_without_hand_returns_none(self):
        builder = TableStateBuilder()
        result = builder.ingest(three_handed_start())
        assert result is None

    def test_hero_raise_detected(self):
        builder = self._started()
        read = three_handed_start(hero_bet=6.0, hero_to_act=False)
        read.seats[1].is_to_act = True
        state = builder.ingest(read)
        last = state.action_history[-1]
        assert last.action_type == ActionType.RAISE
        assert last.player_id == 0
        assert last.amount == 6
        assert state.pot == 3 + 6
        assert state.min_raise == 10   # 6 + max(6-2, 2)

    def test_rejected_frame_keeps_state_unchanged(self):
        builder = self._started()
        before = builder.current_state
        bad_read = three_handed_start(hero_bet=6.0, hero_to_act=False)
        bad_read.pot = -50.0   # invalide
        result = builder.ingest(bad_read)
        assert result is before
        assert builder.current_state is before

    def test_fold_updates_player_status(self):
        builder = self._started()
        read = three_handed_start(hero_bet=6.0, hero_to_act=False)
        read.seats[1].is_to_act = True
        builder.ingest(read)

        fold_read = three_handed_start(hero_bet=6.0, hero_to_act=False)
        fold_read.seats[1] = seat(1, 999, 1.0, active=False)
        fold_read.seats[2].is_to_act = True
        state = builder.ingest(fold_read)
        assert state.get_player(1).status == PlayerStatus.FOLDED
        assert state.n_active == 2


# =============================================================================
# ingest — transitions de street
# =============================================================================

class TestStreetTransitions:

    def _preflop_completed(self):
        """3-max, tout le monde a callé preflop, prêt pour le flop."""
        builder = TableStateBuilder()
        builder.start_new_hand(three_handed_start())
        r2 = three_handed_start(hero_bet=2.0, hero_to_act=False)  # hero limp/call à 2 (BB)
        r2.seats[1].is_to_act = True
        builder.ingest(r2)
        r3 = three_handed_start(hero_bet=2.0, sb_bet=2.0, hero_to_act=False)  # SB complète à 2
        r3.seats[2].is_to_act = True
        builder.ingest(r3)
        return builder

    def test_board_and_street_updated(self):
        builder = self._preflop_completed()
        flop_read = TableRead(
            timestamp=5.0,
            seats=[
                seat(0, 998, 0.0, [c("As"), c("Kd")], dealer=True, to_act=False),
                seat(1, 998, 0.0, to_act=True),
                seat(2, 998, 0.0),
            ],
            board_cards=[c("2c"), c("7d"), c("Jh")],
            pot=6.0, hero_seat_index=0, small_blind=1.0, big_blind=2.0,
        )
        state = builder.ingest(flop_read)
        assert state.street == Street.FLOP
        assert state.board_str() == ["2c", "7d", "Jh"]
        assert all(p.bet_this_street == 0 for p in state.players)
        assert state.to_call == 0
        assert state.min_raise == 2
        assert state.aggressor_id is None

    def test_bet_already_present_on_first_flop_frame_is_caught(self):
        # Cas limite : le polling a manqué la frame "flop tout juste
        # distribué, personne n'a agi" et cette frame montre déjà un bet.
        builder = self._preflop_completed()
        flop_read = TableRead(
            timestamp=5.0,
            seats=[
                seat(0, 993, 5.0, [c("As"), c("Kd")], dealer=True, to_act=False),
                seat(1, 998, 0.0, to_act=True),
                seat(2, 998, 0.0),
            ],
            board_cards=[c("2c"), c("7d"), c("Jh")],
            pot=11.0, hero_seat_index=0, small_blind=1.0, big_blind=2.0,
        )
        state = builder.ingest(flop_read)
        last = state.action_history[-1]
        assert last.player_id == 0
        assert last.action_type == ActionType.RAISE
        assert last.amount == 5
        assert state.pot == 6 + 5

    def test_skips_directly_to_turn_if_flop_missed(self):
        builder = self._preflop_completed()
        turn_read = TableRead(
            timestamp=5.0,
            seats=[
                seat(0, 998, 0.0, [c("As"), c("Kd")], dealer=True),
                seat(1, 998, 0.0, to_act=True), seat(2, 998, 0.0),
            ],
            board_cards=[c("2c"), c("7d"), c("Jh"), c("4s")],
            pot=6.0, hero_seat_index=0, small_blind=1.0, big_blind=2.0,
        )
        state = builder.ingest(turn_read)
        assert state.street == Street.TURN
        assert len(state.board) == 4


# =============================================================================
# to_call / min_raise / aggressor_id — cohérence générale
# =============================================================================

class TestBookkeeping:

    def test_to_call_zero_when_hero_is_last_aggressor(self):
        builder = TableStateBuilder()
        builder.start_new_hand(three_handed_start())
        read = three_handed_start(hero_bet=6.0, hero_to_act=False)
        read.seats[1].is_to_act = True
        state = builder.ingest(read)
        assert state.to_call == 0

    def test_aggressor_id_set_on_raise(self):
        builder = TableStateBuilder()
        builder.start_new_hand(three_handed_start())
        read = three_handed_start(hero_bet=6.0, hero_to_act=False)
        read.seats[1].is_to_act = True
        state = builder.ingest(read)
        assert state.aggressor_id == 0

    def test_aggressor_id_not_touched_by_call(self):
        builder = TableStateBuilder()
        builder.start_new_hand(three_handed_start())
        r1 = three_handed_start(hero_bet=6.0, hero_to_act=False)
        r1.seats[1].is_to_act = True
        builder.ingest(r1)

        r2 = three_handed_start(hero_bet=6.0, sb_bet=6.0, hero_to_act=False)
        r2.seats[1].is_to_act = False
        r2.seats[2].is_to_act = True
        state = builder.ingest(r2)
        assert state.aggressor_id == 0   # toujours hero, pas le caller


# =============================================================================
# showdown reveals
# =============================================================================

class TestShowdownSync:

    def test_opponent_cards_captured_when_revealed(self):
        builder = TableStateBuilder()
        builder.start_new_hand(three_handed_start())
        read = three_handed_start(hero_bet=0.0, hero_to_act=False)
        read.seats[1] = seat(1, 999, 1.0, cards=[c("Qc"), c("Qh")])  # showdown
        state = builder.ingest(read)
        assert [str(x) for x in state.get_player(1).hole_cards] == ["Qc", "Qh"]

    def test_hero_cards_never_overwritten(self):
        builder = TableStateBuilder()
        state = builder.start_new_hand(three_handed_start())
        original = list(state.our_player.hole_cards)
        read = three_handed_start(hero_bet=0.0, hero_to_act=False)
        state = builder.ingest(read)
        assert state.our_player.hole_cards == original
