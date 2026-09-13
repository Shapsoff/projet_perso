"""
test_ocr_hand_lifecycle.py — Tests unitaires de ocr/hand_lifecycle.py
Phase 7 — Bot Poker Académique

Placement : tests/
Commande   : depuis la racine du projet
    pytest tests/test_ocr_hand_lifecycle.py -v

Ce fichier teste aussi l'intégration réelle avec
core/player_db/hand_recorder.py : un HandResultData produit par
HandLifecycleTracker doit pouvoir être passé tel quel à record_hand()
sans adaptation, exactement comme promis dans hand_lifecycle.py.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from core.game_state import Action, ActionType, Street
from core.player_db.hand_recorder import record_hand
from core.player_db.player_db import PlayerDB
from ocr.hand_lifecycle import HandLifecycleTracker, HandResultData
from ocr.state_builder import TableStateBuilder
from ocr.vision_types import CardRead, SeatRead, TableRead


def c(s: str) -> CardRead:
    return CardRead(s[0], s[1])


def seat(idx, stack, bet=0.0, cards=None, active=True, dealer=False, to_act=False):
    return SeatRead(seat_index=idx, is_occupied=True, pseudo=f"P{idx}", stack=stack,
                     bet_this_street=bet, hole_cards=cards or [], is_active=active,
                     is_dealer=dealer, is_to_act=to_act)


def hand_start(hero_cards, sb_bet=1.0, bb_bet=2.0, hero_to_act=True):
    return TableRead(
        timestamp=0.0,
        seats=[
            seat(0, 1000, 0.0, hero_cards, dealer=True, to_act=hero_to_act),
            seat(1, 999, sb_bet), seat(2, 998, bb_bet),
        ],
        board_cards=[], pot=sb_bet + bb_bet,
        hero_seat_index=0, small_blind=1.0, big_blind=2.0,
    )


class TestHandLifecycleTracker:

    def test_no_new_hand_detected_before_hero_cards_seen_once(self):
        builder = TableStateBuilder()
        tracker = HandLifecycleTracker(builder)
        read = hand_start([c("As"), c("Kd")])
        assert tracker.observe(read) is None   # 1ère fois : rien à comparer

    def test_no_new_hand_while_hero_cards_unchanged(self):
        builder = TableStateBuilder()
        tracker = HandLifecycleTracker(builder)
        read = hand_start([c("As"), c("Kd")])
        tracker.observe(read)
        builder.start_new_hand(read)
        assert tracker.observe(read) is None
        assert tracker.observe(hand_start([c("As"), c("Kd")])) is None  # même main, cartes identiques

    def test_new_hand_detected_on_hero_cards_change(self):
        builder = TableStateBuilder()
        tracker = HandLifecycleTracker(builder)
        read1 = hand_start([c("As"), c("Kd")])
        tracker.observe(read1)
        builder.start_new_hand(read1)

        read2 = hand_start([c("Qc"), c("Qh")])
        finished = tracker.observe(read2)
        assert finished is not None
        assert isinstance(finished, HandResultData)

    def test_finalized_result_contains_previous_hand_actions(self):
        builder = TableStateBuilder()
        tracker = HandLifecycleTracker(builder)
        read1 = hand_start([c("As"), c("Kd")])
        tracker.observe(read1)
        builder.start_new_hand(read1)

        raise_read = hand_start([c("As"), c("Kd")], hero_to_act=False)
        raise_read.seats[0] = seat(0, 994, 6.0, [c("As"), c("Kd")], dealer=True)
        raise_read.seats[1].is_to_act = True
        builder.ingest(raise_read)

        read2 = hand_start([c("Qc"), c("Qh")])
        finished = tracker.observe(read2)
        types = [a.action_type for a in finished.actions]
        assert types == [ActionType.POST_BLIND, ActionType.POST_BLIND, ActionType.RAISE]

    def test_finalize_does_not_mutate_builder(self):
        builder = TableStateBuilder()
        tracker = HandLifecycleTracker(builder)
        read1 = hand_start([c("As"), c("Kd")])
        tracker.observe(read1)
        state_before = builder.start_new_hand(read1)

        read2 = hand_start([c("Qc"), c("Qh")])
        tracker.observe(read2)
        assert builder.current_state is state_before   # observe() ne touche pas au builder

    def test_showdown_excludes_folded_players(self):
        builder = TableStateBuilder()
        tracker = HandLifecycleTracker(builder)
        read1 = hand_start([c("As"), c("Kd")])
        tracker.observe(read1)
        builder.start_new_hand(read1)

        # seat1 (SB) foldé + son "hole_cards" ne sera jamais vu -> jamais
        # dans showdown, même si un client bugué les montrait par erreur.
        fold_read = hand_start([c("As"), c("Kd")], hero_to_act=False)
        fold_read.seats[1] = seat(1, 999, 1.0, active=False)
        fold_read.seats[2].is_to_act = True
        builder.ingest(fold_read)

        read2 = hand_start([c("Qc"), c("Qh")])
        finished = tracker.observe(read2)
        assert 1 not in finished.showdown

    def test_reset_clears_tracking(self):
        builder = TableStateBuilder()
        tracker = HandLifecycleTracker(builder)
        read1 = hand_start([c("As"), c("Kd")])
        tracker.observe(read1)
        builder.start_new_hand(read1)
        tracker.reset()

        # Après reset, même un changement de cartes ne déclenche rien tant
        # qu'on n'a pas revu une 1ère fois des cartes de référence.
        read2 = hand_start([c("Qc"), c("Qh")])
        assert tracker.observe(read2) is None


class TestHandResultDataIntegratesWithRecordHand:
    """
    Vérifie que HandResultData est un HandResultLike valide au sens de
    core/player_db/hand_recorder.py::record_hand — pas seulement en
    théorie (duck-typing), mais en l'appelant réellement.
    """

    def test_record_hand_accepts_hand_result_data_directly(self, tmp_path):
        db = PlayerDB(db_path=str(tmp_path / "test_player_db.sqlite3"))

        result = HandResultData(
            hand_id=1,
            actions=[
                Action(1, ActionType.POST_BLIND, 1, Street.PREFLOP),
                Action(2, ActionType.POST_BLIND, 2, Street.PREFLOP),
                Action(1, ActionType.RAISE, 6, Street.PREFLOP),
                Action(2, ActionType.CALL, 4, Street.PREFLOP),
            ],
            showdown={},
            board=[],
        )

        # Ne doit lever aucune exception : c'est le contrat central de ce module.
        record_hand(
            db=db,
            hand_result=result,
            board=[],
            player_id_map={1: "ocr:villain1", 2: "ocr:villain2"},
            position_map={1: "BTN", 2: "BB"},
            our_player_id=0,
        )
        row = db.get_player_row("ocr:villain1")
        assert row is not None
