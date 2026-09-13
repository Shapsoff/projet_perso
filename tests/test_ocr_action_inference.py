"""
test_ocr_action_inference.py — Tests unitaires de ocr/action_inference.py
Phase 7 — Bot Poker Académique

Placement : tests/
Commande   : depuis la racine du projet
    pytest tests/test_ocr_action_inference.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.game_state import ActionType, Street
from ocr.action_inference import infer_actions, post_blinds
from ocr.vision_types import CardRead, SeatRead, TableRead


def _seat(idx, bet=0.0, active=True, allin=False, to_act=False, occupied=True):
    return SeatRead(
        seat_index=idx, is_occupied=occupied, bet_this_street=bet,
        is_active=active, is_allin=allin, is_to_act=to_act,
    )


def _read(seats, board=None):
    return TableRead(timestamp=0.0, seats=seats, board_cards=board or [])


# =============================================================================
# post_blinds
# =============================================================================

class TestPostBlinds:

    def test_posts_both_blinds(self):
        read = TableRead(timestamp=0.0, seats=[_seat(0), _seat(1), _seat(2)],
                          small_blind=1.0, big_blind=2.0)
        actions = post_blinds(read, sb_seat_index=1, bb_seat_index=2)
        assert len(actions) == 2
        assert actions[0].seat_index == 1
        assert actions[0].action_type == ActionType.POST_BLIND
        assert actions[0].amount == 1.0
        assert actions[1].seat_index == 2
        assert actions[1].amount == 2.0

    def test_all_preflop_street(self):
        read = TableRead(timestamp=0.0, seats=[], small_blind=1.0, big_blind=2.0)
        actions = post_blinds(read, 0, 1)
        assert all(a.street == Street.PREFLOP for a in actions)

    def test_missing_blind_amount_skipped(self):
        read = TableRead(timestamp=0.0, seats=[], small_blind=None, big_blind=2.0)
        actions = post_blinds(read, 0, 1)
        assert len(actions) == 1
        assert actions[0].seat_index == 1


# =============================================================================
# infer_actions — fold
# =============================================================================

class TestInferFold:

    def test_detects_fold(self):
        prev = _read([_seat(0, bet=2), _seat(1, bet=2)])
        curr = _read([_seat(0, bet=2, active=False), _seat(1, bet=2)])
        actions = infer_actions(prev, curr, Street.FLOP)
        assert len(actions) == 1
        assert actions[0].seat_index == 0
        assert actions[0].action_type == ActionType.FOLD
        assert actions[0].amount == 0.0

    def test_no_fold_no_action(self):
        prev = _read([_seat(0, bet=2), _seat(1, bet=2)])
        curr = _read([_seat(0, bet=2), _seat(1, bet=2)])
        assert infer_actions(prev, curr, Street.FLOP) == []


# =============================================================================
# infer_actions — call / raise / bet d'ouverture
# =============================================================================

class TestInferCallRaise:

    def test_opening_bet_tagged_raise_with_total_amount(self):
        # Personne à suivre (bets tous à 0) -> premier à miser = RAISE,
        # amount = TOTAL misé (convention simulator.py), pas le delta.
        prev = _read([_seat(0, bet=0, to_act=True), _seat(1, bet=0)])
        curr = _read([_seat(0, bet=10), _seat(1, bet=0)])
        actions = infer_actions(prev, curr, Street.FLOP)
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.RAISE
        assert actions[0].amount == 10.0

    def test_exact_call_amount_is_delta(self):
        prev = _read([_seat(0, bet=10), _seat(1, bet=0, to_act=True)])
        curr = _read([_seat(0, bet=10), _seat(1, bet=10)])
        actions = infer_actions(prev, curr, Street.FLOP)
        assert len(actions) == 1
        assert actions[0].seat_index == 1
        assert actions[0].action_type == ActionType.CALL
        assert actions[0].amount == 10.0   # delta = to_call, pas le total (ici identiques)

    def test_call_amount_is_delta_not_total_when_already_partially_in(self):
        # siège 1 avait déjà 2 (blinde BB), doit encore 8 pour suivre un raise à 10
        prev = _read([_seat(0, bet=10), _seat(1, bet=2, to_act=True)])
        curr = _read([_seat(0, bet=10), _seat(1, bet=10)])
        actions = infer_actions(prev, curr, Street.PREFLOP)
        assert actions[0].action_type == ActionType.CALL
        assert actions[0].amount == 8.0   # delta, PAS 10 (le total)

    def test_reraise_tagged_raise_with_total_amount(self):
        prev = _read([_seat(0, bet=10), _seat(1, bet=0, to_act=True)])
        curr = _read([_seat(0, bet=10), _seat(1, bet=30)])
        actions = infer_actions(prev, curr, Street.FLOP)
        assert actions[0].action_type == ActionType.RAISE
        assert actions[0].amount == 30.0   # total, pas le delta de 30-0

    def test_multiple_actions_in_one_diff(self):
        # Deux joueurs ont agi entre deux frames capturées (polling large)
        prev = _read([_seat(0, bet=0, to_act=True), _seat(1, bet=0), _seat(2, bet=0)])
        curr = _read([_seat(0, bet=10), _seat(1, bet=10), _seat(2, bet=0, active=False)])
        actions = infer_actions(prev, curr, Street.FLOP)
        types = {(a.seat_index, a.action_type) for a in actions}
        assert (0, ActionType.RAISE) in types
        assert (1, ActionType.CALL) in types
        assert (2, ActionType.FOLD) in types
        assert len(actions) == 3

    def test_raise_reraise_call_reconstructed_from_one_diff(self):
        # A ouvre à 10, B relance à 30, C paie 30 — 3 actions manquées
        # d'affilée, reconstruites correctement via le plafond courant
        # trié par mise finale croissante (indépendant de l'ordre des sièges).
        prev = _read([_seat(2, bet=0, to_act=True), _seat(0, bet=0), _seat(1, bet=0)])
        curr = _read([_seat(2, bet=10), _seat(0, bet=30), _seat(1, bet=30)])
        actions = infer_actions(prev, curr, Street.FLOP)
        by_seat = {a.seat_index: a for a in actions}
        assert by_seat[2].action_type == ActionType.RAISE and by_seat[2].amount == 10.0
        assert by_seat[0].action_type == ActionType.RAISE and by_seat[0].amount == 30.0
        assert by_seat[1].action_type == ActionType.CALL and by_seat[1].amount == 30.0

    def test_tied_final_bet_ambiguity_resolved_by_seat_index(self):
        # Cas non reconstructible avec certitude (2 photos seulement) :
        # comportement documenté et déterministe -> le plus petit
        # seat_index est conventionnellement l'ouvreur/relanceur.
        prev = _read([_seat(3, bet=0), _seat(1, bet=0, to_act=True)])
        curr = _read([_seat(3, bet=20), _seat(1, bet=20)])
        actions = infer_actions(prev, curr, Street.FLOP)
        by_seat = {a.seat_index: a for a in actions}
        assert by_seat[1].action_type == ActionType.RAISE   # seat_index le plus petit
        assert by_seat[3].action_type == ActionType.CALL
        # Le pot (somme des montants) reste correct quel que soit le label choisi.
        assert by_seat[1].amount + by_seat[3].amount == 20.0 + 20.0


# =============================================================================
# infer_actions — all-in
# =============================================================================

class TestInferAllin:

    def test_detects_allin_with_total_amount(self):
        prev = _read([_seat(0, bet=10), _seat(1, bet=0, to_act=True)])
        curr = _read([_seat(0, bet=10), _seat(1, bet=500, allin=True)])
        actions = infer_actions(prev, curr, Street.FLOP)
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.ALLIN
        assert actions[0].amount == 500.0

    def test_already_allin_produces_no_new_action(self):
        prev = _read([_seat(0, bet=500, allin=True), _seat(1, bet=10, to_act=True)])
        curr = _read([_seat(0, bet=500, allin=True), _seat(1, bet=10)])
        assert infer_actions(prev, curr, Street.FLOP) == []


# =============================================================================
# infer_actions — check
# =============================================================================

class TestInferCheck:

    def test_detects_check_when_turn_passes_with_nothing_owed(self):
        prev = _read([_seat(0, bet=0, to_act=True), _seat(1, bet=0)])
        curr = _read([_seat(0, bet=0, to_act=False), _seat(1, bet=0, to_act=True)])
        actions = infer_actions(prev, curr, Street.FLOP)
        assert len(actions) == 1
        assert actions[0].seat_index == 0
        assert actions[0].action_type == ActionType.CHECK
        assert actions[0].amount == 0.0

    def test_no_check_inferred_if_still_to_act(self):
        # Le siège est toujours signalé "à agir" -> pas encore d'action jouée.
        prev = _read([_seat(0, bet=0, to_act=True), _seat(1, bet=0)])
        curr = _read([_seat(0, bet=0, to_act=True), _seat(1, bet=0)])
        assert infer_actions(prev, curr, Street.FLOP) == []

    def test_no_check_inferred_without_to_act_signal(self):
        # is_to_act jamais vrai dans prev -> on ne peut pas savoir si un
        # tour est réellement passé, donc pas de check inventé.
        prev = _read([_seat(0, bet=0, to_act=False), _seat(1, bet=0)])
        curr = _read([_seat(0, bet=0, to_act=False), _seat(1, bet=0)])
        assert infer_actions(prev, curr, Street.FLOP) == []


# =============================================================================
# infer_actions — sièges qui disparaissent / non actifs dans prev
# =============================================================================

class TestInferEdgeCases:

    def test_seat_that_left_the_table_ignored(self):
        prev = _read([_seat(0, bet=0), _seat(1, bet=0)])
        curr = _read([_seat(0, bet=0)])   # siège 1 disparu de la lecture
        assert infer_actions(prev, curr, Street.FLOP) == []

    def test_folded_seats_in_prev_never_produce_actions(self):
        prev = _read([_seat(0, bet=2, active=False), _seat(1, bet=2, to_act=True)])
        curr = _read([_seat(0, bet=2, active=False), _seat(1, bet=10)])
        actions = infer_actions(prev, curr, Street.FLOP)
        assert all(a.seat_index != 0 for a in actions)

    def test_no_diff_no_actions(self):
        prev = _read([_seat(0, bet=5), _seat(1, bet=5)])
        curr = _read([_seat(0, bet=5), _seat(1, bet=5)])
        assert infer_actions(prev, curr, Street.TURN) == []

    def test_street_is_propagated_to_all_actions(self):
        prev = _read([_seat(0, bet=0, to_act=True), _seat(1, bet=0)])
        curr = _read([_seat(0, bet=0, active=False)])
        curr = _read([_seat(0, bet=0, active=False), _seat(1, bet=0)])
        actions = infer_actions(prev, curr, Street.RIVER)
        assert all(a.street == Street.RIVER for a in actions)
