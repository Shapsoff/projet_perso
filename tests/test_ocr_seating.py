"""
test_ocr_seating.py — Tests unitaires de ocr/seating.py
Phase 7 — Bot Poker Académique

Placement : tests/
Commande   : depuis la racine du projet
    pytest tests/test_ocr_seating.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from core.game_state import Position
from ocr.seating import POSITIONS_BY_N, positions_from_button


class TestPositionsFromButton:

    def test_heads_up(self):
        pos = positions_from_button([0, 1], dealer_seat_index=0)
        assert pos == {0: Position.BTN, 1: Position.BB}

    def test_heads_up_dealer_on_other_seat(self):
        pos = positions_from_button([0, 1], dealer_seat_index=1)
        assert pos == {1: Position.BTN, 0: Position.BB}

    def test_six_max_standard(self):
        pos = positions_from_button([0, 1, 2, 3, 4, 5], dealer_seat_index=0)
        assert pos[0] == Position.BTN
        assert pos[1] == Position.SB
        assert pos[2] == Position.BB
        assert pos[3] == Position.UTG
        assert pos[4] == Position.MP
        assert pos[5] == Position.CO

    def test_six_max_button_not_on_seat_zero(self):
        pos = positions_from_button([0, 1, 2, 3, 4, 5], dealer_seat_index=3)
        assert pos[3] == Position.BTN
        assert pos[4] == Position.SB
        assert pos[5] == Position.BB
        assert pos[0] == Position.UTG
        assert pos[1] == Position.MP
        assert pos[2] == Position.CO

    def test_non_contiguous_seats(self):
        # Table 6 places physiques, seuls 3 sièges occupés et non contigus
        # (courant en cashgame : joueurs assis en 0, 2, 4 sur 6 sièges).
        pos = positions_from_button([0, 2, 4], dealer_seat_index=2)
        assert pos == {2: Position.BTN, 4: Position.SB, 0: Position.BB}

    def test_all_supported_table_sizes(self):
        for n in range(2, 7):
            seats = list(range(n))
            pos = positions_from_button(seats, dealer_seat_index=0)
            assert len(pos) == n
            assert pos[0] == Position.BTN
            assert set(pos.values()) == set(POSITIONS_BY_N[n])

    def test_rejects_too_few_players(self):
        with pytest.raises(ValueError):
            positions_from_button([0], dealer_seat_index=0)

    def test_rejects_too_many_players(self):
        with pytest.raises(ValueError):
            positions_from_button(list(range(7)), dealer_seat_index=0)

    def test_rejects_dealer_not_in_occupied_seats(self):
        with pytest.raises(ValueError):
            positions_from_button([0, 1, 2], dealer_seat_index=5)

    def test_every_occupied_seat_gets_a_position(self):
        seats = [1, 3, 5]
        pos = positions_from_button(seats, dealer_seat_index=3)
        assert set(pos.keys()) == set(seats)


class TestSyncWithSimulator:
    """
    Garde de synchronisation : POSITIONS_BY_N est dupliqué depuis
    core/simulator.py::PokerTable.POSITIONS_BY_N (cf. docstring de
    seating.py) pour ne pas dépendre de poker_engine (moteur C++
    compilé). Ce test détecte une divergence si l'une des deux tables
    est modifiée sans l'autre — mais est ignoré (skip) si le module
    compilé n'est pas disponible dans l'environnement d'exécution.
    """

    def test_positions_by_n_matches_simulator(self):
        try:
            from core.simulator import PokerTable
        except RuntimeError:
            pytest.skip("poker_engine (moteur C++) non disponible dans cet environnement")

        assert POSITIONS_BY_N == PokerTable.POSITIONS_BY_N, (
            "ocr/seating.py::POSITIONS_BY_N a divergé de "
            "core/simulator.py::PokerTable.POSITIONS_BY_N — répercuter le "
            "changement dans les deux fichiers."
        )
