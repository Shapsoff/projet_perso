"""
state_builder.py — Construction incrémentale d'un GameState depuis la table
Phase 7 — Bot Poker Académique

Orchestrateur central du package ocr/ : reçoit des TableRead validés et
maintient un core.game_state.GameState à jour, exactement dans le
format que core/simulator.py::PokerTable produit pour le self-play.
C'est la seule pièce du pipeline OCR qui connaît à la fois vision_types
(entrée) et game_state (sortie).

Reproduit volontairement la bookkeeping de core/simulator.py
(PokerTable._apply_action, _post_blindes) plutôt que de la
réinventer : mêmes conventions de montant (CALL = delta, RAISE/ALLIN =
total misé sur la rue), même formule de min_raise, même façon de ne
faire avancer aggressor_id que sur RAISE/ALLIN. Voir action_inference.py
pour le détail de cette convention.

Ce que ce module NE fait PAS :
  - Pas de calcul de side pots (GameState.side_pots reste vide). Le
    détail des side pots ne sert qu'au partage des gains en fin de main
    (répartition entre plusieurs all-in), jamais à une décision en cours
    de main — seul `pot` (montant total, toujours correct) est consommé
    par best_response_engine.py / ev_calculator.py.
  - Pas de lecture de `to_call`/`min_raise` sur l'écran : ces valeurs
    sont recalculées ici à partir des mises observées, plus robuste
    qu'une dépendance à un libellé de bouton ("Call 120") qui varie
    d'un client à l'autre.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Dict, List, Optional

from core.deck import Card
from core.game_state import (
    Action, ActionType, GameState, Player, PlayerStatus, Position, Street,
)

from .action_inference import InferredAction, infer_actions, post_blinds
from .seating import positions_from_button
from .validation import ValidationResult, validate_table_read
from .vision_types import SeatRead, TableRead

logger = logging.getLogger(__name__)

_STREET_BY_BOARD_LEN: Dict[int, Street] = {
    0: Street.PREFLOP,
    3: Street.FLOP,
    4: Street.TURN,
    5: Street.RIVER,
}


class StateBuilderError(Exception):
    """Levée quand une main ne peut pas être démarrée à partir de la lecture fournie."""


class TableStateBuilder:
    """
    Usage :
        builder = TableStateBuilder(session_id=1)
        state = builder.start_new_hand(first_read_of_hand)
        ...
        state = builder.ingest(next_read)   # None si la lecture est rejetée
        ...
        decision = ehs_bot.decide(builder.current_state)
    """

    def __init__(
        self,
        session_id: int = 0,
        min_confidence: float = 0.6,
        check_confidence: bool = True,
    ) -> None:
        self.session_id = session_id
        self.min_confidence = min_confidence
        self.check_confidence = check_confidence

        self._state: Optional[GameState] = None
        self._last_read: Optional[TableRead] = None
        self._hand_counter = 0

    @property
    def current_state(self) -> Optional[GameState]:
        return self._state

    # =========================================================================
    # Début de main
    # =========================================================================

    def start_new_hand(
        self,
        table_read: TableRead,
        hand_id: Optional[int] = None,
    ) -> GameState:
        """
        Initialise un nouveau GameState à partir de la première lecture
        d'une main (board vide, cartes hero visibles, blindes lisibles).

        Raises:
            StateBuilderError si les champs indispensables au démarrage
            d'une main ne sont pas lisibles sur cette frame. Le GameState
            précédent (le cas échéant) est conservé intact dans
            self.current_state — l'appelant peut retenter sur la frame
            suivante.
        """
        problems = self._missing_start_fields(table_read)
        if problems:
            raise StateBuilderError(
                "Impossible de démarrer une nouvelle main : " + "; ".join(problems)
            )

        validation = validate_table_read(
            table_read, self.min_confidence, self.check_confidence
        )
        if not validation.ok:
            raise StateBuilderError(
                "Lecture de départ de main invalide : " + "; ".join(validation.errors)
            )

        active_seats = table_read.active_seats()
        seat_indices = [s.seat_index for s in active_seats]
        dealer_seat = table_read.dealer_seat_index()
        assert dealer_seat is not None  # garanti par _missing_start_fields
        positions = positions_from_button(seat_indices, dealer_seat)

        players: List[Player] = []
        for seat in active_seats:
            # seat.stack correspond exactement à la sémantique attendue par
            # Player.stack : les jetons NON engagés sur la mise en cours,
            # tels qu'affichés par le client (la blinde déjà postée est
            # déjà déduite de ce nombre à l'écran) — donc lu tel quel, sans
            # retraitement. Cf. _apply_inferred_action, cas POST_BLIND.
            players.append(Player(
                player_id=seat.seat_index,
                name=seat.pseudo or f"seat{seat.seat_index}",
                stack=self._to_int_amount(seat.stack or 0.0),
                position=positions[seat.seat_index],
                status=PlayerStatus.ACTIVE,
                hole_cards=[Card.from_str(c.as_str()) for c in seat.hole_cards if c.is_visible()],
                bet_this_street=0,
            ))

        self._hand_counter += 1
        state = GameState(
            hand_id=hand_id if hand_id is not None else self._hand_counter,
            session_id=self.session_id,
            players=players,
            our_id=table_read.hero_seat_index,
            board=[],
            street=Street.PREFLOP,
            pot=0,
            main_pot=0,
            side_pots=[],
            to_call=0,
            min_raise=self._to_int_amount(table_read.big_blind or 0.0),
            aggressor_id=None,
            small_blind=self._to_int_amount(table_read.small_blind or 0.0),
            big_blind=self._to_int_amount(table_read.big_blind or 0.0),
            action_history=[],
        )
        self._state = state

        sb_seat, bb_seat = self._blind_seats(positions, active_seats)
        for ia in post_blinds(table_read, sb_seat, bb_seat):
            self._apply_inferred_action(ia)
        # cf. core/simulator.py::_post_blindes — le min-raise initial est
        # 2x la grosse blinde, fixé une fois pour toutes après les 2
        # POST_BLIND (pas via la formule générique de raise, qui ne
        # s'applique qu'à une vraie décision d'agression).
        state.min_raise = 2 * state.big_blind

        self._last_read = table_read
        return state

    @staticmethod
    def _blind_seats(
        positions: Dict[int, Position], active_seats: List[SeatRead],
    ) -> tuple[int, int]:
        by_pos = {pos: seat for seat, pos in positions.items()}
        if Position.SB in by_pos:
            return by_pos[Position.SB], by_pos[Position.BB]
        # Heads-up : BTN poste la petite blinde (cf. core/simulator.py::_post_blindes).
        return by_pos[Position.BTN], by_pos[Position.BB]

    @staticmethod
    def _missing_start_fields(table_read: TableRead) -> List[str]:
        problems: List[str] = []
        if table_read.board_cards:
            problems.append("le board n'est pas vide (ce n'est pas un tout début de main)")
        if table_read.hero_seat_index is None:
            problems.append("hero_seat_index manquant")
        else:
            hero = table_read.seat(table_read.hero_seat_index)
            if hero is None or not hero.is_occupied:
                problems.append("le siège hero n'est pas occupé")
            elif len([c for c in hero.hole_cards if c.is_visible()]) != 2:
                problems.append("les 2 cartes hero ne sont pas visibles")
        if not table_read.small_blind or not table_read.big_blind:
            problems.append("blindes non lisibles")
        if table_read.dealer_seat_index() is None:
            problems.append("bouton (dealer) non identifié")
        n_active = len(table_read.active_seats())
        if n_active < 2 or n_active > 6:
            problems.append(f"nombre de joueurs actifs invalide ({n_active})")
        return problems

    # =========================================================================
    # Frames suivantes
    # =========================================================================

    def ingest(self, table_read: TableRead) -> Optional[GameState]:
        """
        Consomme une nouvelle lecture de table.

        Returns:
            Le GameState mis à jour, ou None si la lecture est rejetée
            (validation échouée) ou si aucune main n'est en cours
            (start_new_hand() doit être appelé d'abord). Dans le cas
            rejeté, self.current_state n'est PAS modifié.
        """
        if self._state is None:
            logger.warning("ingest() appelé sans main en cours — ignoré")
            return None

        validation = validate_table_read(
            table_read, self.min_confidence, self.check_confidence
        )
        if not validation.ok:
            logger.info("Frame rejetée : %s", "; ".join(validation.errors))
            return self._state

        new_street = _STREET_BY_BOARD_LEN.get(len(table_read.board_cards))
        if new_street is None:
            logger.warning(
                "Longueur de board inattendue (%d), frame ignorée",
                len(table_read.board_cards),
            )
            return self._state

        if new_street != self._state.street:
            self._advance_street(new_street, table_read)
        else:
            assert self._last_read is not None
            for ia in infer_actions(self._last_read, table_read, self._state.street):
                self._apply_inferred_action(ia)
            self._last_read = table_read

        self._sync_showdown_reveals(table_read)
        return self._state

    def _sync_showdown_reveals(self, table_read: TableRead) -> None:
        """
        Capture les cartes adverses qui deviennent visibles (showdown).
        Ne touche jamais les cartes du hero (déjà fixées au début de la
        main) ni un joueur dont les cartes sont déjà connues (évite
        qu'une lecture bruitée plus tardive n'écrase une lecture correcte).
        """
        state = self._state
        assert state is not None
        for player in state.players:
            if player.player_id == state.our_id or player.hole_cards:
                continue
            seat = table_read.seat(player.player_id)
            if seat is None:
                continue
            visible = [c for c in seat.hole_cards if c.is_visible()]
            if len(visible) == 2:
                player.hole_cards = [Card.from_str(c.as_str()) for c in visible]

    def _advance_street(self, new_street: Street, table_read: TableRead) -> None:
        state = self._state
        assert state is not None

        for p in state.players:
            p.bet_this_street = 0
        state.street = new_street
        state.to_call = 0
        state.min_raise = state.big_blind
        state.aggressor_id = None
        state.board = [Card.from_str(c.as_str()) for c in table_read.board_cards if c.is_visible()]

        # Cas limite : le polling a manqué la toute première frame de la
        # nouvelle rue (personne n'a encore agi) et cette frame montre déjà
        # une action effectuée sur la nouvelle rue. On rattrape en diffant
        # contre une base à mises nulles reconstruite pour l'occasion.
        # is_to_act=False dans cette base synthétique : on ne peut pas savoir
        # qui était "à agir" avant un point de départ qu'on n'a jamais
        # observé, donc on n'invente aucun check ici (seuls bet/raise/fold/
        # allin sont détectables sans cette information).
        zero_baseline = replace(
            table_read,
            seats=[replace(s, bet_this_street=0.0, is_to_act=False) for s in table_read.seats],
        )
        for ia in infer_actions(zero_baseline, table_read, new_street):
            self._apply_inferred_action(ia)

        self._last_read = table_read

    # =========================================================================
    # Application d'une action inférée
    # =========================================================================

    def _apply_inferred_action(self, ia: InferredAction) -> None:
        state = self._state
        assert state is not None
        player = state.get_player(ia.seat_index)
        amount = self._to_int_amount(ia.amount)
        action = Action(player_id=ia.seat_index, action_type=ia.action_type,
                         amount=amount, street=ia.street)

        if ia.action_type == ActionType.FOLD:
            player.status = PlayerStatus.FOLDED

        elif ia.action_type == ActionType.CHECK:
            pass

        elif ia.action_type == ActionType.CALL:
            call_amount = min(amount, player.stack)
            player.stack -= call_amount
            player.bet_this_street += call_amount
            state.pot += call_amount
            if player.stack <= 0:
                player.status = PlayerStatus.ALLIN
            action.amount = call_amount

        elif ia.action_type == ActionType.POST_BLIND:
            # Cas particulier : contrairement à toutes les autres actions
            # (dont le montant est dérivé d'un delta et appliqué au stack
            # suivi en interne), le tout premier stack d'un joueur
            # (Player.stack, initialisé depuis seat.stack dans
            # start_new_hand) reflète DÉJÀ la blinde postée — c'est ce que
            # montre le client à l'écran dès la 1ère frame de la main. La
            # soustraire une seconde fois ici doublerait le décompte.
            player.bet_this_street += amount
            if player.stack <= 0:
                player.status = PlayerStatus.ALLIN
            state.pot += amount
            action.amount = amount

        elif ia.action_type in (ActionType.RAISE, ActionType.ALLIN):
            max_bet_before = max((p.bet_this_street for p in state.active_players), default=0)
            total_bet = amount   # convention : montant TOTAL misé sur la rue
            additional = max(0, total_bet - player.bet_this_street)
            additional = min(additional, player.stack)

            player.stack -= additional
            player.bet_this_street += additional
            state.pot += additional

            if player.stack <= 0:
                player.status = PlayerStatus.ALLIN
                action.action_type = ActionType.ALLIN
            else:
                action.action_type = ActionType.RAISE

            action.amount = player.bet_this_street
            raise_size = player.bet_this_street - max_bet_before
            state.min_raise = player.bet_this_street + max(raise_size, state.big_blind)
            state.aggressor_id = player.player_id

        state.action_history.append(action)
        self._recompute_to_call()

    def _recompute_to_call(self) -> None:
        state = self._state
        assert state is not None
        hero = state.our_player
        max_bet_now = max((p.bet_this_street for p in state.active_players), default=0)
        state.to_call = max(0, max_bet_now - hero.bet_this_street)

    @staticmethod
    def _to_int_amount(x: float) -> int:
        return int(round(x))
