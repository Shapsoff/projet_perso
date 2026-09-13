"""
hand_lifecycle.py — Détection de fin de main et pont vers la Player DB
Phase 7 — Bot Poker Académique

core/player_db/hand_recorder.py (phase 6) documente explicitement son
intention d'être "branché ... plus tard, sur un flux de mains
reconstruit depuis l'OCR (phase 7)" — ce module est ce pont.

Détection de fin de main : le signal retenu est le changement des
cartes hole du hero. Les 2 cartes du hero ne changent JAMAIS en cours
de main et changent TOUJOURS d'une main à l'autre (nouvelle donne) —
un signal binaire et sans ambiguïté, contrairement à "le board redevient
vide" (vrai aussi en tout début de PREFLOP, donc inutilisable tel quel)
ou à "un seul joueur actif reste" (raté si la lecture manque exactement
la frame du fold gagnant, ce qui arrive avec un polling non garanti).

showdown est reconstruit progressivement au fil des ingest()
(cf. TableStateBuilder._sync_showdown_reveals) : seuls les joueurs dont
les cartes sont devenues visibles ET qui n'ont pas foldé sont retenus —
ce qui reproduit fidèlement ce qu'un observateur de la table voit
réellement (jamais les cartes d'un joueur qui a foldé, jamais plus que
ce que le client montre volontairement à l'écran).

winners / gains ne sont PAS calculés ici : hand_recorder.record_hand()
n'en a besoin à aucun moment (seuls .actions et .showdown sont lus, cf.
HandResultLike dans core/player_db/hand_recorder.py). Les calculer
correctement demanderait de rejouer la répartition du pot (side pots,
égalités au showdown) — que ni ce module ni state_builder.py ne
modélisent (cf. state_builder.py, section "Ce que ce module NE fait
PAS"). Laissés vides par défaut ; à implémenter séparément (à partir
d'une lecture des stacks post-main) si un besoin réel apparaît un jour
(affichage, logs de bankroll).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.game_state import Action, PlayerStatus, Street

from .state_builder import TableStateBuilder
from .vision_types import TableRead


@dataclass
class HandResultData:
    """
    Implémente HandResultLike (core/player_db/hand_recorder.py) par
    duck-typing : expose .actions et .showdown avec les types attendus.
    Les champs supplémentaires suivent la forme de core.simulator.HandResult
    pour rester familier, sans en dépendre (cf. state_builder.py pour la
    raison de ne jamais importer core.simulator depuis ce package).
    """
    hand_id: int
    actions: List[Action] = field(default_factory=list)
    showdown: Dict[int, List[str]] = field(default_factory=dict)
    board: List[str] = field(default_factory=list)
    final_pot: int = 0
    street_reached: Street = Street.PREFLOP
    winners: List[int] = field(default_factory=list)    # non calculé, cf. docstring module
    gains: Dict[int, int] = field(default_factory=dict)  # non calculé, cf. docstring module


class HandLifecycleTracker:
    """
    À utiliser EN PLUS de TableStateBuilder, jamais à sa place :

        builder = TableStateBuilder()
        tracker = HandLifecycleTracker(builder)

        for table_read in stream_of_reads:
            if builder.current_state is None:
                builder.start_new_hand(table_read)
                continue

            finished = tracker.observe(table_read)
            if finished is not None:
                record_hand(db, finished, finished.board, player_id_map, ...)
                builder.start_new_hand(table_read)
                continue

            builder.ingest(table_read)
    """

    def __init__(self, builder: TableStateBuilder) -> None:
        self.builder = builder
        self._last_hero_cards: Optional[Tuple[str, ...]] = None

    def observe(self, table_read: TableRead) -> Optional[HandResultData]:
        """
        Retourne un HandResultData si `table_read` marque le début d'une
        NOUVELLE main (donc la fin de celle en cours dans le builder) ;
        None sinon. Ne modifie JAMAIS le builder ni son propre état
        interne au-delà du suivi des cartes hero — c'est à l'appelant de
        décider d'enchaîner sur start_new_hand() après un résultat
        non-None (cf. exemple ci-dessus).
        """
        hero_cards = self._hero_cards_of(table_read)
        result: Optional[HandResultData] = None

        if (
            hero_cards is not None
            and self._last_hero_cards is not None
            and hero_cards != self._last_hero_cards
            and self.builder.current_state is not None
        ):
            result = self._finalize()

        if hero_cards is not None:
            self._last_hero_cards = hero_cards

        return result

    def reset(self) -> None:
        """À appeler si le suivi doit repartir de zéro (ex: reconnexion à la table)."""
        self._last_hero_cards = None

    @staticmethod
    def _hero_cards_of(table_read: TableRead) -> Optional[Tuple[str, ...]]:
        hero = table_read.hero()
        if hero is None:
            return None
        visible = [c.as_str() for c in hero.hole_cards if c.is_visible()]
        if len(visible) != 2:
            return None
        return tuple(sorted(visible))  # type: ignore[arg-type]

    def _finalize(self) -> HandResultData:
        state = self.builder.current_state
        assert state is not None

        # Le hero est inclus s'il n'a pas foldé (ses cartes sont toujours
        # connues, dès le début de la main) — sans conséquence côté
        # record_hand(), qui exclut explicitement our_player_id. On garde
        # ce comportement pour rester fidèle à core.simulator.HandResult.
        showdown: Dict[int, List[str]] = {
            p.player_id: [str(c) for c in p.hole_cards]
            for p in state.players
            if len(p.hole_cards) == 2 and p.status != PlayerStatus.FOLDED
        }

        return HandResultData(
            hand_id=state.hand_id,
            actions=list(state.action_history),
            showdown=showdown,
            board=state.board_str(),
            final_pot=state.pot,
            street_reached=state.street,
        )
