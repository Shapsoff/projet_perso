"""
hand_recorder.py — Alimentation de la Player DB depuis une main terminée
Phase 6 — Bot Poker Académique

Rôle : pont entre le simulateur (HandResult, cf. simulator.py) et la
Player DB. Ce module ne fait AUCUNE hypothèse sur EHSBot ou RangeBot — il
consomme des structures génériques (actions, cartes, board) et peut donc
être branché aussi bien sur le self-play (phase 6) que, plus tard, sur un
flux de mains reconstruit depuis l'OCR (phase 7).

Volontairement séparé d'EHSBot : le bot décide, il n'observe pas
rétroactivement la main entière (il n'a pas accès aux cartes des joueurs
qui ont foldé, ni à la HandResult complète — ce serait une fuite
d'information qu'un vrai bot n'a pas non plus). L'enregistrement en DB
est une opération de post-traitement, exactement comme un tracker/HUD de
poker classique qui ingère les hand-histories après coup.

Deux niveaux d'information distincts (cf. schema.py) :

  1. Compteurs globaux (VPIP/PFR/3bet/agression/fold-to-flop-bet) —
     ne nécessitent AUCUNE connaissance des cartes cachées. Alimentés pour
     TOUS les adversaires de la main, qu'ils aient foldé ou non.

  2. Fréquences par bucket d'EHS + combos de showdown — nécessitent de
     connaître les cartes de l'adversaire. Alimentées UNIQUEMENT pour les
     joueurs présents dans hand_result.showdown (donnée réellement connue
     après une main réelle, ni plus ni moins).

Limite assumée (cohérente avec les rapports de phases précédentes) :
    "fold to cbet" est approximé par "fold à une mise sur le flop" (on ne
    retrace pas qui était l'agresseur préflop pour qualifier précisément
    une c-bet). Suffisant pour le Palier 1/2, affinable en phase
    ultérieure si besoin.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Protocol

from core.game_state import Action, ActionType, Street
from core.bots.range_definitions import hand_to_class

from .player_db import PlayerDB, canonical_combo

logger = logging.getLogger(__name__)

_STREET_BOARD_LEN = {
    Street.PREFLOP: 0,
    Street.FLOP:    3,
    Street.TURN:    4,
    Street.RIVER:   5,
}

_POSTFLOP_STREETS = [Street.FLOP, Street.TURN, Street.RIVER]

_AGGRESSIVE_TYPES = (ActionType.RAISE, ActionType.ALLIN)
_VPIP_TYPES        = (ActionType.CALL, ActionType.RAISE, ActionType.ALLIN)


class HandResultLike(Protocol):
    """Interface minimale attendue — duck-typing, pas de dépendance dure à simulator.py."""
    actions:  List[Action]
    showdown: Dict[int, List[str]]


def record_hand(
    db:              PlayerDB,
    hand_result:     HandResultLike,
    board:           List[str],
    player_id_map:   Dict[int, str],
    position_map:    Optional[Dict[int, str]] = None,
    our_player_id:   Optional[int] = None,
    ehs_calculator=None,
    n_sims:          int = 300,
) -> None:
    """
    Met à jour la Player DB avec toutes les informations d'une main
    terminée, pour chaque adversaire présent dans player_id_map.

    Args:
        db             : PlayerDB ouverte.
        hand_result    : objet exposant .actions (list[Action]) et
                         .showdown (dict[int, list[str]]) — typiquement un
                         HandResult du simulateur, mais duck-typé.
        board          : cartes communes finales révélées (ex ['7c','2h','Jd','Ts','As']),
                         tronquées automatiquement par street.
        player_id_map  : {player_id (int, table) → player_id (str, DB)}.
                         Seuls ces joueurs sont enregistrés (permet
                         d'exclure notre propre bot).
        position_map   : {player_id (int) → position str}, optionnel —
                         améliore le prior de population (Palier 0) et les
                         stats position-par-position.
        our_player_id  : id de notre bot, exclu automatiquement même s'il
                         apparaît dans player_id_map par erreur.
        ehs_calculator : instance exposant .calculate_multiway(hand, board,
                         n_opponents, n_sims) -> objet avec attribut .EHS
                         (ex: poker_engine.EHSCalculator, ou un stub de
                         test). Si None, l'étape 2 (fréquences par bucket)
                         est ignorée pour cette main — seuls les compteurs
                         globaux (étape 1) sont mis à jour.
        n_sims         : simulations Monte Carlo pour l'EHS de showdown.

    Note : les mises forcées (SB/BB) sont automatiquement exclues du calcul
    VPIP/PFR/3bet — elles portent ActionType.POST_BLIND (cf. game_state.py
    et simulator.py::_post_blindes), un type distinct de RAISE, donc jamais
    présent dans _VPIP_TYPES/_AGGRESSIVE_TYPES ci-dessous. Aucun paramètre
    ni logique de filtrage supplémentaire n'est nécessaire ici.
    """
    position_map = position_map or {}
    n_opponents  = max(len(player_id_map) - 1, 1)

    for table_pid, db_pid in player_id_map.items():
        if table_pid == our_player_id:
            continue

        db.new_hand_observed(db_pid)
        position = position_map.get(table_pid)

        _record_preflop_stats(db, db_pid, position, hand_result.actions, table_pid)
        _record_postflop_aggression(db, db_pid, hand_result.actions, table_pid)
        _record_fold_to_flop_bet(db, db_pid, hand_result.actions, table_pid)

        shown = hand_result.showdown.get(table_pid)
        if shown and len(shown) == 2:
            _record_showdown(
                db, db_pid, shown, position, board,
                hand_result.actions, table_pid, n_opponents,
                ehs_calculator, n_sims,
            )


# =============================================================================
# Étape 1 — compteurs globaux (aucune carte cachée nécessaire)
# =============================================================================

def _record_preflop_stats(
    db: PlayerDB, db_pid: str, position: Optional[str],
    actions: List[Action], table_pid: int,
) -> None:
    preflop = [a for a in actions if a.street == Street.PREFLOP]
    mine    = [a for a in preflop if a.player_id == table_pid]
    if not mine:
        return

    vpip = any(a.action_type in _VPIP_TYPES for a in mine)
    pfr  = any(a.action_type in _AGGRESSIVE_TYPES for a in mine)

    # 3bet : ce joueur relance alors qu'une relance a déjà eu lieu avant
    # lui sur cette street (nombre de raises précédents dans l'ordre
    # chronologique des actions, tous joueurs confondus). ActionType.
    # POST_BLIND n'étant pas dans _AGGRESSIVE_TYPES, les blindes ne
    # comptent jamais comme un raise précédent ici.
    is_3bet   = False
    faced_opp = False
    n_raises_before = 0
    for a in preflop:
        if a.player_id == table_pid:
            if a.action_type in _AGGRESSIVE_TYPES and n_raises_before >= 1:
                is_3bet = True
            if n_raises_before >= 1:
                faced_opp = True
        elif a.action_type in _AGGRESSIVE_TYPES:
            n_raises_before += 1

    db.record_preflop_action(
        db_pid, position, vpip=vpip, pfr=pfr,
        is_3bet=is_3bet, faced_3bet_opportunity=faced_opp,
    )


def _record_postflop_aggression(
    db: PlayerDB, db_pid: str, actions: List[Action], table_pid: int,
) -> None:
    for a in actions:
        if a.street in _POSTFLOP_STREETS and a.player_id == table_pid:
            if a.action_type in _AGGRESSIVE_TYPES:
                db.record_postflop_aggression(db_pid, aggressive=True)
            elif a.action_type == ActionType.CALL:
                db.record_postflop_aggression(db_pid, aggressive=False)
            # CHECK/FOLD exclus du calcul d'AF (ni agressif ni passif au
            # sens classique VPIP/AF — convention standard des trackers).


def _record_fold_to_flop_bet(
    db: PlayerDB, db_pid: str, actions: List[Action], table_pid: int,
) -> None:
    """
    Approximation documentée : 'fold to cbet' ≈ 'fold face à la première
    mise du flop', sans vérifier que le miseur était l'agresseur préflop.
    """
    flop = [a for a in actions if a.street == Street.FLOP]
    facing_bet = False
    for a in flop:
        if a.player_id == table_pid and facing_bet:
            db.record_fold_to_cbet(db_pid, folded=(a.action_type == ActionType.FOLD))
            return  # une seule décision face à la 1ère mise du flop
        if a.action_type in _AGGRESSIVE_TYPES:
            facing_bet = True


# =============================================================================
# Étape 2 — fréquences par bucket d'EHS + combo de showdown
# =============================================================================

def _record_showdown(
    db: PlayerDB, db_pid: str, hole_cards: List[str], position: Optional[str],
    board: List[str], actions: List[Action], table_pid: int, n_opponents: int,
    ehs_calculator, n_sims: int,
) -> None:
    c1, c2 = hole_cards[0], hole_cards[1]
    combo      = canonical_combo(c1, c2)
    hand_class = hand_to_class(c1, c2)
    db.record_showdown_combo(db_pid, combo, hand_class, position)

    if ehs_calculator is None:
        return

    for street in _POSTFLOP_STREETS:
        street_actions = [a for a in actions if a.street == street]
        facing_bet = False
        for a in street_actions:
            if a.player_id == table_pid:
                board_slice = board[:_STREET_BOARD_LEN[street]]
                if len(board_slice) < 3:
                    continue
                try:
                    result = ehs_calculator.calculate_multiway(
                        [c1, c2], board_slice, n_opponents, n_sims
                    )
                    ehs = result.EHS
                except Exception as e:
                    logger.debug("EHS de showdown indisponible (%s) — main ignorée", e)
                    continue

                if a.action_type == ActionType.FOLD:
                    label = 'fold'
                elif a.action_type in _AGGRESSIVE_TYPES:
                    label = 'raise'
                else:  # CALL ou CHECK
                    label = 'call'
                db.record_ehs_bucket_action(db_pid, street.value, ehs, facing_bet, label)
            if a.action_type in _AGGRESSIVE_TYPES:
                facing_bet = True


# =============================================================================
# Tests intégrés
# =============================================================================

if __name__ == "__main__":
    import tempfile

    print("\n=== Tests hand_recorder.py ===\n")
    passed = failed = 0

    def t(label, cond, detail=""):
        global passed, failed
        if cond:
            print(f"  ✓ {label}")
            passed += 1
        else:
            print(f"  ✗ {label}" + (f" : {detail}" if detail else ""))
            failed += 1

    class _FakeHandResult:
        def __init__(self, actions, showdown):
            self.actions  = actions
            self.showdown = showdown

    class _StubEHS:
        class _R:
            def __init__(self, ehs): self.EHS = ehs
        def calculate_multiway(self, hand, board, n_opp, n_sims):
            return self._R(0.75)

    with tempfile.TemporaryDirectory() as tmp:
        _open_dbs: list = []  # fermées dans le finally, même en cas d'erreur —
        # sur Windows, TemporaryDirectory() ne peut pas nettoyer un dossier
        # contenant un fichier SQLite encore ouvert (PermissionError), contrai-
        # rement à Linux où c'est silencieusement toléré. Une exception en plein
        # test (ex: POST_BLIND absent d'un game_state.py périmé) laissait donc
        # des connexions ouvertes et masquait l'erreur réelle derrière un
        # PermissionError de nettoyage — cf. session 6.
        try:
            db = PlayerDB(f"{tmp}/db.sqlite3")
            _open_dbs.append(db)

            # Main simulée : joueur 1 (nous, id=0) vs joueur adverse id=1.
            # Preflop : villain (1) raise, nous callons.
            # Flop : villain bet, nous callons. Showdown : villain montre AsKs.
            actions = [
                Action(player_id=1, action_type=ActionType.RAISE, amount=6, street=Street.PREFLOP),
                Action(player_id=0, action_type=ActionType.CALL,  amount=6, street=Street.PREFLOP),
                Action(player_id=1, action_type=ActionType.RAISE, amount=10, street=Street.FLOP),
                Action(player_id=0, action_type=ActionType.CALL,  amount=10, street=Street.FLOP),
            ]
            result = _FakeHandResult(
                actions=actions,
                showdown={1: ["As", "Ks"], 0: ["7c", "2d"]},
            )
            board = ["Kh", "5d", "2s", "9c", "3h"]

            record_hand(
                db=db, hand_result=result, board=board,
                player_id_map={0: "us", 1: "villain_1"},
                position_map={1: "CO"},
                our_player_id=0,
                ehs_calculator=_StubEHS(), n_sims=50,
            )

            row = db.get_player_row("villain_1")
            t("hands_seen incrémenté", row.hands_seen == 1, f"obtenu {row.hands_seen}")
            t("VPIP=True (a raise préflop)", row.hands_vpip == 1)
            t("PFR=True (a raise préflop)", row.hands_pfr == 1)
            t("aggressive_acts incrémenté (raise flop)", row.aggressive_acts == 1)

            showdowns = db.get_showdown_combos("villain_1")
            t("showdown enregistré", len(showdowns) == 1, f"obtenu {showdowns}")
            if showdowns:
                combo, hand_class, pos = showdowns[0]
                t("hand_class = AKs", hand_class == "AKs", f"obtenu {hand_class}")
                t("position enregistrée = CO", pos == "CO", f"obtenu {pos}")

            freq_table = db.get_action_freq_table("villain_1")
            t("fréquence flop enregistrée (raise, EHS stub=0.75)",
              len(freq_table) >= 1, f"obtenu {list(freq_table.keys())}")

            # "nous" ne devons jamais être enregistrés dans la DB
            us_row = db.get_player_row("us")
            t("notre propre bot n'est jamais enregistré", us_row is None)

            # ── Reproduction du bug trouvé en conditions réelles (session 6) ─
            # simulator.py::_post_blindes() loggait les blindes en ActionType.
            # RAISE, les rendant indiscernables d'une vraie relance volontaire.
            # Corrigé à la source (ActionType.POST_BLIND dédié) — vérifié ici
            # que hand_recorder en bénéficie automatiquement, sans logique de
            # filtrage spécifique.
            actions_blind_fold = [
                Action(player_id=1, action_type=ActionType.POST_BLIND, amount=5,  street=Street.PREFLOP),  # SB poste (forcé)
                Action(player_id=0, action_type=ActionType.POST_BLIND, amount=10, street=Street.PREFLOP),  # BB poste (forcé)
                Action(player_id=1, action_type=ActionType.FOLD,       amount=0,  street=Street.PREFLOP),  # SB fold (vrai choix)
            ]
            result_blind_fold = _FakeHandResult(actions=actions_blind_fold, showdown={})
            db_blind = PlayerDB(f"{tmp}/db_blind.sqlite3")
            _open_dbs.append(db_blind)
            record_hand(db=db_blind, hand_result=result_blind_fold, board=[],
                        player_id_map={1: "villain_blind"}, our_player_id=0)
            row_blind = db_blind.get_player_row("villain_blind")
            t("POST_BLIND puis fold -> VPIP=False (pas de filtre spécifique nécessaire)",
              row_blind.hands_vpip == 0, f"obtenu hands_vpip={row_blind.hands_vpip}")
            t("hands_seen quand même incrémenté", row_blind.hands_seen == 1,
              f"obtenu {row_blind.hands_seen}")

            # Un vrai raise APRÈS le poste de blind doit rester correctement compté
            actions_blind_then_raise = [
                Action(player_id=1, action_type=ActionType.POST_BLIND, amount=5,  street=Street.PREFLOP),  # SB poste
                Action(player_id=0, action_type=ActionType.POST_BLIND, amount=10, street=Street.PREFLOP),  # BB poste
                Action(player_id=1, action_type=ActionType.RAISE,      amount=25, street=Street.PREFLOP),  # SB relance vraiment
            ]
            result_real_raise = _FakeHandResult(actions=actions_blind_then_raise, showdown={})
            db_real = PlayerDB(f"{tmp}/db_real.sqlite3")
            _open_dbs.append(db_real)
            record_hand(db=db_real, hand_result=result_real_raise, board=[],
                        player_id_map={1: "villain_real_raise"}, our_player_id=0)
            row_real = db_real.get_player_row("villain_real_raise")
            t("Vraie relance après le poste de blind toujours comptée (VPIP=True)",
              row_real.hands_vpip == 1 and row_real.hands_pfr == 1,
              f"obtenu vpip={row_real.hands_vpip} pfr={row_real.hands_pfr}")

            # Un raise qui suit 2 POST_BLIND ne doit JAMAIS être compté comme 3bet
            # (n_raises_before doit rester à 0 après les blindes)
            t("3bet=False après un simple open suivant les blindes (pas de faux 3bet)",
              row_real.hands_3bet == 0, f"obtenu hands_3bet={row_real.hands_3bet}")
        finally:
            for _db in _open_dbs:
                try:
                    _db.close()
                except Exception:
                    pass

    print(f"\n  {passed}/{passed+failed} tests passés")
    print("  ✅ OK\n" if failed == 0 else f"  ⚠ {failed} échec(s)\n")
