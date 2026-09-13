"""
action_inference.py — Inférence d'actions par diff de deux lectures de table
Phase 7 — Bot Poker Académique

Un client de poker n'expose jamais directement "le joueur X a fait
raise" : seulement de nouveaux pixels. Ce module retrouve l'action
jouée en comparant deux TableRead successifs déjà validés/stabilisés —
comme un joueur humain déduit "il a callé" en comparant le stack
adverse avant/après, sans jamais voir de bouton pressé.

Convention volontairement alignée sur core/game_state.py : ActionType
n'a PAS de valeur BET distincte de RAISE (cf. core/action_history.py,
qui traite les tags 'bet' et 'raise' de façon strictement identique
dans son classificateur). Toute mise d'agression — qu'elle ouvre les
enchères ou relance une mise existante — est donc taguée
ActionType.RAISE ici, à l'identique de ce que produirait le simulateur.

Convention Action.amount — asymétrique, à respecter exactement (cf.
core/simulator.py::PokerTable._apply_action) :
  - CALL  : montant INCRÉMENTAL ajouté au pot (= to_call au moment de
            l'action, éventuellement plafonné par le stack).
  - RAISE / ALLIN : montant TOTAL misé sur la rue après l'action (PAS
            l'incrément du raise). Le commentaire du simulateur est
            explicite : "le bot envoie le montant TOTAL de sa mise de
            rue, pas juste l'incrément du raise".
  - POST_BLIND : montant de la blinde (incrément = total, premier bet
            de la main).
Une confusion ici romprait silencieusement la compatibilité avec
action_history.py / hand_recorder.py, qui ont tous deux été calibrés
sur cette convention exacte.

Les blindes (POST_BLIND) ne sont PAS déduites par diff : elles sont
postées explicitement par post_blinds(), appelé une fois en tout début
de main. Les injecter via la logique générique de diff les exposerait
au même bug que celui documenté en session 6 dans action_history.py
(une blinde taguée RAISE par erreur fausse tout le classificateur de
bucket adverse en aval).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from core.game_state import ActionType, Street

from .vision_types import TableRead

# Tolérance numérique pour comparer des montants lus (float) à un call exact.
_AMOUNT_EPSILON = 1e-6


@dataclass(frozen=True)
class InferredAction:
    seat_index: int
    action_type: ActionType
    amount: float
    street: Street


def post_blinds(
    table_read: TableRead,
    sb_seat_index: int,
    bb_seat_index: int,
) -> List[InferredAction]:
    """
    Construit explicitement les actions POST_BLIND de début de main à
    partir des montants de blindes lus sur la table. Ne fait aucune
    hypothèse sur bet_this_street : les montants viennent directement de
    TableRead.small_blind / big_blind, pas d'un diff.
    """
    actions: List[InferredAction] = []
    if table_read.small_blind:
        actions.append(InferredAction(
            sb_seat_index, ActionType.POST_BLIND, table_read.small_blind, Street.PREFLOP,
        ))
    if table_read.big_blind:
        actions.append(InferredAction(
            bb_seat_index, ActionType.POST_BLIND, table_read.big_blind, Street.PREFLOP,
        ))
    return actions


def infer_actions(
    prev: TableRead,
    curr: TableRead,
    street: Street,
) -> List[InferredAction]:
    """
    Compare deux lectures successives d'une MÊME street et retourne les
    actions détectées depuis `prev`. Peut retourner 0, 1 ou plusieurs
    actions si plusieurs joueurs ont agi entre deux frames capturées
    (polling trop espacé, ou plusieurs all-in successifs sans que
    l'appelant n'ait eu le temps de capturer les états intermédiaires).

    N'émet JAMAIS d'action inventée dans le doute : un GameState
    incomplet (action manquante) est préférable à une action fausse qui
    fausserait silencieusement action_history / Player DB en aval.

    Désambiguïsation raise/call sur un diff à plusieurs actions : le
    poker est strictement séquentiel, donc quand plusieurs sièges
    augmentent leur mise entre deux frames, on les trie par mise finale
    croissante et on avance un "plafond courant" (parti de la mise max
    de `prev`) — quiconque dépasse ce plafond est un relanceur (le
    plafond monte alors à sa mise), quiconque l'égale est un call. Ça
    reconstruit correctement une séquence raise → re-raise → call même
    répartie sur plusieurs frames manquées. La seule ambiguïté réelle
    est le cas de deux sièges qui finissent à EXACTEMENT la même mise
    (un relanceur et son unique caller) : lequel des deux a agi en
    premier n'est pas reconstructible à partir de deux photos seules —
    on tranche alors par seat_index croissant, arbitrairement mais de
    façon déterministe. Ça n'affecte jamais le pot ni les stacks
    (calculés à partir des montants observés, pas du label choisi),
    seulement l'étiquette RAISE/CALL de l'action la plus ancienne du
    lot. Ce cas ne se présente que si au moins 2 actions ont été
    manquées d'affilée ; avec un polling suffisant, chaque diff ne
    contient qu'une action et la question ne se pose jamais.
    """
    actions: List[InferredAction] = []
    max_bet_before = max((s.bet_this_street for s in prev.active_seats()), default=0.0)
    aggressive: List[tuple] = []  # (prev_seat, curr_seat)

    for prev_seat in prev.active_seats():
        curr_seat = curr.seat(prev_seat.seat_index)
        if curr_seat is None or not curr_seat.is_occupied:
            # Siège vidé entre les deux frames : pas une action de jeu.
            # Géré par player_identity / hand_lifecycle, pas ici.
            continue

        # ── Fold ─────────────────────────────────────────────────────────
        if not curr_seat.is_active:
            actions.append(InferredAction(prev_seat.seat_index, ActionType.FOLD, 0.0, street))
            continue

        delta = curr_seat.bet_this_street - prev_seat.bet_this_street

        # ── All-in ───────────────────────────────────────────────────────
        # amount = TOTAL misé sur la rue (convention simulator.py), pas le
        # delta — cf. note de convention dans la docstring du module.
        if curr_seat.is_allin and not prev_seat.is_allin:
            actions.append(InferredAction(
                prev_seat.seat_index, ActionType.ALLIN, curr_seat.bet_this_street, street,
            ))
            continue

        # ── Bet / Call / Raise : différé en phase 2 (cf. docstring) ────────
        if delta > _AMOUNT_EPSILON:
            aggressive.append((prev_seat, curr_seat))
            continue

        # ── Check ────────────────────────────────────────────────────────
        # Rien n'a changé pour ce siège, mais son tour vient de passer
        # (il était signalé "à agir" avant, ne l'est plus dans `curr`) et
        # rien ne lui était dû. Sans is_to_act, on ne pourrait pas
        # distinguer "il vient de checker" de "ce n'est simplement pas
        # encore son tour" — dans le doute on n'émet rien plutôt que
        # d'inventer un check.
        to_call_now = max_bet_before - prev_seat.bet_this_street
        if prev_seat.is_to_act and not curr_seat.is_to_act and to_call_now <= _AMOUNT_EPSILON:
            actions.append(InferredAction(prev_seat.seat_index, ActionType.CHECK, 0.0, street))

    # ── Phase 2 : raise/call parmi les sièges dont la mise a augmenté ──────
    running_cap = max_bet_before
    for prev_seat, curr_seat in sorted(
        aggressive, key=lambda pair: (pair[1].bet_this_street, pair[0].seat_index)
    ):
        if curr_seat.bet_this_street > running_cap + _AMOUNT_EPSILON:
            actions.append(InferredAction(
                prev_seat.seat_index, ActionType.RAISE, curr_seat.bet_this_street, street,
            ))
            running_cap = curr_seat.bet_this_street
        else:
            delta = curr_seat.bet_this_street - prev_seat.bet_this_street
            actions.append(InferredAction(prev_seat.seat_index, ActionType.CALL, delta, street))

    return actions
