"""
simulator.py — Simulateur de table poker Texas Hold'em
Version 4 - 12/06/2026

Gère :
  - Distribution des cartes
  - Rotation des positions (BTN, blindes)
  - Gestion du pot et des mises par rue
  - Side pots pour les all-ins
  - Showdown et attribution des gains
  - Interface avec le hand evaluator C++ (ou fallback Python)
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Callable, Any
from .deck import Deck, Card
from .game_state import (
    GameState, Player, Action, Street, Position, ActionType, PlayerStatus
)

# Import du moteur C++ — 
try:
    import poker_engine as _cpp
    _has_cpp = True
except ImportError:
    _has_cpp = False
    raise RuntimeError("[simulator] ⚠️  poker_engine C++ non trouvé — requis")


def evaluate_hand(cards: list[str]) -> int | tuple[int, ...]:
    """Évalue une main de 7 cartes. Utilise C++ si disponible."""
    return _cpp.evaluate_hand(cards) # type: ignore


def compare_hands(hand_a: list[str], hand_b: list[str]) -> int:
    """1 = A gagne, -1 = B gagne, 0 = égalité."""
    return _cpp.compare_hands(hand_a, hand_b) # type: ignore


# ── Bot interface ─────────────────────────────────────────────────────────────
# Un bot est simplement une fonction : GameState → Action

BotFn = Callable[[GameState], Action]


# ── Résultat d'une main ────────────────────────────────────────────────────────

@dataclass
class HandResult:
    hand_id:    int
    winners:    list[int]          # player_ids des gagnants
    gains:      dict[int, int]     # player_id → gain net (peut être négatif)
    showdown:   dict[int, list[str]] = field(default_factory=dict)  # type: ignore  # cartes montrées
    final_pot:  int = 0
    actions:    list[Action] = field(default_factory=list) # type: ignore
    street_reached: Street = Street.PREFLOP # ❌ à voir clairement dans le game_state voir ce que ca fait


# ── Simulateur ────────────────────────────────────────────────────────────────

class PokerTable:
    """
    Simule une table de poker Texas Hold'em multi-joueurs.

    Usage :
        bots = {0: our_bot, 1: random_bot, 2: rule_bot}
        table = PokerTable(n_players=3, starting_stacks=1000, big_blind=10)
        result = table.run_hand(bots)
    """
    # ❌ table à 7 joueur, pour les tournois sur CoinPoker
    # Si on reste sur cashgame, 6 joueur est suffisant
    POSITIONS_BY_N = {
        2: [Position.BTN, Position.BB],
        3: [Position.BTN, Position.SB, Position.BB],
        4: [Position.BTN,  Position.SB, Position.BB, Position.CO],
        5: [Position.BTN,  Position.SB,  Position.BB, Position.MP, Position.CO],
        6: [Position.BTN, Position.SB,  Position.BB,  Position.UTG, Position.MP, Position.CO],
    }

    def __init__(
        self,
        n_players:       int,
        starting_stacks: int  = 1000,
        small_blind:     int  = 5,
        big_blind:       int  = 10, # Donc 100BB
        our_player_id:   int  = 0,
        seed:            int | None = None,
    ) -> None:
        if n_players < 2 or n_players > 6:
            raise ValueError("2 à 6 joueurs supportés")

        self.n_players       = n_players
        self.small_blind     = small_blind
        self.big_blind       = big_blind
        self.our_player_id   = our_player_id
        self.rng             = random.Random(seed)
        self.hand_counter    = 0
        self.btn_seat        = 0  # siège du bouton, tourne à chaque main

        # Stacks persistants entre les mains
        self.stacks: dict[int, int] = {i: starting_stacks for i in range(n_players)}

    # ── Main publique ─────────────────────────────────────────────────────────

    def run_hand(self, bots: dict[int, BotFn]) -> HandResult:
        """Joue une main complète. Retourne le HandResult."""
        self.hand_counter += 1
        deck = Deck()
        deck.shuffle(seed=self.rng.randint(0, 2**32))

        # 1. Créer les joueurs avec positions
        players = self._create_players()

        # 2. Initialiser le game state
        state = GameState(
            hand_id=self.hand_counter,
            players=players,
            our_id=self.our_player_id,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
        )

        # 3. Poster les blindes
        pot = self._post_blindes(state)

        # 4. Distribuer les cartes
        self._deal_hole_cards(state, deck)

        # 5. Jouer chaque rue
        street_reached = Street.PREFLOP
        for street in Street:
            if state.n_active <= 1:
                break
            street_reached = street

            # Compléter le board
            self._deal_board(state, street, deck)

            # Tour de mise
            all_folded = self._betting_round(state, street, bots)
            if all_folded:
                break

            # Réinitialiser les mises pour la prochaine rue
            for p in state.players:
                p.bet_this_street = 0 # ❌ à vérifier que le to_call est aussi remis à 0 à chaque street, sauf préflop où les blindes comptent encore
            state.to_call   = 0
            state.min_raise = self.big_blind

        # 6. Showdown et attribution des gains
        result = self._showdown(state, street_reached)
        result.actions = list(state.action_history)

        # 7. Mettre à jour les stacks persistants
        for pid, gain in result.gains.items():
            self.stacks[pid] = state.get_player(pid).stack # ❌ je ne comprends pas comment est gérer le stack

        # 8. Faire tourner le bouton
        self.btn_seat = (self.btn_seat + 1) % self.n_players

        return result

    def run_session(
        self,
        bots:       dict[int, BotFn],
        n_hands:    int = 10000,
        verbose:    bool = False,
    ) -> dict[str, Any]:
        """Lance n_hands mains et retourne les statistiques agrégées."""
        gains_total: dict[int, int] = {i: 0 for i in range(self.n_players)}
        hands_played = 0

        for _ in range(n_hands):
            # Éliminer les joueurs sans stack
            active_bots = {pid: bot for pid, bot in bots.items() if self.stacks[pid] > 0}
            if len(active_bots) < 2:
                break

            result = self.run_hand(active_bots)
            hands_played += 1

            for pid, gain in result.gains.items():
                gains_total[pid] += gain

            if verbose and hands_played % 1000 == 0:
                print(f"  Main {hands_played}/{n_hands} — stacks: {self.stacks}")

        # Calcul BB/100
        bb100: dict[int, float] = {}
        for pid in range(self.n_players):
            # en moyenne, le joueur gagne X big blinds toutes les 100 mains.
            bb100[pid] = (gains_total[pid] / self.big_blind) / (hands_played / 100) 

        data: dict[str, Any] = {
            "hands_played": hands_played,
            "gains_total":  gains_total,
            "bb_per_100":   bb100,
            "final_stacks": dict(self.stacks),
        }

        return data

    # ── Internals ─────────────────────────────────────────────────────────────

    def _create_players(self) -> list[Player]:
        positions = self.POSITIONS_BY_N[self.n_players]
        # Le BTN est au siège btn_seat, on assigne les positions dans l'ordre
        players: list[Player] = []
        for i in range(self.n_players):
            seat = (self.btn_seat + i) % self.n_players
            pos  = positions[i]
            players.append(Player(
                player_id=seat,
                name=f"P{seat}",
                stack=self.stacks[seat],
                position=pos,
            ))
        return players

    def _post_blindes(self, state: GameState) -> int:
        """Poste les blindes, retourne le pot initial."""
        # Heads-up : BTN = SB
        if state.n_active == 2:
            sb_player = next(p for p in state.players if p.position == Position.BTN)
            bb_player = next(p for p in state.players if p.position == Position.BB)
        else:
            sb_player = next(p for p in state.players if p.position == Position.SB)
            bb_player = next(p for p in state.players if p.position == Position.BB)

        sb_amount = min(self.small_blind, sb_player.stack)
        bb_amount = min(self.big_blind,   bb_player.stack)

        sb_player.stack           -= sb_amount
        sb_player.bet_this_street  = sb_amount
        bb_player.stack           -= bb_amount
        bb_player.bet_this_street  = bb_amount

        if sb_player.stack == 0: sb_player.status = PlayerStatus.ALLIN
        if bb_player.stack == 0: bb_player.status = PlayerStatus.ALLIN

        state.pot      = sb_amount + bb_amount
        state.to_call  = bb_amount
        state.min_raise = bb_amount * 2


        # Mises forcées (SB/BB) — ActionType.POST_BLIND, distinct de RAISE.
        # Une mise forcée n'est jamais une décision volontaire du joueur ;
        # la confondre avec un RAISE faussait tous les calculs en aval qui
        # scrutent action_history (VPIP/PFR dans hand_recorder.py et
        # range_estimator.py, comptage 3bet/4bet dans action_history.py —
        # cf. session 6, où une simple ouverture se retrouvait comptée
        # comme un 4bet à cause des 2 blindes déjà taguées RAISE).
        state.action_history.append(Action(sb_player.player_id, ActionType.POST_BLIND, sb_amount, Street.PREFLOP))
        state.action_history.append(Action(bb_player.player_id, ActionType.POST_BLIND, bb_amount, Street.PREFLOP))

        return state.pot

    def _deal_hole_cards(self, state: GameState, deck: Deck) -> None:
        for p in state.players:
            p.hole_cards = deck.deal(2) # ❌ à vérifier que les cartes sont retirés du deck

    def _deal_board(self, state: GameState, street: Street, deck: Deck) -> None:
        if street == Street.FLOP:
            state.board = deck.deal(3)
        elif street == Street.TURN:
            state.board += deck.deal(1)
        elif street == Street.RIVER:
            state.board += deck.deal(1)
        state.street = street

    def _betting_round( 
        self,
        state:  GameState,
        street: Street,
        bots:   dict[int, BotFn],
    ) -> bool:
        """
        Gère un tour de mise complet.
        Retourne True si tous les joueurs sauf un ont foldé.
        """
        # Ordre d'action selon la rue
        if street == Street.PREFLOP:
            action_order = self._preflop_order(state)
            state.to_call = self.big_blind
        else:
            action_order = self._postflop_order(state)
            state.to_call = 0
            state.min_raise = self.big_blind

        # Réinitialiser les mises de rue (sauf préflop où les blindes comptent)
        if street != Street.PREFLOP:
            for p in state.players:
                p.bet_this_street = 0

        # ── Curseur circulaire ────────────────────────────────────────────────
        # Règle : après un raise par X, le prochain joueur est celui APRÈS X
        # dans l'ordre. X ne reparle pas, sauf si quelqu'un re-raise après lui.
        # Le tour se ferme quand tous les joueurs actifs ont une mise égale
        # ET que le dernier aggressor (ou la BB préflop) a eu sa chance d'agir.
        n = len(action_order)
        if n == 0:
            return False
 
        current_idx        = 0        # index dans action_order
        last_aggressor_id: int | None = None
        players_acted: set[int] = set()
        max_iterations     = n * (n + 1) * 4  # garde-fou anti-boucle infinie
        iterations         = 0
 
        while iterations < max_iterations:
            iterations += 1
 
            # Trouver le prochain joueur pouvant agir à partir de current_idx
            player = None
            for offset in range(n):
                idx       = (current_idx + offset) % n
                candidate = action_order[idx]
                if candidate.can_act:
                    player      = candidate
                    current_idx = (idx + 1) % n   # le prochain tour repart après lui
                    break
 
            if player is None:
                break  # plus personne ne peut agir
 
            # Mettre à jour to_call pour ce joueur
            max_bet       = max(p.bet_this_street for p in state.players)
            state.to_call = max(0, max_bet - player.bet_this_street)
 
            # Vérifier si le tour est déjà fermé AVANT de demander au joueur
            # (cas : tout le monde a callé et on est revenu au dernier aggressor,
            #  ou tout le monde a checké)
            can_act_now  = [p for p in state.players if p.can_act]
            max_bet_now  = max(p.bet_this_street for p in state.players)
            all_even     = all(p.bet_this_street == max_bet_now for p in can_act_now)
            
            if all_even and player.player_id in players_acted:
                if last_aggressor_id is None:
                    # Postflop sans mise : les checks ont tous été faits
                    # Préflop : la BB doit encore avoir son option
                    if street != Street.PREFLOP:
                        break
                    # Préflop — vérifier si la BB a déjà agi volontairement
                    bb = next(
                        (p for p in state.players
                         if p.position == Position.BB and p.can_act),
                        None
                    )
                    if bb is None or bb.player_id != player.player_id:
                        break  # la BB a déjà parlé ou est partie → fin du tour
                    # sinon on laisse la BB parler (option)
                else:
                    # Il y a eu un raise : on ferme si le joueur courant
                    # est le dernier aggressor (tout le monde l'a callé/foldé)
                    if player.player_id == last_aggressor_id:
                        break
 
            # ── Demander l'action au bot ──────────────────────────────────────
            if player.player_id in bots:
                action = bots[player.player_id](state)
            else:
                action = self._default_action(state, player)
 
            action.street    = street
            action.player_id = player.player_id
 
            self._apply_action(state, player, action, max_bet)
            state.action_history.append(action)
            players_acted.add(player.player_id)
            
            # Après un raise/allin : le prochain closing point est le raiser.
            # current_idx pointe déjà sur le joueur APRÈS le raiser.
            if action.action_type in (ActionType.RAISE, ActionType.ALLIN):
                last_aggressor_id  = player.player_id
                state.aggressor_id = player.player_id
 
            # Fin de main si un seul joueur n'a pas foldé
            remaining = [p for p in state.players if p.status != PlayerStatus.FOLDED]
            if len(remaining) <= 1:
                return True

        return False

    def _apply_action(
        self,
        state:   GameState,
        player:  Player,
        action:  Action,
        max_bet: int,
    ) -> None:
        if action.action_type == ActionType.FOLD:
            player.status = PlayerStatus.FOLDED
 
        elif action.action_type == ActionType.CHECK: # ❌ à ajouter aussi dans les bots
            # CHECK n'est légal que si rien n'est à suivre pour ce joueur.
            if state.to_call > 0:
                # Action illégale du bot : on force un FOLD plutôt que d'accepter
                # silencieusement un check sur une mise ouverte.
                action.action_type = ActionType.FOLD
                player.status      = PlayerStatus.FOLDED
 
        elif action.action_type == ActionType.CALL:
            call_amount             = min(state.to_call, player.stack)
            player.stack           -= call_amount
            player.bet_this_street += call_amount
            state.pot              += call_amount
            if player.stack == 0:
                player.status      = PlayerStatus.ALLIN
            action.amount = call_amount
 
        elif action.action_type in (ActionType.RAISE, ActionType.ALLIN):
            # action.amount = total misé sur la rue après le raise
            # (convention : le bot envoie le montant TOTAL de sa mise de rue,
            #  pas juste l'incrément du raise)
            total_bet_intended = action.amount
 
            # Montant supplémentaire à mettre dans le pot depuis le stack
            additional = max(0, total_bet_intended - player.bet_this_street)
 
            # Capper au stack disponible (all-in automatique si insuffisant)
            actual_additional = min(additional, player.stack)
 
            if actual_additional <= 0:
                # Le bot a envoyé un raise à 0 ou inférieur à sa mise actuelle :
                # c'est une action invalide → on force un CALL (ou CHECK si rien à call)
                if state.to_call > 0:
                    call_amount             = min(state.to_call, player.stack)
                    player.stack           -= call_amount
                    player.bet_this_street += call_amount
                    state.pot              += call_amount
                    action.action_type     = ActionType.CALL
                    action.amount          = call_amount
                    if player.stack == 0:
                        player.status      = PlayerStatus.ALLIN
                else:
                    action.action_type = ActionType.CHECK
                return
 
            player.stack           -= actual_additional
            player.bet_this_street += actual_additional
            state.pot              += actual_additional
 
            if player.stack == 0:
                player.status      = PlayerStatus.ALLIN
                action.action_type = ActionType.ALLIN
            else:
                action.action_type = ActionType.RAISE
 
            action.amount = player.bet_this_street
 
            # Nouveau min-raise = taille du raise précédent
            # min_raise = bet_total + (bet_total - max_bet_before)
            raise_size        = player.bet_this_street - max_bet
            state.min_raise   = player.bet_this_street + max(raise_size, self.big_blind)

    def _default_action(self, state: GameState, player: Player) -> Action:
        """Action par défaut si le bot n'est pas dans le dict (call/check)."""
        if state.to_call == 0:
            return Action(player.player_id, ActionType.CHECK, 0)
        return Action(player.player_id, ActionType.CALL, state.to_call)

    def _preflop_order(self, state: GameState) -> list[Player]:
        """Ordre préflop : UTG → ... → BTN → SB → BB."""
        pos_order = [
            Position.UTG, Position.UTG1, Position.MP,
            Position.CO, Position.BTN, Position.SB, Position.BB
        ]
        ordered: list[Player] = []
        for pos in pos_order:
            p = next((pl for pl in state.players if pl.position == pos and pl.can_act), None) # ❌ est ce que le can_act s'occupe de ceux fold ou allin ?
            if p:
                ordered.append(p)
        return ordered

    def _postflop_order(self, state: GameState) -> list[Player]:
        """Ordre postflop : SB → BB → UTG → ... → BTN."""
        pos_order = [
            Position.SB, Position.BB, Position.UTG, Position.UTG1,
            Position.MP, Position.CO, Position.BTN
        ]
        ordered: list[Player] = []
        for pos in pos_order:
            p = next((pl for pl in state.players if pl.position == pos and pl.can_act), None)
            if p:
                ordered.append(p)
        return ordered

    # ── Showdown ──────────────────────────────────────────────────────────────

    def _showdown(self, state: GameState, street_reached: Street) -> HandResult:
        active = [p for p in state.players if p.status != PlayerStatus.FOLDED]
        board_strs = [str(c) for c in state.board]

        gains: dict[int, int] = {}
        for p in state.players:
            # Gain net = stack actuel - stack initial
            gains[p.player_id] = p.stack - self.stacks[p.player_id]

        showdown_cards: dict[int, list[str]] = {}
        winners = []

        if len(active) == 1:
            # Tout le monde a foldé
            winner = active[0]
            winner.stack += state.pot
            gains[winner.player_id] = winner.stack - self.stacks[winner.player_id]
            winners = [winner.player_id]
        else:
            # Showdown réel — comparer les mains
            # Compléter le board à 5 cartes si nécessaire (all-in avant la river)
            board_complete = list(board_strs)
            if len(board_complete) < 5:
                deck_remaining = Deck()
                known = [str(c) for c in state.board]
                for p in active:
                    known += [str(c) for c in p.hole_cards]
                deck_remaining.remove([Card.from_str(s) for s in known])
                needed = 5 - len(board_complete)
                board_complete += [str(deck_remaining.deal_one()) for _ in range(needed)]

            # Évaluer chaque main
            scores: dict[int, Any] = {}
            for p in active:
                hand_7 = [str(c) for c in p.hole_cards] + board_complete
                scores[p.player_id] = evaluate_hand(hand_7)
                showdown_cards[p.player_id] = [str(c) for c in p.hole_cards]

            # Trouver le(s) gagnant(s)
            best_score = max(scores.values())
            winners = [pid for pid, s in scores.items() if s == best_score]

            # Distribuer le pot
            share = state.pot // len(winners)
            remainder = state.pot % len(winners)
            for pid in winners:
                p = state.get_player(pid)
                p.stack += share
            # Le reste va au premier gagnant (convention)
            if remainder > 0:
                state.get_player(winners[0]).stack += remainder

            # Recalculer les gains nets
            for p in state.players:
                gains[p.player_id] = p.stack - self.stacks[p.player_id]

        return HandResult(
            hand_id=state.hand_id,
            winners=winners,
            gains=gains,
            showdown=showdown_cards,
            final_pot=state.pot,
            street_reached=street_reached,
        )
