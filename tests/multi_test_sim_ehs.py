#!/usr/bin/env python3
"""
multi_test_sim_ehs.py — Simulations EHSBot vs RuleBot vs RandomBot
Phase 2 — Bot Poker Academique

Modes disponibles :
  --mode mixed     : RandomBot | RuleBot | EHSBot        (defaut)
  --mode vs_rule   : RuleBot   | RuleBot | EHSBot
  --mode vs_random : RandomBot | RandomBot | EHSBot

Placement : tests/
Commande  : depuis la racine du projet
    python tests/multi_test_sim_ehs.py
    python tests/multi_test_sim_ehs.py --mode vs_rule
    python tests/multi_test_sim_ehs.py --mode vs_random
    python tests/multi_test_sim_ehs.py --tests 20 --hands 2000 --sims 500
"""

import sys
import os
import time
import argparse
import numpy as np
from typing import Dict, List, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.simulator import PokerTable
from core.bots.test_bots import RandomBot, RuleBot
from core.game_state import Action, GameState, ActionType, Street
from core.bots.ehs_bot import EHSBot, EHSBotConfig


# =============================================================================
# Parametres par defaut
# =============================================================================

DEFAULT_N_TESTS = 100
DEFAULT_N_HANDS = 10000
DEFAULT_N_SIMS  = 1000
N_PLAYERS       = 3
STARTING_STACKS = 1000
BIG_BLIND       = 10

MODES = {
    'mixed':     {0: 'RandomBot', 1: 'RuleBot',   2: 'EHSBot'},
    'vs_rule':   {0: 'RuleBot',   1: 'RuleBot',   2: 'EHSBot'},
    'vs_random': {0: 'RandomBot', 1: 'RandomBot', 2: 'EHSBot'},
}


# =============================================================================
# Conversion GameState -> dict pour EHSBot
# =============================================================================

def gamestate_to_dict(state: GameState, player_id: int) -> dict:
    player = state.get_player(player_id)

    hand  = [str(c) for c in player.hole_cards] if player.hole_cards else []
    board = [str(c) for c in state.board]        if state.board       else []

    street_map = {
        Street.PREFLOP: 'preflop',
        Street.FLOP:    'flop',
        Street.TURN:    'turn',
        Street.RIVER:   'river',
    }
    street   = street_map.get(state.street, 'preflop')
    position = player.position.name if player.position else 'BTN'

    players_info = [
        {'id': p.player_id, 'stack': p.stack}
        for p in state.players
        if p.player_id != player_id and p.can_act
    ]

    return {
        'hand':           hand,
        'board':          board,
        'street':         street,
        'pot':            float(state.pot),
        'to_call':        float(state.to_call),
        'stack':          float(player.stack),
        'position':       position,
        'players':        players_info,
        'action_history': [],
    }


# =============================================================================
# Conversion Action EHSBot -> Action simulateur
# =============================================================================

ACTION_TYPE_MAP = {
    'fold':  ActionType.FOLD,
    'check': ActionType.CHECK,
    'call':  ActionType.CALL,
    'bet':   ActionType.RAISE,
    'raise': ActionType.RAISE,
    'allin': ActionType.ALLIN,
}

def ehs_action_to_sim_action(ehs_action, player_id: int, state: GameState) -> Action:
    action_type = ACTION_TYPE_MAP.get(ehs_action.action_type, ActionType.FOLD)
    amount = int(state.to_call) if action_type == ActionType.CALL else int(ehs_action.amount)
    return Action(player_id=player_id, action_type=action_type, amount=amount)


# =============================================================================
# Wrapper EHSBot
# =============================================================================

def make_ehs_bot_fn(player_id: int, config: EHSBotConfig) -> Callable:
    bot = EHSBot(config=config)

    def bot_fn(state: GameState) -> Action:
        gs_dict    = gamestate_to_dict(state, player_id)
        ehs_action = bot.decide(gs_dict)
        return ehs_action_to_sim_action(ehs_action, player_id, state)

    return bot_fn


def make_bot_fn(name: str, player_id: int, config: EHSBotConfig) -> Callable:
    """Instancie le bon bot selon le nom."""
    if name == 'EHSBot':
        return make_ehs_bot_fn(player_id, config)
    elif name == 'RuleBot':
        return RuleBot(player_id).__call__
    else:
        return RandomBot(player_id).__call__


# =============================================================================
# Execution d'une session unique
# =============================================================================

def run_single_test(
    seed:    int,
    n_hands: int,
    config:  EHSBotConfig,
    mode:    str,
) -> Dict[int, float]:
    table = PokerTable(
        n_players=N_PLAYERS,
        starting_stacks=STARTING_STACKS,
        big_blind=BIG_BLIND,
        seed=seed,
    )

    bot_names = MODES[mode]
    bots = {pid: make_bot_fn(name, pid, config) for pid, name in bot_names.items()}

    stats = table.run_session(bots, n_hands=n_hands)
    return stats['bb_per_100']


# =============================================================================
# Statistiques
# =============================================================================

def compute_statistics(
    results:   List[Dict[int, float]],
    n_hands:   int,
    n_sims:    int,
    mode:      str,
) -> None:
    n_tests  = len(results)
    bb_data  = np.array([[r[i] for i in range(N_PLAYERS)] for r in results])
    labels   = MODES[mode]

    print("\n" + "=" * 60)
    print(f"STATISTIQUES SUR {n_tests} SESSIONS ({n_hands} mains, {n_sims} sims EHS)")
    print(f"Mode : {mode}  —  {labels[0]} | {labels[1]} | {labels[2]}")
    print("=" * 60)

    # BB/100 moyen et ecart-type
    means = np.mean(bb_data, axis=0)
    stds  = np.std(bb_data, axis=0)

    print("\n[BB/100]  Moyenne +/- ecart-type :")
    for i in range(N_PLAYERS):
        print(f"  Joueur {i} ({labels[i]:9}) : {means[i]:+6.2f} +/- {stds[i]:5.2f}")

    # Victoires
    wins = np.zeros(N_PLAYERS, dtype=int)
    for bb in bb_data:
        best_val = np.max(bb)
        for idx in np.where(bb == best_val)[0]:
            wins[idx] += 1

    print("\n[Victoires] (meilleur BB/100 sur la session) :")
    for i in range(N_PLAYERS):
        print(f"  Joueur {i} ({labels[i]:9}) : {wins[i]:3d}/{n_tests} ({100*wins[i]/n_tests:.1f}%)")

    # Sessions beneficiaires
    positive = np.sum(bb_data > 0, axis=0)
    print("\n[Sessions beneficiaires] (BB/100 > 0) :")
    for i in range(N_PLAYERS):
        print(f"  Joueur {i} ({labels[i]:9}) : {positive[i]:3d}/{n_tests} ({100*positive[i]/n_tests:.1f}%)")

    # EHSBot vs adversaires
    ehs_bb = bb_data[:, 2]
    for opp_id in [0, 1]:
        opp_bb      = bb_data[:, opp_id]
        ehs_better  = np.sum(ehs_bb > opp_bb)
        opp_better  = np.sum(ehs_bb < opp_bb)
        equal       = np.sum(ehs_bb == opp_bb)
        advantage   = np.mean(ehs_bb - opp_bb)
        print(f"\n[EHSBot vs {labels[opp_id]}] :")
        print(f"  EHSBot meilleur    : {ehs_better:3d}/{n_tests} ({100*ehs_better/n_tests:.1f}%)")
        print(f"  {labels[opp_id]} meilleur : {opp_better:3d}/{n_tests} ({100*opp_better/n_tests:.1f}%)")
        print(f"  Egalite            : {equal:3d}/{n_tests} ({100*equal/n_tests:.1f}%)")
        print(f"  Avantage moyen EHSBot : {advantage:+.2f} BB/100")

    # Resume
    print("\n" + "=" * 60)
    print("RESUME — Classement moyen")
    print("=" * 60)
    ranking = sorted(range(N_PLAYERS), key=lambda i: means[i], reverse=True)
    for rank, i in enumerate(ranking, 1):
        print(f"  #{rank}  {labels[i]:9} : {means[i]:+.2f} BB/100")
    print()


# =============================================================================
# Point d'entree
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulation multi-sessions EHSBot vs bots phase 1"
    )
    parser.add_argument("--tests", type=int,  default=DEFAULT_N_TESTS)
    parser.add_argument("--hands", type=int,  default=DEFAULT_N_HANDS)
    parser.add_argument("--sims",  type=int,  default=DEFAULT_N_SIMS)
    parser.add_argument("--mode",  type=str,  default='mixed',
                        choices=list(MODES.keys()),
                        help="mixed | vs_rule | vs_random (defaut: mixed)")
    args = parser.parse_args()

    config    = EHSBotConfig(n_sims=args.sims)
    bot_names = MODES[args.mode]

    print(f"Configuration :")
    print(f"  Mode          : {args.mode}")
    print(f"  Sessions      : {args.tests}")
    print(f"  Mains/session : {args.hands}")
    print(f"  Sims EHS      : {args.sims} Monte Carlo par decision")
    print(f"  Joueurs       : 0={bot_names[0]} | 1={bot_names[1]} | 2=EHSBot")
    print()

    start   = time.time()
    results = []

    for test_idx in range(args.tests):
        seed = test_idx + 42
        bb   = run_single_test(seed, args.hands, config, args.mode)
        results.append(bb)

        if (test_idx + 1) % 10 == 0 or test_idx == 0:
            elapsed = time.time() - start
            done    = test_idx + 1
            eta     = elapsed / done * (args.tests - done)
            print(f"  Session {done:3d}/{args.tests} | "
                  f"EHS={bb[2]:+5.1f} {bot_names[1]}={bb[1]:+5.1f} {bot_names[0]}={bb[0]:+5.1f} BB/100 | "
                  f"ETA {eta:.0f}s")

    elapsed = time.time() - start
    print(f"\nTemps total : {elapsed:.1f}s (moy. {elapsed/args.tests:.2f}s/session)")

    compute_statistics(results, args.hands, args.sims, args.mode)


if __name__ == "__main__":
    main()
