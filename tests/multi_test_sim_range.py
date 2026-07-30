"""
multi_test_sim_range.py — Simulation EHSBot vs 9 RangeBots
Phase 3 — Bot Poker Académique

Modes :
  --mode quick    : 3 sessions × 50 mains    (test rapide, variance élevée)
  --mode standard : 10 sessions × 500 mains  (défaut)
  --mode validate : 50 sessions × 1000 mains (résultats fiables, ~14 min)

Placement : tests/
Commande   : depuis la racine du projet
    python tests/multi_test_sim_range.py
    python tests/multi_test_sim_range.py --mode validate
    python tests/multi_test_sim_range.py --hands 1000 --sessions 30 --sims 500
"""

import sys
import os
import time
import argparse
import math
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.simulator import PokerTable
from core.game_state import Action, GameState, ActionType, Street, PlayerStatus
from core.bots.ehs_bot import EHSBot, EHSBotConfig
from core.bots.range_bot import RangeBot, make_range_bot, make_all_range_bots
from core.bots.range_definitions import ARCHETYPES, MUTATIONS
from core.metrics_range import SessionReport, GlobalReport


# =============================================================================
# Modes prédéfinis
# =============================================================================

MODES = {
    'quick':    {'hands': 50,   'sessions': 3,  'sims': 200,
                 'label': 'Rapide (variance très élevée — non interprétable)'},
    'standard': {'hands': 500,  'sessions': 10, 'sims': 500,
                 'label': 'Standard (variance élevée — indicatif seulement)'},
    'validate': {'hands': 1000, 'sessions': 50, 'sims': 500,
                 'label': 'Validation (résultats fiables à ±20 BB/100 à 95%)'},
}

BIG_BLIND       = 10
N_PLAYERS       = 2
STARTING_STACKS = 1000

ACTION_TYPE_MAP = {
    'fold':  ActionType.FOLD,
    'check': ActionType.CHECK,
    'call':  ActionType.CALL,
    'bet':   ActionType.RAISE,
    'raise': ActionType.RAISE,
    'allin': ActionType.ALLIN,
}

STREET_MAP = {
    Street.PREFLOP: 'preflop',
    Street.FLOP:    'flop',
    Street.TURN:    'turn',
    Street.RIVER:   'river',
}


# =============================================================================
# Structure de résultat enrichie
# =============================================================================

@dataclass
class RangeBotResult:
    """Résultat statistiquement complet pour un RangeBot."""
    key:        str
    archetype:  str
    mutation:   int
    n_sessions: int
    n_hands:    int

    # BB/100
    mean_bb:    float
    std_bb:     float
    se_bb:      float      # erreur standard = std / sqrt(n)

    # Intervalle de confiance à 95% (t-distribution)
    ci_low:     float
    ci_high:    float

    # Significativité
    is_significant: bool   # True si l'IC ne croise pas zéro
    p_value:        float  # p-value approchée (t-test vs H0: mean=0)

    # Données brutes
    bb_per_session: List[float]

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low

    @property
    def verdict(self) -> str:
        if not self.is_significant:
            return "≈ Non significatif"
        return "✓ EHSBot gagne" if self.mean_bb > 0 else "✗ RangeBot gagne"


def compute_result(
    key:            str,
    archetype:      str,
    mutation:       int,
    bb_per_session: List[float],
    n_hands:        int,
) -> RangeBotResult:
    """
    Calcule toutes les statistiques depuis les résultats bruts par session.

    Utilise la t-distribution de Student pour l'IC (adapté aux petits échantillons).
    """
    n   = len(bb_per_session)
    arr = np.array(bb_per_session)

    mean = float(np.mean(arr))
    std  = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se   = std / math.sqrt(n) if n > 1 else float('inf')

    # t critique à 95% (ddof = n-1) — approximation normale pour n >= 30
    if n >= 30:
        t_crit = 1.96
    elif n >= 20:
        t_crit = 2.09
    elif n >= 10:
        t_crit = 2.23
    elif n >= 5:
        t_crit = 2.57
    else:
        t_crit = 3.18   # très conservateur pour n < 5

    margin   = t_crit * se
    ci_low   = mean - margin
    ci_high  = mean + margin

    # Significatif si l'IC entier est d'un seul côté de zéro
    is_sig = (ci_low > 0) or (ci_high < 0)

    # p-value approchée : P(|t| > |mean/se|) avec t-distribution
    t_stat = abs(mean / se) if se > 0 else 0.0
    # Approximation grossière — suffisante pour l'affichage
    if t_stat > 3.5:
        p_val = 0.001
    elif t_stat > 2.6:
        p_val = 0.01
    elif t_stat > 2.0:
        p_val = 0.05
    elif t_stat > 1.6:
        p_val = 0.10
    else:
        p_val = 1.0

    return RangeBotResult(
        key=key,
        archetype=archetype,
        mutation=mutation,
        n_sessions=n,
        n_hands=n_hands,
        mean_bb=mean,
        std_bb=std,
        se_bb=se,
        ci_low=ci_low,
        ci_high=ci_high,
        is_significant=is_sig,
        p_value=p_val,
        bb_per_session=list(bb_per_session),
    )


# =============================================================================
# Conversion GameState → dict (gère les objets Action du simulateur)
# =============================================================================

def gamestate_to_dict(state: GameState, player_id: int) -> dict:
    player = state.get_player(player_id)
    hand   = [str(c) for c in player.hole_cards] if player.hole_cards else []
    board  = [str(c) for c in state.board]        if state.board       else []

    players_info = [
        {'id': p.player_id, 'stack': p.stack}
        for p in state.players
        if p.player_id != player_id
        and p.status != PlayerStatus.FOLDED
    ]

    # action_history contient des objets Action (pas des dicts)
    action_history = []
    for act in getattr(state, 'action_history', []):
        street_str = STREET_MAP.get(
            getattr(act, 'street', Street.PREFLOP), 'preflop'
        )
        action_history.append({
            'street': street_str,
            'player': getattr(act, 'player_id', -1),
            'action': getattr(act, 'action_type', ActionType.FOLD).name.lower(),
            'amount': getattr(act, 'amount', 0),
        })

    return {
        'hand':           hand,
        'board':          board,
        'street':         STREET_MAP.get(state.street, 'preflop'),
        'pot':            float(state.pot),
        'to_call':        float(state.to_call),
        'stack':          float(player.stack),
        'position':       player.position.name if player.position else 'BTN',
        'players':        players_info,
        'player_id':      player_id,
        'big_blind':      float(BIG_BLIND),
        'action_history': action_history,
    }


# =============================================================================
# Wrappers bots → BotFn
# =============================================================================

def make_ehs_bot_fn(player_id: int, config: EHSBotConfig):
    bot = EHSBot(config=config)
    def fn(state: GameState) -> Action:
        gs  = gamestate_to_dict(state, player_id)
        act = bot.decide(gs)
        amt = int(state.to_call) if act.action_type == 'call' else int(act.amount)
        return Action(
            player_id=player_id,
            action_type=ACTION_TYPE_MAP.get(act.action_type, ActionType.FOLD),
            amount=amt,
        )
    return fn


def make_range_bot_fn(player_id: int, range_bot: RangeBot):
    def fn(state: GameState) -> Action:
        gs  = gamestate_to_dict(state, player_id)
        act = range_bot.decide(gs)
        amt = int(state.to_call) if act.action_type == 'call' else int(act.amount)
        return Action(
            player_id=player_id,
            action_type=ACTION_TYPE_MAP.get(act.action_type, ActionType.FOLD),
            amount=amt,
        )
    return fn


# =============================================================================
# Session unique
# =============================================================================

def run_single_session(
    seed:       int,
    n_hands:    int,
    ehs_config: EHSBotConfig,
    range_bot:  RangeBot,
) -> float:
    """Retourne le BB/100 de l'EHSBot (joueur 0) pour cette session."""
    table = PokerTable(
        n_players=N_PLAYERS,
        starting_stacks=STARTING_STACKS,
        big_blind=BIG_BLIND,
        seed=seed,
    )
    bots = {
        0: make_ehs_bot_fn(0, ehs_config),
        1: make_range_bot_fn(1, range_bot),
    }
    stats = table.run_session(bots, n_hands=n_hands)
    return stats['bb_per_100'].get(0, 0.0)


# =============================================================================
# Évaluation complète
# =============================================================================

def _fmt_eta(seconds: float) -> str:
    """Formate un nombre de secondes en chaine lisible."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{int(seconds)//60}m{int(seconds)%60:02d}s"
    else:
        h = int(seconds) // 3600
        m = (int(seconds) % 3600) // 60
        return f"{h}h{m:02d}m"


def _fmt_eta(seconds: float) -> str:
    """Formate un nombre de secondes en chaine lisible."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{int(seconds)//60}m{int(seconds)%60:02d}s"
    else:
        h = int(seconds) // 3600
        m = (int(seconds) % 3600) // 60
        return f"{h}h{m:02d}m"


def evaluate_all_rangebots(
    n_hands:    int,
    n_sessions: int,
    ehs_sims:   int,
    verbose:    bool = True,
) -> Dict[str, RangeBotResult]:
    ehs_config   = EHSBotConfig(n_sims=ehs_sims)
    all_bots     = make_all_range_bots(n_sims=ehs_sims)
    results      = {}
    n_bots       = len(all_bots)
    total_tasks  = n_bots * n_sessions
    tasks_done   = 0
    global_start = time.time()

    for bot_idx, (key, range_bot) in enumerate(all_bots.items()):
        cfg         = range_bot.config
        bb_sessions = []
        bot_start   = time.time()

        if verbose:
            if tasks_done > 0:
                elapsed   = time.time() - global_start
                eta_total = (total_tasks - tasks_done) / (tasks_done / elapsed)
                eta_str   = f"ETA total : {_fmt_eta(eta_total)}"
            else:
                eta_str   = "ETA total : calcul en cours..."
            print(f"\\n  [{bot_idx+1}/{n_bots}] {key:30s}  ({eta_str})")
            print(f"  {chr(8212)*65}")

        for session_idx in range(n_sessions):
            seed = session_idx * 100 + abs(hash(key)) % 1000
            sess_start = time.time()
            try:
                bb = run_single_session(seed, n_hands, ehs_config, range_bot)
                bb_sessions.append(bb)
            except Exception as e:
                if verbose:
                    print(f"    ERREUR session {session_idx}: {e}")
                bb_sessions.append(0.0)

            tasks_done   += 1
            sess_elapsed  = time.time() - sess_start

            if verbose:
                sessions_left_bot = n_sessions - (session_idx + 1)
                eta_bot    = sess_elapsed * sessions_left_bot
                total_el   = time.time() - global_start
                eta_global = (total_tasks - tasks_done) / (tasks_done / total_el)
                running_mean = float(np.mean(bb_sessions))
                print(f"    Session {session_idx+1:3d}/{n_sessions}"
                      f"  BB={bb:+7.1f}"
                      f"  moy={running_mean:+7.1f}"
                      f"  ETA bot:{_fmt_eta(eta_bot):>7}"
                      f"  ETA total:{_fmt_eta(eta_global):>7}",
                      flush=True)

        result = compute_result(key, cfg.archetype, cfg.mutation,
                                bb_sessions, n_hands)

        if verbose:
            bot_elapsed = time.time() - bot_start
            sig = "★" if result.is_significant else "○"
            print(f"  -> {key}: {result.mean_bb:+.1f} BB/100  "
                  f"IC95%=[{result.ci_low:+.0f}, {result.ci_high:+.0f}]  "
                  f"p={result.p_value:.3f}  {sig}  "
                  f"(termine en {_fmt_eta(bot_elapsed)})")

        results[key] = result

    if verbose:
        print(f"\\n  Temps total : {_fmt_eta(time.time() - global_start)}")

    return results


# =============================================================================
# Affichage du rapport
# =============================================================================

def print_report(
    results:    Dict[str, RangeBotResult],
    n_hands:    int,
    n_sessions: int,
    ehs_sims:   int,
    mode:       str,
):
    total_hands = n_hands * n_sessions * 9
    all_means   = [r.mean_bb for r in results.values()]
    mean_global = float(np.mean(all_means))

    print("\n" + "=" * 75)
    print("RAPPORT — EHSBot v2 vs 9 RangeBots")
    print(f"  Mode : {mode} — {MODES[mode]['label']}")
    print(f"  {n_sessions} sessions × {n_hands} mains × 9 bots = {total_hands:,} mains totales")
    print(f"  {ehs_sims} simulations Monte Carlo EHS par décision")
    print("=" * 75)

    print(f"\n  BB/100 moyen global EHSBot : {mean_global:+.2f}")

    # Avertissement variance
    if n_sessions < 30:
        print()
        print("  ⚠ ATTENTION — VARIANCE")
        min_sessions = 50
        print(f"    Avec {n_sessions} sessions, l'intervalle de confiance est très large.")
        print(f"    Les résultats non marqués ★ ne sont PAS statistiquement significatifs.")
        print(f"    → Utiliser --mode validate ({min_sessions} sessions) pour des résultats fiables.")

    print()
    print(f"  {'Bot':30s} {'BB/100':>8}  {'IC 95%':>20}  {'p':>5}  Verdict")
    print(f"  {'─'*30} {'─'*8}  {'─'*20}  {'─'*5}  {'─'*22}")

    for arch in ARCHETYPES:
        for mut in MUTATIONS:
            key = f"{arch}_{mut}"
            r   = results.get(key)
            if not r:
                continue
            ic_str  = f"[{r.ci_low:+.0f}, {r.ci_high:+.0f}]"
            p_str   = f"{r.p_value:.3f}" if r.p_value < 1.0 else ">0.10"
            sig     = " ★" if r.is_significant else "  "
            verdict = r.verdict
            print(f"  {key:30s} {r.mean_bb:+8.1f}  {ic_str:>20}  {p_str:>5}{sig}  {verdict}")

    print()
    print("  ★ = résultat significatif à 95%  |  ○ = non significatif (variance)")
    print()

    # Par archétype
    print("  [Par archétype]")
    for arch in ARCHETYPES:
        arch_results = [r for r in results.values() if r.archetype == arch]
        means        = [r.mean_bb for r in arch_results]
        n_sig_pos    = sum(1 for r in arch_results if r.is_significant and r.mean_bb > 0)
        n_sig_neg    = sum(1 for r in arch_results if r.is_significant and r.mean_bb < 0)
        n_nonsig     = sum(1 for r in arch_results if not r.is_significant)
        print(f"  {arch:25s} : BB/100 moy={float(np.mean(means)):+.1f}  "
              f"★gagne={n_sig_pos}/3  ★perd={n_sig_neg}/3  ○non-sig={n_nonsig}/3")

    # Résumé significativité
    n_sig_pos  = sum(1 for r in results.values() if r.is_significant and r.mean_bb > 0)
    n_sig_neg  = sum(1 for r in results.values() if r.is_significant and r.mean_bb < 0)
    n_nonsig   = sum(1 for r in results.values() if not r.is_significant)

    print()
    print(f"  Résultats significatifs : EHSBot gagne={n_sig_pos}/9  "
          f"RangeBot gagne={n_sig_neg}/9  non-sig={n_nonsig}/9")

    if mode != 'validate':
        print()
        print("  → Pour des conclusions fiables, lancer :")
        print("    python tests\\multi_test_sim_range.py --mode validate")
    print()


# =============================================================================
# Point d'entrée
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EHSBot v2 vs 9 RangeBots — Phase 3"
    )
    parser.add_argument("--mode", type=str, default='standard',
                        choices=list(MODES.keys()),
                        help="quick | standard | validate")
    parser.add_argument("--hands",    type=int, default=None)
    parser.add_argument("--sessions", type=int, default=None)
    parser.add_argument("--sims",     type=int, default=None)
    args = parser.parse_args()

    # Les arguments explicites écrasent le mode
    mode_cfg   = MODES[args.mode]
    n_hands    = args.hands    or mode_cfg['hands']
    n_sessions = args.sessions or mode_cfg['sessions']
    ehs_sims   = args.sims     or mode_cfg['sims']

    print(f"Configuration :")
    print(f"  Mode          : {args.mode} — {mode_cfg['label']}")
    print(f"  Mains/session : {n_hands}")
    print(f"  Sessions      : {n_sessions} par RangeBot")
    print(f"  Sims EHS      : {ehs_sims}")
    print(f"  Total mains   : {n_hands * n_sessions * 9:,}")
    print()

    start   = time.time()
    results = evaluate_all_rangebots(
        n_hands=n_hands,
        n_sessions=n_sessions,
        ehs_sims=ehs_sims,
        verbose=True,
    )
    elapsed = time.time() - start
    print(f"\nTemps total : {elapsed:.1f}s")

    print_report(results, n_hands, n_sessions, ehs_sims, args.mode)


if __name__ == "__main__":
    main()
