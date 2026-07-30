"""
diag_vpip_trace.py — Diagnostic ponctuel (pas un livrable permanent)

A lancer depuis la racine du projet :
    python diag_vpip_trace.py

Compare, pour un RangeBot donne :
  1. Le range_pct CONFIGURE (verite terrain, aucun jeu implique)
  2. Le VPIP MESURE en rejouant N vraies mains via PokerTable + hand_recorder
     (exactement le pipeline de validate_phase6.py)
  3. Une trace detaillee des 20 premieres mains (main distribuee, dans la
     range ou non selon combo_in_range(), decision reelle du bot, ce qui
     est effectivement enregistre en DB) pour voir OU ca diverge.
"""
import sys
sys.path.insert(0, '.')

from core.simulator import PokerTable
from core.bots.range_bot import make_range_bot
from core.bots.range_definitions import combo_in_range
from core.game_state import Action, ActionType, Street, Position, PlayerStatus
from core.player_db.player_db import PlayerDB
from core.player_db.hand_recorder import record_hand
from core.player_db.profile_builder import build_profile

_MAP = {'fold': ActionType.FOLD, 'check': ActionType.CHECK, 'call': ActionType.CALL,
        'bet': ActionType.RAISE, 'raise': ActionType.RAISE, 'allin': ActionType.ALLIN}


def make_bot_fn(bot, player_id, capture=None, trace=None):
    def bot_fn(state):
        our = state.get_player(player_id)
        hand = [str(c) for c in our.hole_cards] if our.hole_cards else []
        players_dict = [
            {'id': p.player_id, 'player_id': p.player_id, 'stack': float(p.stack),
             'status': p.status.value, 'position': p.position.value}
            for p in state.players if p.player_id != player_id
        ]
        gs_dict = {
            'hand': hand, 'board': state.board_str(), 'pot': float(state.pot),
            'to_call': float(state.to_call), 'stack': float(our.stack),
            'position': our.position.value, 'street': state.street.value,
            'players': players_dict, 'player_id': player_id,
            'big_blind': float(state.big_blind),
        }
        if capture is not None:
            capture['board'] = state.board_str()
            other = next((p for p in state.players if p.player_id != player_id), None)
            if other is not None:
                capture['opponent_position'] = other.position.value

        internal_action = bot.decide(gs_dict)

        if trace is not None and state.street == Street.PREFLOP and len(hand) == 2:
            in_range = combo_in_range(hand[0], hand[1], bot.config.preflop_combos)
            trace.append({
                'hand': hand, 'in_range': in_range,
                'decision': internal_action.action_type,
            })

        at = _MAP.get(internal_action.action_type, ActionType.CHECK)
        amount = int(getattr(internal_action, 'amount', 0))
        return Action(player_id, at, amount, state.street)
    return bot_fn


def run_diagnostic(archetype, mutation, n_hands=200, verbose_n=20):
    bot = make_range_bot(archetype, mutation, n_sims=500)
    true_pct = bot.config.range_pct
    print(f"\n{'='*70}")
    print(f"  {archetype} mutation {mutation} — range_pct CONFIGURE = {true_pct:.1%}")
    print(f"{'='*70}")

    class DummyBot:
        def decide(self, gs):
            class A:
                action_type = 'fold'
                amount = 0
            return A()
        def reset(self): pass

    dummy = DummyBot()
    trace = []
    capture = {}
    bot_fn  = make_bot_fn(dummy, 0)
    opp_fn  = make_bot_fn(bot, 1, capture=capture, trace=trace)

    table = PokerTable(n_players=2, starting_stacks=1000, small_blind=5,
                        big_blind=10, seed=123)

    import tempfile, os
    tmpdir = tempfile.mkdtemp()
    db = PlayerDB(os.path.join(tmpdir, 'diag.sqlite3'))

    hands_played = 0
    for i in range(n_hands):
        for pid in [0, 1]:
            if table.stacks[pid] < 700:
                table.stacks[pid] = 1000
        try:
            result = table.run_hand(bots={0: bot_fn, 1: opp_fn})
            hands_played += 1
        except Exception as e:
            print(f"  Main {i} echouee: {e}")
            continue

        record_hand(db=db, hand_result=result, board=capture.get('board', []),
                    player_id_map={1: 'diag_villain'},
                    position_map={1: capture.get('opponent_position')},
                    our_player_id=0)
        dummy.reset(); bot.reset()

    row = db.get_player_row('diag_villain')
    print(f"  Mains jouees (table)     : {hands_played}")
    print(f"  hands_seen (DB)          : {row.hands_seen}")
    print(f"  preflop_opportunities    : {row.preflop_opportunities}")
    print(f"  hands_vpip (DB)          : {row.hands_vpip}")
    print(f"  VPIP MESURE (corrigé)    : {row.vpip:.1%}")
    print(f"  Ecart vs configure       : {row.vpip - true_pct:+.1%}")

    profile = build_profile(db, 'diag_villain')
    print(f"  Archétype classé (DB)    : {profile.archetype} (tier={profile.tier})")
    flag = "OK" if profile.archetype == archetype else "MAL CLASSE"
    print(f"  -> attendu {archetype}, obtenu {profile.archetype}  [{flag}]")

    n_trace_in_range   = sum(1 for t in trace if t['in_range'])
    n_trace_total      = len(trace)
    print(f"\n  Sur {n_trace_total} decisions preflop tracees :")
    print(f"    combo_in_range()=True  : {n_trace_in_range} ({n_trace_in_range/max(n_trace_total,1):.1%})")
    print(f"    -> ce ratio DEVRAIT etre proche de {true_pct:.1%} (config)")

    print(f"\n  10 premieres decisions tracees :")
    for t in trace[:10]:
        print(f"    main={t['hand']}  in_range={t['in_range']}  decision='{t['decision']}'")
        if t['in_range'] and t['decision'] == 'fold':
            print(f"      ⚠️  INCOHERENT : main dans la range mais decision=fold")
        if not t['in_range'] and t['decision'] != 'fold':
            print(f"      ⚠️  INCOHERENT : main hors range mais decision={t['decision']}")

    db.close()
    return row.vpip, true_pct, profile.archetype, archetype


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.ERROR)

    results = []
    for arch, mut in [('TAG', 0), ('LAG', 1), ('CALLING_STATION', 0)]:
        measured, true, classified, expected = run_diagnostic(arch, mut, n_hands=200)
        results.append((f"{arch}_{mut}", measured, true, classified, expected))

    print(f"\n\n{'='*70}")
    print("  RESUME")
    print(f"{'='*70}")
    for name, measured, true, classified, expected in results:
        vpip_flag = "OK" if abs(measured - true) < 0.08 else "ECART SUSPECT"
        arch_flag = "OK" if classified == expected else "MAL CLASSE"
        print(f"  {name:20s} configure={true:.1%}  mesure={measured:.1%}  [{vpip_flag}]  "
              f"archétype={classified:16s} (attendu {expected:16s}) [{arch_flag}]")
