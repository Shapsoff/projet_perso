"""
diag_combined.py — Diagnostic ponctuel (pas un livrable permanent)

A lancer depuis la racine du projet :
    python diag_combined.py "C:\\Users\\Shaps\\AppData\\Local\\Temp\\tmpxjn8fb3d_phase6.sqlite3"

Fait deux choses :
1. Dump les stats brutes VPIP/PFR/AF de chaque adversaire en DB (révèle si
   elles sont à zéro pour tout le monde -> confirme l'hypothèse).
2. Rejoue UNE main réelle via PokerTable et inspecte le type exact de
   `action.street` sur chaque Action retournée -> identifie précisément
   si la comparaison avec Street.PREFLOP/FLOP/TURN/RIVER échoue.
"""
import sys
sys.path.insert(0, '.')

print("=" * 70)
print("1. STATS BRUTES EN DB")
print("=" * 70)
from core.player_db.player_db import PlayerDB

db_path = sys.argv[1] if len(sys.argv) > 1 else "data/player_db.sqlite3"
db = PlayerDB(db_path)
print(f"{'player_id':22s} {'hands':>6s} {'vpip':>7s} {'pfr':>7s} {'af':>7s} {'agg':>5s} {'pass':>5s}")
for pid in db.list_players():
    row = db.get_player_row(pid)
    print(f"{pid:22s} {row.hands_seen:6d} {row.vpip:7.1%} {row.pfr:7.1%} {row.af:7.2f} "
          f"{row.aggressive_acts:5d} {row.passive_acts:5d}")
db.close()

print()
print("=" * 70)
print("2. TYPE REEL DE action.street SUR UNE VRAIE MAIN POKERTABLE")
print("=" * 70)
from core.simulator import PokerTable
from core.game_state import Street, ActionType, Action as SimAction

def dumb_bot_fn(player_id):
    def fn(state):
        return SimAction(player_id, ActionType.CALL, 0, state.street)
    return fn

table = PokerTable(n_players=2, starting_stacks=1000, big_blind=10, seed=1)
result = table.run_hand(bots={0: dumb_bot_fn(0), 1: dumb_bot_fn(1)})

print(f"Street.PREFLOP (attendu) : type={type(Street.PREFLOP)} valeur={Street.PREFLOP!r}")
print()
print("Actions de la main jouee :")
for a in result.actions:
    print(f"  player={a.player_id} action={a.action_type} street={a.street!r} "
          f"type(street)={type(a.street)} "
          f"street==Street.PREFLOP -> {a.street == Street.PREFLOP}  "
          f"street in POSTFLOP -> {a.street in (Street.FLOP, Street.TURN, Street.RIVER)}")
