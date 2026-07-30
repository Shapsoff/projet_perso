# debug_hand.py — à mettre dans tests/
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.simulator import PokerTable
from core.bots.range_bot import make_range_bot
from core.bots.ehs_bot import EHSBot, EHSBotConfig
from tests.validate_phase5 import _make_bot_fn

cfg   = EHSBotConfig(use_best_response=True, use_range_estimator=True)
bot   = EHSBot(config=cfg)
opp   = make_range_bot('LAG', mutation=0, n_sims=500)

table = PokerTable(n_players=2, starting_stacks=1000, big_blind=10, seed=42)
result = table.run_hand(bots={0: _make_bot_fn(bot), 1: _make_bot_fn(opp)})

print("Gains:", result.gains)
print("Actions:", [str(a) for a in result.actions])
print("Stacks restants:", table.stacks)