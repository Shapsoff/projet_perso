import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import poker_engine
from core.best_response.frequency_model import FrequencyModel
from core.bots.range_definitions import get_range

dist = get_range('TAG', 0)
distribution = {c: 1/len(dist.preflop_combos) for c in dist.preflop_combos}

ehs_calc = poker_engine.EHSCalculator(seed=42, use_openmp=False)
freq_model = FrequencyModel(ehs_calculator=ehs_calc, n_sims=500)

board   = ['7c', '2h', 'Js']
sizings = [0.33, 0.50, 0.75, 1.00, 2.00]

t0 = time.time()
results = freq_model.compute_multiple_sizings(
    distribution=distribution,
    archetype='TAG',
    board=board,
    sizings=sizings,
    pot=100,
)
t1 = time.time()

print(f"compute_multiple_sizings : {(t1-t0)*1000:.1f} ms")
print(f"Combos EHS calculés : {sum(r.n_combos_ehs for r in results.values())}")
print()
for sizing, result in results.items():
    print(f"sizing={sizing:.0%} | fold={result.p_fold:.1%}  call={result.p_call:.1%}  raise={result.p_raise:.1%}")
