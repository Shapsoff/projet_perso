"""
validate_phase6.py — Validation Player DB (Phase 6)
Bot Poker Académique

Réécriture (débogage phase 6, suite) — remplace le verdict binaire "label
d'archétype exact à N mains fixes" par un tableau de bord plus large :

  1. Cohérence de config : mix d'actions (fold/check/call/raise/allin) de
     NOTRE bot ET du villain, comparé à ce que la config du RangeBot laisse
     attendre (VPIP/PFR mesurés vs configurés).
  2. Adaptabilité : notre bot varie-t-il sa taille de mise selon la force de
     sa main (EHS) et selon l'archétype adverse cru, ou joue-t-il toujours
     pareil ? Rapporté en % pot moyen par palier d'EHS et par archétype cru.
  3. Bluffs : quelles mains ont été bluffées (BluffLayer.should_bluff=True),
     et l'adversaire a-t-il fold juste après (mesure grossière d'efficacité).
  4. --quick (1 seed, rapide, smoke-test) VS mode complet multi-seed
     (agrège plusieurs seeds : moyenne + écart-type du BB/100, taux
     d'identification correcte sur l'ensemble des seeds plutôt qu'un
     verdict pass/fail sur un seul run).
  5. Persistance multi-session (inchangé depuis la version précédente).

Ce que ce script NE fait PAS (limite connue, cf. document de discussion
séparé) : comparer la range estimée au Palier 2 (fréquence par combo) à la
vraie range configurée du RangeBot. Nécessiterait d'exposer une API de
lecture combo-par-combo côté DBAwareRangeEstimator, pas encore disponible.
Le classement en archétype (Palier 1) reste suivi, mais comme diagnostic
secondaire, plus comme critère pass/fail.

Usage :
    python tests/validate_phase6.py                     # complet, 5 seeds x 1000 mains
    python tests/validate_phase6.py --quick              # rapide, 1 seed x 200 mains
    python tests/validate_phase6.py --opponent TAG_0     # un seul adversaire
    python tests/validate_phase6.py --seeds 10 --n_hands 500
    python tests/validate_phase6.py --db_path data/validate_phase6.sqlite3
"""

from __future__ import annotations

import sys
import os
import time
import argparse
import logging
import tempfile
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.simulator import PokerTable
from core.bots.ehs_bot import EHSBot, EHSBotConfig
from core.bots.range_bot import make_all_range_bots
from core.bots.range_definitions import ARCHETYPES, get_range
from core.game_state import Action as SimAction, ActionType, Street

from core.player_db.player_db import PlayerDB
from core.player_db.hand_recorder import record_hand
from core.player_db.profile_builder import build_profile

try:
    import poker_engine
    _ehs_calc_for_recording = poker_engine.EHSCalculator(seed=42, use_openmp=False)
except ImportError:
    _ehs_calc_for_recording = None

logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)

# Paliers d'EHS utilisés pour le rapport d'adaptabilité (bornes reprises du
# fallback v3 -- fallback_ehs_low/high -- pour rester comparables au reste
# du projet plutôt que d'inventer un nouveau découpage).
_EHS_BUCKETS = [
    ("faible   (EHS<0.45)", 0.00, 0.45),
    ("moyenne  (0.45-0.70)", 0.45, 0.70),
    ("forte    (EHS>0.70)", 0.70, 1.01),
]

_BET_TYPES = ('bet', 'raise', 'allin')


# =============================================================================
# Adaptateur bot → BotFn, avec capture du board/position + trace de décision
# =============================================================================

_MAP = {
    'fold':  ActionType.FOLD,
    'check': ActionType.CHECK,
    'call':  ActionType.CALL,
    'bet':   ActionType.RAISE,
    'raise': ActionType.RAISE,
    'allin': ActionType.ALLIN,
}


def _make_bot_fn(bot, player_id: int, capture: Optional[dict] = None,
                  trace: Optional[list] = None):
    """
    Identique à la version précédente + `trace` optionnel : si fourni, une
    entrée est ajoutée à CHAQUE décision de ce bot avec les diagnostics
    internes (EHS, archétype cru, bluff ou non) quand ils sont disponibles
    (attributs last_ehs/last_decision posés par EHSBot.decide() — cf.
    ehs_bot.py ; absents pour un RangeBot, qui n'a pas ces attributs et pour
    qui l'entrée se limite alors à l'action/montant bruts).
    """

    def bot_fn(state) -> SimAction:
        try:
            our = state.get_player(player_id)
        except KeyError:
            our = state.our_player

        stack      = float(our.stack)
        pos        = our.position
        street_raw = state.street
        hand       = [str(c) for c in our.hole_cards] if our.hole_cards else []
        pot_before = float(state.pot)

        players_dict = [
            {
                'id': p.player_id, 'player_id': p.player_id,
                'stack': float(p.stack), 'status': p.status.value,
                'position': p.position.value,
            }
            for p in state.players if p.player_id != player_id
        ]

        action_history = [
            {'street': act.street.value, 'player': act.player_id,
             'action': act.action_type.value, 'amount': float(act.amount)}
            for act in state.action_history
        ]

        gs_dict = {
            'hand': hand, 'board': state.board_str(), 'pot': float(state.pot),
            'to_call': float(state.to_call), 'stack': stack, 'position': pos.value,
            'street': street_raw.value, 'players': players_dict,
            'player_id': player_id, 'action_history': action_history,
            'big_blind': float(state.big_blind),
        }

        if capture is not None:
            capture['board'] = state.board_str()
            other = next((p for p in state.players if p.player_id != player_id), None)
            if other is not None:
                capture['opponent_position'] = other.position.value

        internal_action = bot.decide(gs_dict)
        at     = _MAP.get(internal_action.action_type, ActionType.CHECK)
        amount = int(getattr(internal_action, 'amount', 0))

        if trace is not None:
            decision = getattr(bot, 'last_decision', None)
            entry = {
                'street': street_raw.value,
                'action_type': internal_action.action_type,
                'amount': float(amount),
                'pot_before': pot_before,
                'ehs': getattr(bot, 'last_ehs', None),
                'sizing_pct': (decision.best_action.sizing_pct
                               if decision is not None else
                               (amount / pot_before if pot_before > 0 else 0.0)),
                'archetype_believed': decision.archetype if decision is not None else None,
                'confidence_believed': decision.confidence if decision is not None else None,
                'used_fallback': decision.used_fallback if decision is not None else None,
                'is_bluff': decision.bluff_added if decision is not None else False,
            }
            trace.append(entry)

        return SimAction(player_id, at, amount, street_raw)

    return bot_fn


def _villain_action_mix(actions, villain_id: int) -> Tuple[Counter, Dict[str, Counter]]:
    """
    Mix d'actions POSTFLOP du villain (global + par rue), en excluant
    POST_BLIND et le préflop -- le préflop est une décision raise-or-fold
    de nature différente (largeur de range, déjà suivie via VPIP), le
    mélanger au mix postflop fausserait la lecture ("fold" y dominerait
    simplement parce que peu de mains sont jouées, pas parce que le
    villain est passif postflop).
    """
    overall = Counter()
    by_street: Dict[str, Counter] = defaultdict(Counter)
    for a in actions:
        if (a.player_id != villain_id or a.action_type == ActionType.POST_BLIND
                or a.street == Street.PREFLOP):
            continue
        overall[a.action_type.value] += 1
        by_street[a.street.value][a.action_type.value] += 1
    return overall, by_street


def _match_bluffs_to_outcomes(actions, our_id: int, our_trace: List[dict]) -> None:
    """
    Aligne `our_trace` (une entrée par décision de notre bot, dans l'ordre)
    avec `actions` (le log complet de la main, chronologique) pour ajouter,
    sur les entrées bluffées, si le villain a foldé juste après. Alignement
    par position : chaque action de `our_id` dans `actions` correspond, dans
    l'ordre, à une entrée de `our_trace` (une décision produit exactement
    une action loggée) -- mute `our_trace` en place.
    """
    our_positions = [i for i, a in enumerate(actions) if a.player_id == our_id]
    if len(our_positions) != len(our_trace):
        # Décalage inattendu (ex. main interrompue) -- on abandonne
        # l'alignement plutôt que de risquer une correspondance fausse.
        return
    for trace_entry, pos in zip(our_trace, our_positions):
        if not trace_entry.get('is_bluff'):
            continue
        villain_folded = False
        for a in actions[pos + 1:]:
            if a.player_id == our_id:
                break
            if a.action_type == ActionType.POST_BLIND:
                continue
            villain_folded = (a.action_type == ActionType.FOLD)
            break
        trace_entry['villain_folded_after'] = villain_folded


# =============================================================================
# Diagnostics d'un match
# =============================================================================

@dataclass
class MatchDiagnostics:
    opponent:              str
    seed:                  int
    n_hands:               int
    profit_chips:          float
    bb_100:                float
    duration_s:            float
    true_archetype:        str
    tier_after:             int
    archetype_after:        Optional[str]
    hands_seen_after:       int
    confidence_after:       float
    correctly_identified:   bool
    our_action_mix:         Counter          # postflop uniquement
    villain_action_mix:     Counter          # postflop uniquement
    our_preflop_mix:        Counter
    villain_preflop_mix:    Counter
    villain_measured_vpip:  float
    villain_configured_vpip: float
    our_trace:               List[dict] = field(default_factory=list)
    hand_net_results:        List[float] = field(default_factory=list)


def run_match_instrumented(
    bot_v4,
    opp,
    opp_name:      str,
    true_archetype: str,
    n_hands:       int,
    db:            PlayerDB,
    starting_stack: int = 1000,
    small_blind:    int = 5,
    big_blind:      int = 10,
    seed:           int = 42,
    rebuy_threshold: float = 0.70,
) -> MatchDiagnostics:
    """Comme run_match_with_recording(), + capture des diagnostics riches."""
    if hasattr(bot_v4, 'full_reset'):
        bot_v4.full_reset(opponent_id=opp_name)
    if hasattr(opp, 'reset'):
        opp.reset()

    capture: dict = {}
    match_trace: List[dict] = []
    bot_fn = _make_bot_fn(bot_v4, player_id=0, capture=capture, trace=match_trace)
    opp_fn = _make_bot_fn(opp, player_id=1)

    table = PokerTable(n_players=2, starting_stacks=starting_stack,
                        small_blind=small_blind, big_blind=big_blind, seed=seed)

    t0 = time.time()
    hands_played = 0
    profit_chips = 0.0
    our_action_mix: Counter = Counter()      # postflop uniquement
    our_preflop_mix: Counter = Counter()
    villain_action_mix: Counter = Counter()  # postflop uniquement
    villain_preflop_mix: Counter = Counter()
    villain_vpip_hands = 0
    villain_preflop_opportunities = 0
    hand_net_results: List[float] = []

    for hand_idx in range(n_hands):
        for pid in [0, 1]:
            if table.stacks[pid] < starting_stack * rebuy_threshold:
                table.stacks[pid] = starting_stack

        hand_trace_start = len(match_trace)
        try:
            result = table.run_hand(bots={0: bot_fn, 1: opp_fn})
            net = float(result.gains.get(0, 0))
            profit_chips += net
            hand_net_results.append(net)
            hands_played += 1
        except Exception as e:
            logger.warning("Main %d échouée : %s", hand_idx, e)
            continue

        hand_trace = match_trace[hand_trace_start:]
        _match_bluffs_to_outcomes(result.actions, our_id=0, our_trace=hand_trace)

        for entry in hand_trace:
            if entry['street'] == 'preflop':
                our_preflop_mix[entry['action_type']] += 1
            else:
                our_action_mix[entry['action_type']] += 1

        villain_preflop_mix.update(
            Counter(a.action_type.value for a in result.actions
                    if a.player_id == 1 and a.street == Street.PREFLOP
                    and a.action_type != ActionType.POST_BLIND)
        )

        v_mix, _ = _villain_action_mix(result.actions, villain_id=1)
        villain_action_mix.update(v_mix)

        villain_preflop = [a for a in result.actions
                            if a.player_id == 1 and a.street == Street.PREFLOP
                            and a.action_type != ActionType.POST_BLIND]
        if villain_preflop:
            villain_preflop_opportunities += 1
            if any(a.action_type in (ActionType.CALL, ActionType.RAISE, ActionType.ALLIN)
                   for a in villain_preflop):
                villain_vpip_hands += 1

        try:
            record_hand(
                db=db, hand_result=result, board=capture.get('board', []),
                player_id_map={1: opp_name},
                position_map={1: capture.get('opponent_position')},
                our_player_id=0, ehs_calculator=_ehs_calc_for_recording, n_sims=200,
            )
        except Exception as e:
            logger.warning("Enregistrement DB échoué (main %d) : %s", hand_idx, e)

        if hasattr(bot_v4, 'reset'): bot_v4.reset()
        if hasattr(opp, 'reset'):    opp.reset()

    duration = time.time() - t0
    bb_100 = (profit_chips / big_blind) / max(hands_played, 1) * 100

    profile_after = build_profile(db, opp_name)
    archetype_after = profile_after.archetype
    correctly_identified = (
        profile_after.tier >= 1 and archetype_after == true_archetype
    )

    true_mutation = int(opp_name.rsplit('_', 1)[1])
    true_configured_vpip = get_range(true_archetype, true_mutation).range_pct
    villain_measured_vpip = (villain_vpip_hands / villain_preflop_opportunities
                              if villain_preflop_opportunities else 0.0)

    return MatchDiagnostics(
        opponent=opp_name, seed=seed, n_hands=hands_played,
        profit_chips=profit_chips, bb_100=bb_100, duration_s=duration,
        true_archetype=true_archetype, tier_after=profile_after.tier,
        archetype_after=archetype_after, hands_seen_after=profile_after.hands_seen,
        confidence_after=profile_after.archetype_confidence,
        correctly_identified=correctly_identified,
        our_action_mix=our_action_mix, villain_action_mix=villain_action_mix,
        our_preflop_mix=our_preflop_mix, villain_preflop_mix=villain_preflop_mix,
        villain_measured_vpip=villain_measured_vpip,
        villain_configured_vpip=true_configured_vpip,
        our_trace=match_trace, hand_net_results=hand_net_results,
    )


# =============================================================================
# Agrégation multi-seed
# =============================================================================

@dataclass
class OpponentAggregate:
    opponent:               str
    true_archetype:         str
    n_seeds:                int
    total_hands:            int
    bb_100_mean:            float
    bb_100_stdev:           float
    bb_100_ci95:            float          # demi-largeur ; NaN si n_seeds < 2
    id_rate:                float          # fraction des seeds correctement identifiées
    last_archetype_seen:    Optional[str]  # à titre indicatif (peut varier entre seeds)
    our_action_mix:         Counter
    villain_action_mix:     Counter
    our_preflop_mix:        Counter
    villain_preflop_mix:    Counter
    villain_measured_vpip:  float
    villain_configured_vpip: float
    all_traces:             List[dict] = field(default_factory=list)


def aggregate_seeds(diags: List[MatchDiagnostics]) -> OpponentAggregate:
    """Combine plusieurs MatchDiagnostics (même adversaire, seeds différentes)."""
    assert diags, "aggregate_seeds() appelé sans aucun résultat"
    opp = diags[0].opponent
    true_arch = diags[0].true_archetype

    bb100_values = [d.bb_100 for d in diags]
    mean_bb100 = statistics.mean(bb100_values)
    stdev_bb100 = statistics.stdev(bb100_values) if len(bb100_values) > 1 else 0.0
    # Approximation normale (pas de scipy dispo) -- indicative seulement,
    # surtout grossière avec peu de seeds (<10). Toujours mieux qu'un seul
    # point sans aucune mesure de dispersion.
    ci95 = (1.96 * stdev_bb100 / (len(bb100_values) ** 0.5)
            if len(bb100_values) > 1 else float('nan'))

    id_rate = statistics.mean([1.0 if d.correctly_identified else 0.0 for d in diags])

    our_mix = Counter()
    villain_mix = Counter()
    our_preflop_mix = Counter()
    villain_preflop_mix = Counter()
    all_traces: List[dict] = []
    vpip_measured = []
    for d in diags:
        our_mix.update(d.our_action_mix)
        villain_mix.update(d.villain_action_mix)
        our_preflop_mix.update(d.our_preflop_mix)
        villain_preflop_mix.update(d.villain_preflop_mix)
        all_traces.extend(d.our_trace)
        vpip_measured.append(d.villain_measured_vpip)

    return OpponentAggregate(
        opponent=opp, true_archetype=true_arch, n_seeds=len(diags),
        total_hands=sum(d.n_hands for d in diags),
        bb_100_mean=mean_bb100, bb_100_stdev=stdev_bb100, bb_100_ci95=ci95,
        id_rate=id_rate, last_archetype_seen=diags[-1].archetype_after,
        our_action_mix=our_mix, villain_action_mix=villain_mix,
        our_preflop_mix=our_preflop_mix, villain_preflop_mix=villain_preflop_mix,
        villain_measured_vpip=statistics.mean(vpip_measured),
        villain_configured_vpip=diags[0].villain_configured_vpip,
        all_traces=all_traces,
    )


# =============================================================================
# Rapports
# =============================================================================

def _pct_table(mix: Counter, label: str, indent: str = "    ") -> str:
    total = sum(mix.values())
    if total == 0:
        return f"{indent}{label} : (aucune action enregistrée)"
    order = ['fold', 'check', 'call', 'raise', 'bet', 'allin']
    parts = [f"{k}={mix[k]} ({100*mix[k]/total:.0f}%)"
             for k in order if mix.get(k, 0) > 0]
    parts += [f"{k}={v} ({100*v/total:.0f}%)" for k, v in mix.items() if k not in order]
    return f"{indent}{label} (n={total}) : " + ", ".join(parts)


def _sizing_by_ehs_bucket(traces: List[dict]) -> str:
    postflop = [t for t in traces if t['street'] != 'preflop']
    lines = []
    for label, lo, hi in _EHS_BUCKETS:
        samples = [t for t in postflop
                   if t['action_type'] in _BET_TYPES and t['ehs'] is not None
                   and lo <= t['ehs'] < hi]
        if not samples:
            lines.append(f"    EHS {label} : aucune mise observée")
            continue
        sizings = [s['sizing_pct'] for s in samples if s['sizing_pct']]
        avg_sizing = statistics.mean(sizings) if sizings else float('nan')
        lines.append(f"    EHS {label} : {len(samples)} mises, "
                     f"sizing moyen = {avg_sizing:.0%} du pot")
    return "\n".join(lines)


def _bluff_summary(traces: List[dict]) -> str:
    bluffs = [t for t in traces if t.get('is_bluff') and t['street'] != 'preflop']
    if not bluffs:
        return "    Aucun bluff détecté sur cet échantillon."
    fold_outcomes = [t['villain_folded_after'] for t in bluffs
                      if 'villain_folded_after' in t]
    avg_ehs = statistics.mean([t['ehs'] for t in bluffs if t['ehs'] is not None])
    lines = [f"    {len(bluffs)} bluffs détectés (EHS moyen = {avg_ehs:.2f}, "
             f"donc sous le seuil de value-bet)"]
    if fold_outcomes:
        fold_rate = statistics.mean([1.0 if f else 0.0 for f in fold_outcomes])
        lines.append(f"    Adversaire a foldé juste après dans "
                     f"{100*fold_rate:.0f}% des cas ({len(fold_outcomes)} bluffs suivis)")
    by_street = Counter(t['street'] for t in bluffs)
    lines.append("    Répartition par rue : " +
                 ", ".join(f"{k}={v}" for k, v in by_street.items()))
    return "\n".join(lines)


def print_opponent_report(agg: OpponentAggregate) -> None:
    print(f"\n{'─'*78}")
    print(f"  {agg.opponent}  (vrai archétype = {agg.true_archetype}, "
          f"{agg.n_seeds} seed(s), {agg.total_hands} mains au total)")
    print(f"{'─'*78}")

    if agg.n_seeds > 1:
        ci_str = f" ± {agg.bb_100_ci95:.1f} (IC95 approx.)" if agg.bb_100_stdev else ""
        print(f"  BB/100 : {agg.bb_100_mean:+.2f}{ci_str}  "
              f"(écart-type inter-seeds = {agg.bb_100_stdev:.1f})")
    else:
        print(f"  BB/100 : {agg.bb_100_mean:+.2f}  (1 seule seed -- "
              f"pas de mesure de dispersion, prudence sur ce chiffre)")

    print(f"  Archétype identifié correctement : {100*agg.id_rate:.0f}% des seeds "
          f"(dernier vu : {agg.last_archetype_seen})")

    print(f"\n  Cohérence de config (villain) :")
    print(f"    VPIP mesuré = {100*agg.villain_measured_vpip:.1f}%  "
          f"vs configuré = {100*agg.villain_configured_vpip:.1f}%")

    print(f"\n  Mix d'actions préflop (fold vs. entrée en jeu) :")
    print(_pct_table(agg.our_preflop_mix, "Notre bot"))
    print(_pct_table(agg.villain_preflop_mix, "Villain  "))

    print(f"\n  Mix d'actions postflop :")
    print(_pct_table(agg.our_action_mix, "Notre bot"))
    print(_pct_table(agg.villain_action_mix, "Villain  "))

    print(f"\n  Sizing par force de main (notre bot, mises seulement) :")
    print(_sizing_by_ehs_bucket(agg.all_traces))

    print(f"\n  Bluffs (notre bot) :")
    print(_bluff_summary(agg.all_traces))


def print_global_adaptiveness_report(aggs: List[OpponentAggregate]) -> None:
    """
    Vue d'ensemble tous adversaires confondus : est-ce que notre bot mise
    différemment selon l'archétype adverse CRU au moment de la décision
    (pas le vrai archétype -- ce qu'il croyait alors) ? Un bot qui
    maximise vraiment devrait miser plus gros / plus souvent contre un
    profil cru passif que contre un profil cru serré-agressif.
    """
    print(f"\n{'='*78}")
    print("  ADAPTABILITÉ GLOBALE -- sizing/agression selon l'archétype CRU")
    print(f"{'='*78}")

    all_traces = [t for agg in aggs for t in agg.all_traces
                  if t['street'] != 'preflop']
    by_belief: Dict[str, List[dict]] = defaultdict(list)
    for t in all_traces:
        key = t['archetype_believed'] or ('fallback_v3' if t['used_fallback'] else 'inconnu')
        by_belief[key].append(t)

    if not all_traces:
        print("  (aucune trace postflop collectée)")
        return

    print("  (périmètre : décisions POSTFLOP uniquement -- le sizing préflop suit")
    print("   des conventions différentes, en multiples de BB, pas en % pot)")

    n_fallback = sum(1 for t in all_traces if t['used_fallback'])
    print(f"  Décisions en repli v3 (pas de Best-Response) : "
          f"{n_fallback}/{len(all_traces)} ({100*n_fallback/len(all_traces):.0f}%)")

    print(f"\n  {'Archétype cru':<20} {'n décisions':<13} {'% agressif':<12} "
          f"{'sizing moyen (mises)':<22}")
    for belief, samples in sorted(by_belief.items(), key=lambda kv: -len(kv[1])):
        n = len(samples)
        n_agg = sum(1 for s in samples if s['action_type'] in _BET_TYPES)
        sizings = [s['sizing_pct'] for s in samples
                   if s['action_type'] in _BET_TYPES and s['sizing_pct']]
        avg_sizing = f"{statistics.mean(sizings):.0%}" if sizings else "n/a"
        print(f"  {str(belief):<20} {n:<13} {100*n_agg/n:<11.0f}% {avg_sizing:<22}")

    print("\n  Si les lignes ci-dessus se ressemblent toutes (% agressif et sizing")
    print("  similaires peu importe l'archétype cru), le bot ne s'adapte probablement")
    print("  pas vraiment au profil adverse malgré la classification -- signal utile")
    print("  indépendamment de la précision de la classification elle-même.")


# =============================================================================
# Validation 1 — Convergence contre les RangeBots
# =============================================================================

def validate_convergence(
    db_path: str,
    n_hands: int,
    n_seeds: int = 1,
    filter_opponents: Optional[List[str]] = None,
    base_seed: int = 42,
) -> List[OpponentAggregate]:
    """
    Joue n_hands mains contre chaque RangeBot, n_seeds fois (essais
    indépendants), et agrège.

    n_seeds == 1 (mode --quick) : réutilise `db_path` tel quel -- DB
    partagée/persistante sur la session, comme avant (cf. validate_persistence
    plus bas, qui compte explicitement sur cette accumulation).

    n_seeds > 1 : chaque seed utilise sa PROPRE DB temporaire fraîche, pour
    garantir des essais réellement indépendants -- sinon les mains d'une
    seed influencent le classement de la suivante et l'agrégation perd son
    sens (on mesurerait la convergence d'UNE SEULE longue session, pas la
    variance entre plusieurs essais indépendants).
    """
    from core.bots.range_bot import make_range_bot

    opponents_all = make_all_range_bots(n_sims=500)
    opponent_names = (
        [n for n in opponents_all if n in filter_opponents]
        if filter_opponents else list(opponents_all.keys())
    )

    aggregates: List[OpponentAggregate] = []
    n_total = len(opponent_names)
    for i, name in enumerate(opponent_names, 1):
        true_arch, true_mut = name.rsplit('_', 1)[0], int(name.rsplit('_', 1)[1])
        print(f"\n  [{i:2d}/{n_total}] vs {name}")

        diags: List[MatchDiagnostics] = []
        for s in range(n_seeds):
            seed = base_seed + s
            this_db_path = (db_path if n_seeds == 1 else
                             tempfile.mktemp(suffix=f'_p6_{name}_s{seed}.sqlite3'))

            opp = make_range_bot(true_arch, true_mut, n_sims=500)
            cfg = EHSBotConfig(use_best_response=True, use_range_estimator=True,
                                use_player_db=True, player_db_path=this_db_path)
            bot_v4 = EHSBot(config=cfg)
            db = PlayerDB(this_db_path)

            diag = run_match_instrumented(
                bot_v4=bot_v4, opp=opp, opp_name=name, true_archetype=true_arch,
                n_hands=n_hands, db=db, seed=seed,
            )
            db.close()
            bot_v4.close()
            diags.append(diag)

            marker = '✅' if diag.correctly_identified else '⚠ '
            print(f"       seed={seed:<4d} BB/100={diag.bb_100:+7.2f}  "
                  f"archétype={diag.archetype_after or '?':16s} {marker}")

        aggregates.append(aggregate_seeds(diags))

    return aggregates


# =============================================================================
# Validation 2 — Persistance multi-session
# =============================================================================

def validate_persistence(db_path: str, n_hands: int) -> bool:
    """
    Joue n_hands mains contre un TAG, ferme la DB, la rouvre avec une
    NOUVELLE instance PlayerDB/EHSBot (simule un nouveau process/une
    nouvelle session), et vérifie que le profil a bien survécu.
    """
    print("\n── PERSISTANCE MULTI-SESSION ──")
    from core.bots.range_bot import make_range_bot
    opp_name = "TAG_0"
    opp = make_range_bot("TAG", 0, n_sims=500)

    db_probe = PlayerDB(db_path)
    profile_start = build_profile(db_probe, opp_name)
    db_probe.close()
    if profile_start.hands_seen > 0:
        print(f"  (DB déjà à {profile_start.hands_seen} mains pour {opp_name} "
              f"avant cette session — accumulation, pas un départ à zéro)")

    print(f"  Session 1 : {n_hands} mains vs {opp_name}...", end=' ', flush=True)
    cfg = EHSBotConfig(use_best_response=True, use_range_estimator=True,
                        use_player_db=True, player_db_path=db_path)
    bot1 = EHSBot(config=cfg)
    db1  = PlayerDB(db_path)

    diag1 = run_match_instrumented(
        bot_v4=bot1, opp=opp, opp_name=opp_name, true_archetype="TAG",
        n_hands=n_hands, db=db1, seed=42,
    )
    print(f"tier={diag1.tier_after} archétype={diag1.archetype_after} "
          f"hands_seen={diag1.hands_seen_after}")
    db1.close()
    bot1.close()

    print("  Fermeture DB, nouvelle connexion (simule un nouveau process)...")
    db2 = PlayerDB(db_path)
    profile2 = build_profile(db2, opp_name)
    print(f"  Session 2 (relecture seule) : tier={profile2.tier} "
          f"archétype={profile2.archetype} hands_seen={profile2.hands_seen}")

    ok = (profile2.hands_seen == diag1.hands_seen_after
          and profile2.tier == diag1.tier_after
          and profile2.archetype == diag1.archetype_after)
    print(f"  {'✅ Profil survit à la réouverture de la DB' if ok else '⚠ Profil perdu ou altéré'}")
    db2.close()
    return ok


# =============================================================================
# Point d'entrée
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Validation Phase 6 — Player DB")
    parser.add_argument('--quick', action='store_true',
                         help="Mode rapide : 1 seed, 200 mains/match, smoke-test")
    parser.add_argument('--opponent', type=str, default=None,
                         help="Tester contre un seul adversaire (ex: TAG_0)")
    parser.add_argument('--n_hands', type=int, default=None,
                         help="Nombre de mains par match (défaut : 200 en --quick, 1000 sinon)")
    parser.add_argument('--seeds', type=int, default=None,
                         help="Nombre d'essais indépendants par adversaire à agréger "
                              "(défaut : 1 en --quick, 5 sinon)")
    parser.add_argument('--db_path', type=str, default=None,
                         help="Chemin de la Player DB (défaut : fichier temporaire jetable)")
    parser.add_argument('--no_persistence', action='store_true',
                         help="Désactiver le test de persistance multi-session")
    parser.add_argument('--no_adaptiveness', action='store_true',
                         help="Désactiver le rapport d'adaptabilité globale")
    args = parser.parse_args()

    n_hands = args.n_hands or (200 if args.quick else 1000)
    n_seeds = args.seeds or (1 if args.quick else 5)
    db_path = args.db_path or tempfile.mktemp(suffix='_phase6.sqlite3')

    print(f"\n{'═' * 78}")
    print("  VALIDATION PHASE 6 — Player DB")
    print(f"  DB     : {db_path}")
    print(f"  Mode   : {'rapide' if args.quick else 'complet'} "
          f"({n_seeds} seed(s) x {n_hands} mains/match)")
    print(f"  EHS calculator pour showdowns : "
          f"{'poker_engine (réel)' if _ehs_calc_for_recording else 'indisponible — VPIP/PFR seuls'}")
    print(f"{'═' * 78}")

    print("\n── CONVERGENCE VS RANGEBOTS ──")
    filter_opp = [args.opponent] if args.opponent else None
    aggregates = validate_convergence(db_path, n_hands, n_seeds=n_seeds,
                                       filter_opponents=filter_opp)

    for agg in aggregates:
        print_opponent_report(agg)

    if not args.no_adaptiveness:
        print_global_adaptiveness_report(aggregates)

    print(f"\n{'═' * 78}")
    print("  RÉSUMÉ")
    print(f"{'═' * 78}")
    mean_id_rate = statistics.mean([a.id_rate for a in aggregates])
    print(f"  Taux moyen d'identification correcte : {100*mean_id_rate:.0f}% "
          f"(diagnostic, plus un critère pass/fail à lui seul)")
    for a in aggregates:
        marker = '✅' if a.id_rate == 1.0 else ('⚠ ' if a.id_rate > 0 else '❌')
        print(f"  {marker} {a.opponent:<20s} BB/100={a.bb_100_mean:+7.2f}  "
              f"identifié {100*a.id_rate:.0f}% des seeds")

    persistence_ok = True
    if not args.no_persistence:
        persistence_ok = validate_persistence(db_path, min(n_hands, 200))

    print(f"\n{'═' * 78}")
    print(f"  {'✅ Persistance OK' if persistence_ok else '⚠  Persistance en échec'}")
    print("  Rappel : ce script ne rend plus un verdict global pass/fail unique --")
    print("  la classification d'archétype est un diagnostic parmi d'autres, pas un")
    print("  critère de réussite isolé. Juge l'ensemble (BB/100, cohérence de config,")
    print("  adaptabilité, bluffs) plutôt qu'un seul chiffre.")
    print(f"{'═' * 78}\n")
    sys.exit(0 if persistence_ok else 1)


if __name__ == '__main__':
    main()
