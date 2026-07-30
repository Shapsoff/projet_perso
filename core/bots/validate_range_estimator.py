"""
validate_range_estimator.py — Validation du Range Estimator (Phase 4 — P2 + P2.5)
Bot Poker Académique

Valide le Range Estimator bayésien contre les 9 RangeBots (vérité terrain connue).

Métriques calculées :
  1. Précision d'archétype   — % sessions où l'archétype est correctement identifié
  2. Erreur de range (L1)    — sum |P_estimée - P_réelle| sur tous les combos
  3. Convergence             — nombre de mains pour L1 < 0.10
  4. Calibration (Brier)     — qualité de la confiance (pas juste la précision)
  5. Dérive en exploitation  — mesure P2.5 : dérive sous exploitation active

Usage:
    python validate_range_estimator.py --n-hands 100 --n-sessions 5 --verbose
    python validate_range_estimator.py --quick   # 20 mains, 3 sessions, rapide

    # Résultats affichés en console + sauvegardés dans reports/p4_validation_*.json
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from core.range_estimator import RangeEstimator, ALL_COMBOS
from core.bots.range_definitions import (
    get_range, get_all_configs, ARCHETYPES, MUTATIONS, ArchetypeConfig
)
from core.bots.range_bot import make_range_bot, make_all_range_bots, RangeBot
from core.action_history import ActionBucket, classify_history
from core.metrics_range import RangeMetrics, SessionReport, GlobalReport


# =============================================================================
# Générateur d'observations synthétiques (sans simulateur complet)
# =============================================================================

def simulate_rangebot_actions(
    range_bot:  RangeBot,
    n_hands:    int = 100,
    seed:       int = 42,
) -> List[Dict]:
    """
    Génère des observations synthétiques depuis un RangeBot.

    Simule le comportement COMPLET d'un RangeBot, incluant :
      - Preflop : raise si main dans la range, fold sinon (les deux cas)
      - Postflop : actions sur flop, turn, river selon EHS et profil
        (uniquement si la main est dans la range)

    Point clé : les folds preflop (main hors range) sont TOUJOURS inclus.
    C'est le signal le plus discriminant pour distinguer TAG (fold 85%)
    de LAG (fold 65%) de Calling Station (fold 60%).

    Sans les folds, le modèle ne voit que les mains jouées → il ne peut
    pas estimer la tightness de la range → TAG confondu avec LAG.

    Args:
        range_bot : RangeBot à simuler
        n_hands   : nombre de mains à générer
        seed      : graine aléatoire

    Returns:
        Liste de dicts {'action_history': [...], 'game_state': {...}}
    """
    import random
    rng         = random.Random(seed)
    cfg         = range_bot.config
    combos_list = list(cfg.preflop_combos)
    all_combos  = ALL_COMBOS
    out_combos  = [c for c in all_combos if c not in cfg.preflop_combos]

    observations = []

    for hand_idx in range(n_hands):
        # Tirer une main selon la range_pct réelle du bot
        in_range = rng.random() < cfg.range_pct

        if in_range and combos_list:
            combo = rng.choice(combos_list)
        else:
            combo    = rng.choice(out_combos) if out_combos else rng.choice(all_combos)
            in_range = False

        history = []
        pot     = float(rng.randint(20, 60))   # pot avant action preflop

        # ── Action preflop ────────────────────────────────────────────────────
        # Les folds hors-range sont TOUJOURS enregistrés — signal clé pour
        # estimer la tightness de la range (différence TAG vs LAG vs CS)
        if in_range:
            history.append({
                'street': 'preflop', 'player': 1,
                'action': 'raise', 'amount': 30
            })
            pot += 30
        else:
            history.append({
                'street': 'preflop', 'player': 1,
                'action': 'fold', 'amount': 0
            })

        # ── Actions postflop (seulement si dans la range) ─────────────────────
        if in_range:
            board = ['2c', '7d', 'Js']

            for street in ['flop', 'turn', 'river']:
                ehs_stub = rng.uniform(0.25, 0.90)
                to_call  = float(rng.randint(0, 1)) * pot * 0.5  # parfois face à une mise

                if cfg.aggression == 'aggressive':
                    if ehs_stub > cfg.postflop_ehs_threshold:
                        action = 'raise' if to_call > 0 else 'bet'
                    elif ehs_stub > cfg.postflop_ehs_call:
                        if to_call > 0:
                            # Vérifier pot odds
                            pot_odds = to_call / (pot + to_call)
                            action   = 'call' if ehs_stub > pot_odds else 'fold'
                        else:
                            action = 'check'
                    else:
                        action = 'fold' if to_call > 0 else 'check'
                else:  # passive / Calling Station
                    if ehs_stub > cfg.postflop_ehs_threshold and to_call == 0:
                        action = 'bet'
                    elif ehs_stub > cfg.postflop_ehs_call:
                        action = 'call' if to_call > 0 else 'check'
                    else:
                        action = 'fold' if to_call > 0 else 'check'

                history.append({
                    'street': street, 'player': 1,
                    'action': action,
                    'amount': int(pot * 0.5) if action in ('bet', 'raise') else
                              int(to_call)   if action == 'call' else 0
                })

                # Arrêter si fold
                if action == 'fold':
                    break
                pot += pot * 0.5 if action in ('bet', 'raise', 'call') else 0

        observations.append({
            'action_history': history,
            'game_state': {
                'hand':    ['Ah', 'Kd'],
                'board':   ['2c', '7d', 'Js'] if in_range else [],
                'pot':     pot,
                'to_call': 0.0,
                'street':  'flop' if in_range else 'preflop',
            },
            '_in_range': in_range,
            '_combo':    combo,
        })

    return observations


# =============================================================================
# Métriques de calibration (Brier score)
# =============================================================================

def brier_score(
    estimated_probs: Dict[str, float],
    true_combos:     set,
) -> float:
    """
    Calcule le Brier Score : mesure la qualité des probabilités estimées.

    Brier = (1/N) × sum_{combo} (P_estimée(combo) - P_réelle(combo))²

    Contrairement à L1 qui mesure l'erreur absolue, le Brier score pénalise
    les erreurs de calibration : un modèle qui dit "60%" alors que c'est
    toujours 0% ou 100% aura un mauvais Brier même avec un bon L1.

    ∈ [0, 1]. 0 = parfait, 1 = inverse parfait.
    Référence : prior uniforme → Brier ≈ 2 × range_pct × (1 - range_pct)
    """
    if not true_combos:
        return 1.0

    n       = len(ALL_COMBOS)
    p_true  = 1.0 / len(true_combos)

    total   = sum(estimated_probs.values())
    if total <= 0:
        normalized = {c: 1.0 / n for c in estimated_probs}
    else:
        normalized = {c: p / total for c, p in estimated_probs.items()}

    brier = 0.0
    for combo in ALL_COMBOS:
        p_est  = normalized.get(combo, 0.0)
        p_real = p_true if combo in true_combos else 0.0
        brier += (p_est - p_real) ** 2

    return brier / n


# =============================================================================
# Test P2.5 : dérive sous exploitation
# =============================================================================

def test_exploitation_drift(
    range_bot:  RangeBot,
    n_hands:    int  = 80,
    seed:       int  = 99,
    verbose:    bool = False,
) -> Dict:
    """
    P2.5 — Mesure la dérive du Range Estimator quand l'EHSBot exploite le RangeBot.

    Problème identifié en session 3 :
      Quand l'EHSBot apprend à exploiter le RangeBot, les actions observées
      sont biaisées : le RangeBot joue contre un bot qui adapte son sizing
      et ses seuils, ce qui modifie l'historique d'actions enregistré.
      Cela peut biaiser le Range Estimator.

    Test simplifié :
      - Phase 1 (mains 1-40)  : EHSBot joue normalement → estimateur converge
      - Phase 2 (mains 41-80) : EHSBot exploite (raise systématique face au
        Calling Station) → mesurer si la L1 dérive vs la vérité terrain

    Returns:
        Dict avec L1 phase 1, L1 phase 2, et dérive (delta_L1).
    """
    cfg        = range_bot.config
    estimator  = RangeEstimator()
    metrics    = RangeMetrics()
    half       = n_hands // 2

    # ── Phase 1 : observation normale ─────────────────────────────────────────
    obs_phase1 = simulate_rangebot_actions(range_bot, n_hands=half, seed=seed)
    for obs in obs_phase1:
        history = obs['action_history']
        for event in history:
            if event['player'] == 1:
                estimator.observe_action(
                    action_type=event['action'],
                    street=event['street'],
                    pot=obs['game_state']['pot'],
                    to_call=obs['game_state']['to_call'],
                )
        estimator.new_hand()

    dist_phase1 = estimator.get_distribution()
    l1_phase1   = metrics.l1_error(dist_phase1, cfg.preflop_combos)

    if verbose:
        arch, mut = estimator.get_best_archetype()
        print(f"    Phase 1 ({half} mains) : L1={l1_phase1:.4f} "
              f"arch={arch} mut={mut}")

    # ── Phase 2 : simulation d'exploitation ───────────────────────────────────
    # L'exploitation change les sizings adverses → le RangeBot voit des raises
    # plus gros, ce qui peut modifier légèrement ses actions (SPR, commitment)
    # Simulation simplifiée : on ajoute du bruit sur les actions postflop
    import random
    rng = random.Random(seed + 100)

    obs_phase2 = simulate_rangebot_actions(range_bot, n_hands=half, seed=seed + 1)
    for obs in obs_phase2:
        history = obs['action_history']
        # Simulation du biais d'exploitation : les actions fold postflop
        # sont plus fréquentes (le RangeBot se fait presser par un bot agressif)
        biased_history = []
        for event in history:
            if (event['player'] == 1 and event['street'] == 'flop'
                    and rng.random() < 0.20):  # 20% de chance d'action différente
                biased_event = dict(event)
                if event['action'] == 'bet':
                    biased_event['action'] = 'check'  # pression → check
                elif event['action'] == 'call':
                    biased_event['action'] = 'fold'   # pression → fold
                biased_history.append(biased_event)
            else:
                biased_history.append(event)

        for event in biased_history:
            if event['player'] == 1:
                estimator.observe_action(
                    action_type=event['action'],
                    street=event['street'],
                    pot=obs['game_state']['pot'],
                    to_call=obs['game_state']['to_call'],
                )
        estimator.new_hand()

    dist_phase2 = estimator.get_distribution()
    l1_phase2   = metrics.l1_error(dist_phase2, cfg.preflop_combos)

    delta_l1 = l1_phase2 - l1_phase1

    if verbose:
        arch, mut = estimator.get_best_archetype()
        print(f"    Phase 2 (exploitation, {half} mains) : L1={l1_phase2:.4f} "
              f"arch={arch} mut={mut}")
        print(f"    Dérive L1 : {delta_l1:+.4f} "
              f"({'⚠ dérive significative' if abs(delta_l1) > 0.05 else '✓ stable'})")

    return {
        'l1_phase1':  l1_phase1,
        'l1_phase2':  l1_phase2,
        'delta_l1':   delta_l1,
        'stable':     abs(delta_l1) <= 0.05,
    }


# =============================================================================
# Évaluateur principal
# =============================================================================

class Phase4Validator:
    """
    Valide le Range Estimator contre les 9 RangeBots.

    Étapes :
      P2   — Précision, L1, convergence, Brier
      P2.5 — Dérive sous exploitation
    """

    def __init__(self, n_hands: int = 100, n_sessions: int = 5):
        self.n_hands    = n_hands
        self.n_sessions = n_sessions
        self.metrics    = RangeMetrics()

    def validate_single_bot(
        self,
        range_bot: RangeBot,
        session:   int = 0,
        verbose:   bool = False,
    ) -> Dict:
        """
        Valide l'estimateur sur une session contre un RangeBot.

        Returns:
            Dict avec toutes les métriques de la session.
        """
        cfg         = range_bot.config
        estimator   = RangeEstimator()
        true_combos = cfg.preflop_combos

        l1_history        = []
        brier_history     = []
        arch_correct_list = []
        mut_correct_list  = []

        seed = session * 1000 + abs(hash(cfg.archetype + str(cfg.mutation))) % 10000

        for hand_idx in range(self.n_hands):
            obs_list = simulate_rangebot_actions(range_bot, n_hands=1,
                                                 seed=seed + hand_idx)
            obs = obs_list[0]

            for event in obs['action_history']:
                if event['player'] == 1:
                    estimator.observe_action(
                        action_type=event['action'],
                        street=event['street'],
                        pot=obs['game_state']['pot'],
                        to_call=obs['game_state']['to_call'],
                    )

            estimator.new_hand()

            # Métriques à cette main
            dist        = estimator.get_distribution()
            l1          = self.metrics.l1_error(dist, true_combos)
            brier       = brier_score(dist, true_combos)
            arch, mut   = estimator.get_best_archetype()

            l1_history.append((hand_idx + 1, l1))
            brier_history.append((hand_idx + 1, brier))
            arch_correct_list.append(arch == cfg.archetype)
            mut_correct_list.append(mut  == cfg.mutation)

            if verbose and (hand_idx + 1) % 20 == 0:
                print(f"      Main {hand_idx+1:3d} : L1={l1:.4f} "
                      f"arch={arch}(±{'✓' if arch == cfg.archetype else '✗'}) "
                      f"Brier={brier:.5f}")

        n = len(l1_history)
        return {
            'archetype':          cfg.archetype,
            'mutation':           cfg.mutation,
            'n_hands':            n,
            'archetype_accuracy': sum(arch_correct_list) / n if n else 0,
            'mutation_accuracy':  sum(mut_correct_list) / n if n else 0,
            'l1_final':           l1_history[-1][1] if l1_history else 2.0,
            'l1_at_30':           self.metrics.l1_at_hand(l1_history, 30),
            'l1_at_50':           self.metrics.l1_at_hand(l1_history, 50),
            'brier_final':        brier_history[-1][1] if brier_history else 1.0,
            'hands_to_converge':  self.metrics.hands_to_l1_threshold(l1_history),
            'l1_history':         l1_history,
        }

    def run_full_validation(
        self,
        verbose:    bool = True,
        test_drift: bool = True,
    ) -> Dict:
        """
        Lance la validation complète sur les 9 RangeBots × n_sessions.

        Returns:
            Dict avec tous les résultats, prêt pour export JSON.
        """
        all_bots = make_all_range_bots(n_sims=500)
        results  = {}
        start    = time.time()

        for key, range_bot in all_bots.items():
            cfg = range_bot.config

            if verbose:
                print(f"\n  [{key}] {cfg.description[:55]}...")

            # Moyenner sur n_sessions
            session_results = []
            for s in range(self.n_sessions):
                res = self.validate_single_bot(range_bot, session=s,
                                               verbose=(verbose and s == 0))
                session_results.append(res)

            # Agréger
            n = len(session_results)
            agg = {
                'archetype':          cfg.archetype,
                'mutation':           cfg.mutation,
                'n_sessions':         n,
                'n_hands_per_session': self.n_hands,
                'archetype_accuracy': sum(r['archetype_accuracy'] for r in session_results) / n,
                'mutation_accuracy':  sum(r['mutation_accuracy']  for r in session_results) / n,
                'l1_final':           sum(r['l1_final']           for r in session_results) / n,
                'l1_at_30':           sum(r['l1_at_30']           for r in session_results) / n,
                'l1_at_50':           sum(r['l1_at_50']           for r in session_results) / n,
                'brier_final':        sum(r['brier_final']        for r in session_results) / n,
                'hands_to_converge':  [r['hands_to_converge'] for r in session_results],
            }

            if verbose:
                self._print_agg(agg)

            # P2.5 — Test de dérive sous exploitation
            if test_drift:
                drift = test_exploitation_drift(range_bot, n_hands=80,
                                                verbose=verbose)
                agg['exploitation_drift'] = drift

            results[key] = agg

        elapsed = time.time() - start
        self._print_global_summary(results, elapsed, verbose)

        return {
            'results':        results,
            'elapsed_seconds': elapsed,
            'n_hands':        self.n_hands,
            'n_sessions':     self.n_sessions,
        }

    # =========================================================================
    # Affichage
    # =========================================================================

    @staticmethod
    def _print_agg(agg: Dict) -> None:
        conv_list = agg['hands_to_converge']
        conv_str  = (
            f"{sum(c for c in conv_list if c) / max(len([c for c in conv_list if c]), 1):.0f}m"
            if any(conv_list) else "—"
        )
        print(f"    arch_acc={agg['archetype_accuracy']:.0%} "
              f"mut_acc={agg['mutation_accuracy']:.0%} "
              f"L1={agg['l1_final']:.4f} "
              f"Brier={agg['brier_final']:.5f} "
              f"conv≈{conv_str}")
        if 'exploitation_drift' in agg:
            d = agg['exploitation_drift']
            icon = '✓' if d['stable'] else '⚠'
            print(f"    Drift P2.5: ΔL1={d['delta_l1']:+.4f} {icon}")

    @staticmethod
    def _print_global_summary(results: Dict, elapsed: float, verbose: bool) -> None:
        if not verbose:
            return

        vals = list(results.values())
        n    = len(vals)

        print("\n" + "=" * 65)
        print("RAPPORT GLOBAL — Phase 4 — Range Estimator vs 9 RangeBots")
        print("=" * 65)
        print(f"  Temps        : {elapsed:.1f}s")
        print(f"  Arch. acc.   : {sum(v['archetype_accuracy'] for v in vals)/n:.1%}  (cible > 80%)")
        print(f"  L1 final moy : {sum(v['l1_final'] for v in vals)/n:.4f}  (cible < 0.15)")
        print(f"  Brier moy    : {sum(v['brier_final'] for v in vals)/n:.5f}")
        conv_ok = sum(1 for v in vals if any(v['hands_to_converge']))
        print(f"  Convergences : {conv_ok}/{n}  (cible 9/9)")

        for archetype in ARCHETYPES:
            arch_vals = [v for v in vals if v['archetype'] == archetype]
            m = len(arch_vals)
            if not m:
                continue
            mean_l1   = sum(v['l1_final'] for v in arch_vals) / m
            mean_acc  = sum(v['archetype_accuracy'] for v in arch_vals) / m
            drift_ok  = sum(
                1 for v in arch_vals
                if v.get('exploitation_drift', {}).get('stable', True)
            )
            print(f"\n  [{archetype}]  L1={mean_l1:.4f}  arch_acc={mean_acc:.0%}  "
                  f"drift_stable={drift_ok}/{m}")
            for v in arch_vals:
                hconv = v['hands_to_converge']
                conv  = f"{sum(c for c in hconv if c)/max(len([c for c in hconv if c]),1):.0f}m" if any(hconv) else "—"
                print(f"    mut{v['mutation']}: L1={v['l1_final']:.4f} "
                      f"@30={v['l1_at_30']:.4f} "
                      f"@50={v['l1_at_50']:.4f} "
                      f"conv={conv}")
        print()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 — Validation du Range Estimator vs 9 RangeBots"
    )
    parser.add_argument('--n-hands',    type=int, default=100,
                        help='Mains par session (défaut: 100)')
    parser.add_argument('--n-sessions', type=int, default=5,
                        help='Sessions par RangeBot (défaut: 5)')
    parser.add_argument('--quick',      action='store_true',
                        help='Mode rapide : 20 mains, 3 sessions')
    parser.add_argument('--no-drift',   action='store_true',
                        help='Désactiver le test P2.5 (dérive sous exploitation)')
    parser.add_argument('--verbose',    action='store_true', default=True,
                        help='Afficher la progression')
    parser.add_argument('--output',     type=str, default=None,
                        help='Fichier JSON de sortie (défaut: reports/p4_validation_<ts>.json)')

    args = parser.parse_args()

    if args.quick:
        args.n_hands    = 20
        args.n_sessions = 3

    print(f"\n{'='*65}")
    print(f"Phase 4 — Validation Range Estimator")
    print(f"  Mains/session : {args.n_hands}  |  Sessions/bot : {args.n_sessions}")
    print(f"  Total mains   : {args.n_hands * args.n_sessions * 9:,}")
    print(f"{'='*65}\n")

    validator = Phase4Validator(n_hands=args.n_hands, n_sessions=args.n_sessions)
    report    = validator.run_full_validation(
        verbose=args.verbose,
        test_drift=not args.no_drift,
    )

    # Export JSON
    import os
    ts      = int(time.time())
    outfile = args.output or f"reports/p4_validation_{ts}.json"
    os.makedirs(os.path.dirname(outfile) if os.path.dirname(outfile) else '.', exist_ok=True)

    # Nettoyer les types non-sérialisables
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        return str(obj)

    with open(outfile, 'w') as f:
        json.dump(clean(report), f, indent=2, ensure_ascii=False)

    print(f"\n  Rapport sauvegardé : {outfile}")


if __name__ == '__main__':
    main()
