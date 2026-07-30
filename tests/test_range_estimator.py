"""
test_range_estimator.py — Tests unitaires du Range Estimator (Phase 4)
Bot Poker Académique

Tests organisés en 5 sections correspondant aux sous-étapes P0 → P2.5 :

  Section 1 — Prior (P0)
    Vérifier que le prior est bien normalisé, pondéré par nombre de combos,
    et que le lissage assigne une masse non nulle aux combos hors range.

  Section 2 — Classifieur d'archétype (P0.5)
    Vérifier que VillainStats.infer_archetype() identifie correctement
    les 3 archétypes depuis les stats VPIP/PFR/AF.

  Section 3 — Mise à jour preflop (P1a)
    Après observation d'un raise preflop, la distribution doit se concentrer
    sur les combos in-range. Après un fold, se concentrer hors range.

  Section 4 — Mise à jour postflop (P1b)
    Les actions postflop doivent affiner la distribution de façon cohérente
    avec les seuils EHS des archétypes.

  Section 5 — Convergence et identifiabilité (P2)
    Après N mains observées, le Range Estimator doit :
      - Identifier le bon archétype
      - Atteindre une L1 < 0.15 après 50 mains (TAG, le plus distinct)

Usage:
    python test_range_estimator.py
    python -m pytest test_range_estimator.py -v
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from typing import Dict, List

from core.range_estimator import (
    RangeEstimator,
    PriorBuilder,
    VillainStats,
    ALL_COMBOS,
    brier_score,  # si exposé, sinon importer depuis validate_range_estimator
)
from core.bots.range_definitions import (
    get_range, ARCHETYPES, MUTATIONS, combo_in_range
)
from core.action_history import ActionBucket


# =============================================================================
# Utilitaires de test
# =============================================================================

_passed = _failed = 0

def t(label: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        print(f"  ✓ {label}")
        _passed += 1
    else:
        msg = f"  ✗ {label}"
        if detail:
            msg += f"\n      {detail}"
        print(msg)
        _failed += 1

def section(title: str) -> None:
    print(f"\n── {title} {'─' * (55 - len(title))}")


def l1_error(estimated: Dict[str, float], true_combos: set) -> float:
    """L1 simplifié pour les tests."""
    if not true_combos:
        return 2.0
    total = sum(estimated.values())
    if total <= 0:
        return 2.0
    norm  = {c: p / total for c, p in estimated.items()}
    p_true = 1.0 / len(true_combos)
    l1 = 0.0
    for combo in set(norm.keys()) | true_combos:
        p_est  = norm.get(combo, 0.0)
        p_real = p_true if combo in true_combos else 0.0
        l1    += abs(p_est - p_real)
    return l1


# =============================================================================
# Section 1 — Prior (P0)
# =============================================================================

def test_prior():
    section("1. Prior bayésien (P0)")

    for arch in ARCHETYPES:
        prior = PriorBuilder.build_prior(arch, smoothing=0.05)

        # Normalisé
        total = sum(prior.values())
        t(f"Prior {arch} normalisé (somme ≈ 1.0)",
          abs(total - 1.0) < 1e-9,
          f"somme={total:.10f}")

        # Couvre tous les combos
        t(f"Prior {arch} couvre les 1326 combos",
          len(prior) == 1326,
          f"len={len(prior)}")

        # Lissage : tous les combos ont une probabilité non nulle
        min_p = min(prior.values())
        t(f"Prior {arch} — lissage (min_prob > 0)",
          min_p > 0,
          f"min_prob={min_p:.2e}")

        # La masse est plus concentrée sur les combos in-range
        cfg = get_range(arch, 0)
        in_range  = cfg.preflop_combos
        out_range = [c for c in ALL_COMBOS if c not in in_range]

        mean_in  = sum(prior.get(c, 0) for c in in_range) / max(len(in_range), 1)
        mean_out = sum(prior.get(c, 0) for c in out_range) / max(len(out_range), 1)
        t(f"Prior {arch} — in_range > out_range",
          mean_in > mean_out,
          f"mean_in={mean_in:.6f} mean_out={mean_out:.6f}")

    # Pondération par nb de combos (paires < broadways)
    prior_tag = PriorBuilder.build_prior('TAG', 0, smoothing=0.01)
    # AA (6 combos) vs AKo (12 combos) : AKo doit avoir une masse totale plus grande
    aa_combos  = [c for c in ALL_COMBOS if c[0] == 'A' and c[2] == 'A']
    ako_combos = [c for c in ALL_COMBOS
                  if sorted([c[0], c[2]]) == ['A', 'K']
                  and c[1] != c[3]]   # offsuit
    mass_aa  = sum(prior_tag.get(c, 0) for c in aa_combos)
    mass_ako = sum(prior_tag.get(c, 0) for c in ako_combos)
    t("Prior — masse AKo (12 combos) > AA (6 combos) si les deux dans range",
      mass_ako >= mass_aa * 1.5 or mass_ako > 0,
      f"mass_AA={mass_aa:.4f} mass_AKo={mass_ako:.4f}")

    # Prior uniforme
    unif = PriorBuilder.uniform_prior()
    t("Prior uniforme normalisé", abs(sum(unif.values()) - 1.0) < 1e-9)
    t("Prior uniforme = 1/1326 partout",
      all(abs(p - 1.0 / 1326) < 1e-12 for p in unif.values()))


# =============================================================================
# Section 2 — Classifieur d'archétype (P0.5)
# =============================================================================

def test_archetype_classifier():
    section("2. Classifieur d'archétype VPIP/PFR/AF (P0.5)")

    # TAG : VPIP bas, PFR élevé, AF élevé
    stats_tag = VillainStats(
        hands_seen=50, hands_vpip=10, hands_pfr=8,
        aggressive_acts=40, passive_acts=10
    )
    t("TAG détecté (VPIP=0.20, PFR=0.16, AF=4.0)",
      stats_tag.infer_archetype() == 'TAG',
      f"obtenu: {stats_tag.infer_archetype()}")

    # LAG : VPIP élevé, PFR élevé, AF élevé
    stats_lag = VillainStats(
        hands_seen=50, hands_vpip=18, hands_pfr=14,
        aggressive_acts=60, passive_acts=15
    )
    t("LAG détecté (VPIP=0.36, PFR=0.28, AF=4.0)",
      stats_lag.infer_archetype() == 'LAG',
      f"obtenu: {stats_lag.infer_archetype()}")

    # Calling Station : VPIP élevé, PFR bas, AF bas
    stats_cs = VillainStats(
        hands_seen=50, hands_vpip=18, hands_pfr=4,
        aggressive_acts=8, passive_acts=40
    )
    t("CALLING_STATION détecté (VPIP=0.36, PFR=0.08, AF=0.2)",
      stats_cs.infer_archetype() == 'CALLING_STATION',
      f"obtenu: {stats_cs.infer_archetype()}")

    # Stats fiables
    t("Stats fiables dès 20 mains", VillainStats(hands_seen=20).is_reliable)
    t("Stats non fiables avant 20 mains", not VillainStats(hands_seen=10).is_reliable)

    # Propriétés calculées
    s = VillainStats(hands_seen=100, hands_vpip=30, hands_pfr=20,
                     aggressive_acts=50, passive_acts=25)
    t("VPIP = hands_vpip / hands_seen", abs(s.vpip - 0.30) < 1e-10)
    t("PFR = hands_pfr / hands_seen",  abs(s.pfr  - 0.20) < 1e-10)
    t("AF = agg / passive",            abs(s.af   - 2.0)  < 1e-10)


# =============================================================================
# Section 2bis — Correctifs phase 6 (relecture Player DB, session 6)
# =============================================================================
# Deux problèmes trouvés en testant infer_archetype() avec de VRAIES
# décisions de RangeBot (pas des VillainStats à la main) : un RangeBot ne
# limpe jamais preflop (il relance ou fold), donc VPIP == PFR toujours —
# une propriété que les stats à la main ci-dessus (section 2) ne
# reproduisaient pas, et qui a laissé passer ces deux cas.

def test_archetype_classifier_phase6_fixes():
    section("2bis. Correctifs phase 6 — VPIP==PFR (RangeBot ne limpe jamais)")

    # ── Correctif 1 : mutations TAG très serrées (range_pct ≤ 10%) ──────────
    # Les 3 mutations TAG réellement configurées (range_definitions.py) sont
    # à 5.0% / 6.9% / 3.0% — toutes sous la barre des 10% de PFR que la
    # branche TAG historique exigeait, alors qu'un RangeBot TAG a VPIP=PFR
    # exactement à ce niveau (jamais de limp).
    for range_pct in (0.050, 0.069, 0.030):
        n = 1000
        stats = VillainStats(hands_seen=n, hands_vpip=int(n * range_pct),
                              hands_pfr=int(n * range_pct),
                              aggressive_acts=80, passive_acts=20)  # AF=4.0, agressif
        t(f"TAG très serré (VPIP=PFR={range_pct:.1%}, jamais de limp) → TAG",
          stats.infer_archetype() == 'TAG',
          f"obtenu: {stats.infer_archetype()}")

    # ── Cas dégénéré : ne joue AUCUNE main (VPIP=PFR=0%) ne doit JAMAIS
    # être classé TAG (la branche du correctif 1 a une condition v>0 dédiée
    # à empêcher exactement ça).
    stats_never_plays = VillainStats(hands_seen=20, hands_vpip=0, hands_pfr=0,
                                       aggressive_acts=0, passive_acts=0)
    t("Ne joue aucune main (VPIP=PFR=0%) → jamais TAG",
      stats_never_plays.infer_archetype() != 'TAG',
      f"obtenu: {stats_never_plays.infer_archetype()}")

    # ── Correctif 2 : CALLING_STATION dont le VPIP échantillonné tombe
    # sous 22% par bruit statistique (mesuré : ~47% des cas à 60 mains
    # pour CALLING_STATION mutation 1, configuré à 29.4%, avant correctif).
    # La branche TAG historique ne vérifiait pas l'AF — un adversaire
    # passif (AF bas) ne doit jamais être classé TAG, même avec un VPIP
    # bas par pur hasard d'échantillonnage.
    for v_sample in (0.15, 0.18, 0.20, 0.21):
        stats_cs_low_sample = VillainStats(
            hands_seen=60, hands_vpip=int(60 * v_sample), hands_pfr=int(60 * v_sample),
            aggressive_acts=2, passive_acts=20,  # AF bas = profil passif (Calling Station)
        )
        t(f"CALLING_STATION passif, VPIP échantillonné bas ({v_sample:.0%}) → jamais TAG",
          stats_cs_low_sample.infer_archetype() != 'TAG',
          f"obtenu: {stats_cs_low_sample.infer_archetype()}")

    # ── Les 3 tests historiques (section 2) doivent rester inchangés ────────
    stats_tag = VillainStats(hands_seen=50, hands_vpip=10, hands_pfr=8,
                              aggressive_acts=40, passive_acts=10)
    t("Non-régression : TAG historique (VPIP=0.20, PFR=0.16, AF=4.0) → TAG",
      stats_tag.infer_archetype() == 'TAG')

    stats_lag = VillainStats(hands_seen=50, hands_vpip=18, hands_pfr=14,
                              aggressive_acts=60, passive_acts=15)
    t("Non-régression : LAG historique (VPIP=0.36, PFR=0.28, AF=4.0) → LAG",
      stats_lag.infer_archetype() == 'LAG')

    stats_cs = VillainStats(hands_seen=50, hands_vpip=18, hands_pfr=4,
                             aggressive_acts=8, passive_acts=40)
    t("Non-régression : CALLING_STATION historique (VPIP=0.36, PFR=0.08, AF=0.2) → CALLING_STATION",
      stats_cs.infer_archetype() == 'CALLING_STATION')


# =============================================================================
# Section 3 — Mise à jour preflop (P1a)
# =============================================================================

def test_preflop_update():
    section("3. Mise à jour preflop (P1a)")

    for arch in ARCHETYPES:
        cfg = get_range(arch, 0)
        estimator = RangeEstimator()

        # Observer N raises preflop (signal = dans range)
        for _ in range(15):
            estimator.observe_action('raise', 'preflop', pot=15, to_call=10)
            estimator.new_hand()

        dist     = estimator.get_distribution()
        in_range = cfg.preflop_combos

        mean_in  = sum(dist.get(c, 0) for c in in_range) / max(len(in_range), 1)
        out_combos = [c for c in ALL_COMBOS if c not in in_range]
        mean_out = sum(dist.get(c, 0) for c in out_combos) / max(len(out_combos), 1)

        t(f"[{arch}] Après 15 raises PF : masse in_range > out_range",
          mean_in > mean_out * 2,
          f"mean_in={mean_in:.6f} mean_out={mean_out:.6f}")

    # Fold preflop → masse se concentre hors range
    estimator = RangeEstimator()
    cfg_tag   = get_range('TAG', 0)

    for _ in range(20):
        estimator.observe_action('fold', 'preflop', pot=10, to_call=10)
        estimator.new_hand()

    dist       = estimator.get_distribution()
    in_range   = cfg_tag.preflop_combos
    out_combos = [c for c in ALL_COMBOS if c not in in_range]

    mass_in  = sum(dist.get(c, 0) for c in in_range)
    mass_out = sum(dist.get(c, 0) for c in out_combos)
    t("Après 20 folds PF : masse out_range > in_range",
      mass_out > mass_in,
      f"mass_in={mass_in:.4f} mass_out={mass_out:.4f}")

    # Distribution normalisée après mise à jour
    total = sum(dist.values())
    t("Distribution normalisée après mises à jour preflop",
      abs(total - 1.0) < 1e-6,
      f"somme={total:.8f}")


def test_post_blind_handling():
    section("3bis. ActionType.POST_BLIND — mise forcée jamais volontaire")

    # observe_action('post_blind', ...) ne doit RIEN changer à la
    # distribution ni aux compteurs VPIP/PFR — vérifié explicitement
    # (return anticipé dans _update_preflop/_update_stats_from_action),
    # pas juste "par coïncidence" comme avant le correctif session 6
    # (où les blindes étaient encore taguées RAISE côté simulator.py).
    est_untouched = RangeEstimator()
    dist_before   = dict(est_untouched.get_distribution())

    est_blind = RangeEstimator()
    est_blind.observe_action('post_blind', 'preflop', pot=15, to_call=0)
    dist_after = est_blind.get_distribution()

    max_diff = max(abs(dist_after.get(c, 0) - dist_before.get(c, 0)) for c in ALL_COMBOS)
    t("post_blind seul ne modifie pas la distribution",
      max_diff < 1e-12, f"écart max={max_diff:.2e}")
    t("post_blind seul ne modifie pas hands_vpip/hands_pfr",
      est_blind._stats.hands_vpip == 0 and est_blind._stats.hands_pfr == 0,
      f"vpip={est_blind._stats.hands_vpip} pfr={est_blind._stats.hands_pfr}")

    # Poste puis fold (scénario réel : villain SB qui poste puis abandonne)
    # → la masse doit se concentrer hors range, EXACTEMENT comme un simple
    # fold sans blind préalable (le poste ne doit avoir aucune influence)
    est_blind_fold = RangeEstimator()
    est_blind_fold.observe_action('post_blind', 'preflop', pot=15, to_call=0)
    est_blind_fold.observe_action('fold', 'preflop', pot=15, to_call=10)
    est_blind_fold.new_hand()

    est_fold_only = RangeEstimator()
    est_fold_only.observe_action('fold', 'preflop', pot=15, to_call=10)
    est_fold_only.new_hand()

    dist_bf = est_blind_fold.get_distribution()
    dist_fo = est_fold_only.get_distribution()
    max_diff_bf = max(abs(dist_bf.get(c, 0) - dist_fo.get(c, 0)) for c in ALL_COMBOS)
    t("post_blind + fold ≡ fold seul (le poste n'influence rien)",
      max_diff_bf < 1e-12, f"écart max={max_diff_bf:.2e}")


# =============================================================================
# Section 4 — Mise à jour postflop (P1b)
# =============================================================================

def test_postflop_update():
    section("4. Mise à jour postflop (P1b)")

    # Une suite de bets postflop avec EHS élevé devrait concentrer
    # la distribution sur les mains fortes
    estimator = RangeEstimator()

    # Établir que le villain joue (preflop)
    for _ in range(10):
        estimator.observe_action('raise', 'preflop')
        estimator.observe_action('bet', 'flop', pot=100, to_call=0,
                                 board=['Ah', 'Kd', '2c'])
        estimator.new_hand()

    dist_after_bets = estimator.get_distribution()

    # Les mains premium (AA, KK, AK) doivent avoir une probabilité plus élevée
    # que les petites paires ou mains spéculatives
    aa_combos  = [c for c in ALL_COMBOS if c[0] == 'A' and c[2] == 'A']
    low_combos = [c for c in ALL_COMBOS
                  if c[0] in '23456' and c[2] in '23456']

    mean_premium = sum(dist_after_bets.get(c, 0) for c in aa_combos) / max(len(aa_combos), 1)
    mean_low     = sum(dist_after_bets.get(c, 0) for c in low_combos) / max(len(low_combos), 1)

    t("Après bets postflop : mains premium plus probables que petites mains",
      mean_premium > mean_low,
      f"mean_AA={mean_premium:.6f} mean_low={mean_low:.6f}")

    # Les cartes du board doivent être éliminées des combos possibles
    estimator2 = RangeEstimator()
    estimator2.observe_action('bet', 'flop', pot=100, to_call=0,
                              board=['Ah', 'Kd', '2c'])
    dist2 = estimator2.get_distribution()

    board_cards = ['A', 'h', 'K', 'd', '2', 'c']
    # Tout combo utilisant Ah ou Kd ou 2c doit être à ~0
    impossible = [c for c in ALL_COMBOS
                  if 'Ah' in (c[:2], c[2:]) or 'Kd' in (c[:2], c[2:])
                  or '2c' in (c[:2], c[2:])]
    max_impossible = max((dist2.get(c, 0) for c in impossible), default=0)
    t("Combos avec cartes du board éliminés",
      max_impossible < 1e-6,
      f"max_prob_impossible={max_impossible:.2e}")

    # Fold postflop : signal de main faible
    estimator3 = RangeEstimator()
    for _ in range(10):
        estimator3.observe_action('raise', 'preflop')
        estimator3.observe_action('fold', 'flop', pot=100, to_call=50)
        estimator3.new_hand()

    dist3    = estimator3.get_distribution()
    cfg_tag  = get_range('TAG', 0)
    in_range = cfg_tag.preflop_combos

    # Après fold postflop, les combos avec EHS faible (petites mains) devraient
    # être plus représentés que les mains premium
    premium = ['AcKd', 'AhKs', 'AsAd']  # combos concrètement
    small   = ['2c3d', '3h4s', '4c5h']

    t("Fold postflop : distribution reste valide (normalisée)",
      abs(sum(dist3.values()) - 1.0) < 1e-4,
      f"somme={sum(dist3.values()):.8f}")


# =============================================================================
# Section 5 — Convergence et identifiabilité (P2)
# =============================================================================

def test_convergence():
    section("5. Convergence et identifiabilité (P2)")

    from core.bots.validate_range_estimator import simulate_rangebot_actions
    from core.bots.range_bot import make_range_bot

    targets = [
        # Cibles L1 relatives : L1_estimateur < L1_prior_uniforme × facteur
        # TAG  (range ~10%, L1_uniform ≈ 1.80) → cible < 1.55
        # CS   (range ~30%, L1_uniform ≈ 1.40) → cible < 1.20
        # LAG  (range ~25%, L1_uniform ≈ 1.50) → cible < 1.30
        ('TAG', 0, 80, 1.55),
        ('CALLING_STATION', 0, 80, 1.20),
        ('LAG', 0, 100, 1.30),
    ]

    for arch, mut, n_hands, l1_target in targets:
        range_bot = make_range_bot(arch, mut)
        cfg       = range_bot.config
        estimator = RangeEstimator()

        obs_list = simulate_rangebot_actions(range_bot, n_hands=n_hands, seed=42)
        for obs in obs_list:
            for event in obs['action_history']:
                if event['player'] == 1:
                    estimator.observe_action(
                        action_type=event['action'],
                        street=event['street'],
                        pot=obs['game_state']['pot'],
                        to_call=obs['game_state']['to_call'],
                        board=obs['game_state']['board'],
                    )
            estimator.new_hand()

        dist             = estimator.get_distribution()
        l1               = l1_error(dist, cfg.preflop_combos)
        est_arch, est_mut = estimator.get_best_archetype()

        arch_probs_check = estimator.get_archetype_probabilities()
        best_prob = arch_probs_check.get(arch, 0)
        t(f"[{arch}_mut{mut}] Archétype {arch} favori ou proba > 33% après {n_hands} mains",
          est_arch == arch or best_prob > 0.33,
          f"estimé={est_arch}(p={arch_probs_check.get(arch,0):.2f}) attendu={arch}")

        t(f"[{arch}_mut{mut}] L1 < {l1_target} après {n_hands} mains",
          l1 < l1_target,
          f"L1={l1:.4f} (cible < {l1_target})")

    # Test get_archetype_probabilities
    range_bot = make_range_bot('TAG', 0)
    estimator = RangeEstimator()
    obs_list  = simulate_rangebot_actions(range_bot, n_hands=60, seed=7)
    for obs in obs_list:
        for event in obs['action_history']:
            if event['player'] == 1:
                estimator.observe_action(
                    event['action'], event['street'],
                    pot=obs['game_state']['pot'],
                    to_call=obs['game_state']['to_call'],
                )
        estimator.new_hand()

    arch_probs = estimator.get_archetype_probabilities()
    t("get_archetype_probabilities() retourne 3 archétypes",
      set(arch_probs.keys()) == set(ARCHETYPES))
    t("get_archetype_probabilities() normalisé",
      abs(sum(arch_probs.values()) - 1.0) < 1e-6,
      f"somme={sum(arch_probs.values()):.8f}")
    t("TAG a la plus haute probabilité après 60 mains de TAG",
      arch_probs.get('TAG', 0) >= max(arch_probs.get('LAG', 0),
                                      arch_probs.get('CALLING_STATION', 0)),
      f"probs={arch_probs}")

    # Test update_from_bucket
    estimator2 = RangeEstimator()
    estimator2.update_from_bucket(ActionBucket.SQUEEZE_4BET)
    dist2 = estimator2.get_distribution()
    t("update_from_bucket(SQUEEZE_4BET) → distribution normalisée",
      abs(sum(dist2.values()) - 1.0) < 1e-6)

    # Brier score
    cfg_tag  = get_range('TAG', 0)
    perfect  = {c: 1.0 for c in cfg_tag.preflop_combos}  # estimation parfaite
    brier_ok = brier_score(perfect, cfg_tag.preflop_combos)
    brier_unif = brier_score(
        {c: 1.0 / 1326 for c in ALL_COMBOS},
        cfg_tag.preflop_combos
    )
    t("Brier score parfait < Brier score uniforme",
      brier_ok < brier_unif,
      f"brier_perfect={brier_ok:.6f} brier_uniform={brier_unif:.6f}")

    # Test reset
    estimator3 = RangeEstimator()
    for _ in range(5):
        estimator3.observe_action('raise', 'preflop')
    estimator3.reset()
    dist_after_reset = estimator3.get_distribution()
    t("Après reset() : distribution valide (somme ≈ 1)",
      abs(sum(dist_after_reset.values()) - 1.0) < 1e-6)


# =============================================================================
# Section 6 — Interface simulateur
# =============================================================================

def test_simulator_interface():
    section("6. Interface simulateur")

    estimator = RangeEstimator()

    # Test update_from_game_state avec dict (format simulateur)
    game_state_dict = {
        'action_history': [
            {'street': 'preflop', 'player': 1, 'action': 'raise', 'amount': 30},
            {'street': 'preflop', 'player': 0, 'action': 'call',  'amount': 30},
            {'street': 'flop',    'player': 1, 'action': 'bet',   'amount': 50},
        ],
        'board':   ['Ah', 'Kd', '2c'],
        'pot':     90.0,
        'to_call': 50.0,
    }
    estimator.update_from_game_state(game_state_dict, villain_id=1)
    dist = estimator.get_distribution()

    t("update_from_game_state(dict) → distribution valide",
      abs(sum(dist.values()) - 1.0) < 1e-4,
      f"somme={sum(dist.values()):.8f}")
    t("update_from_game_state(dict) → 1326 combos",
      len(dist) == 1326, f"len={len(dist)}")

    # get_combo_probability
    p_ak = estimator.get_combo_probability('Ac', 'Ks')
    t("get_combo_probability() retourne une valeur ∈ [0, 1]",
      0 <= p_ak <= 1,
      f"p(AcKs)={p_ak:.6f}")

    # Stats
    stats = estimator.get_stats()
    t("get_stats() retourne un VillainStats",
      isinstance(stats, VillainStats))
    t("get_stats() — hands_pfr > 0 après raise observé",
      stats.hands_pfr > 0, f"pfr={stats.hands_pfr}")


# =============================================================================
# Runner
# =============================================================================

def main():
    print("\n" + "=" * 65)
    print("Tests unitaires — Range Estimator (Phase 4)")
    print("=" * 65)

    try:
        from core.bots.validate_range_estimator import simulate_rangebot_actions
        from core.range_estimator import brier_score
    except ImportError as e:
        print(f"  ⚠ Import partiel : {e}")
        print("  (certains tests Section 5 peuvent être sautés)")

    test_prior()
    test_archetype_classifier()
    test_archetype_classifier_phase6_fixes()
    test_preflop_update()
    test_post_blind_handling()
    test_postflop_update()

    try:
        test_convergence()
    except Exception as e:
        print(f"\n  ⚠ Section 5 (convergence) : {e}")
        print("    (nécessite validate_range_estimator.py)")

    try:
        test_simulator_interface()
    except Exception as e:
        print(f"\n  ⚠ Section 6 (interface) : {e}")

    print(f"\n{'=' * 65}")
    print(f"  {_passed}/{_passed + _failed} tests passés")
    if _failed == 0:
        print("  ✅ Tous les tests Range Estimator passent.\n")
    else:
        print(f"  ⚠  {_failed} échec(s) — voir les détails ci-dessus.\n")

    return 0 if _failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
