"""
profile_builder.py — Construction du profil adverse à partir de la Player DB
Phase 6 — Bot Poker Académique

Implémente la stratégie à 3 paliers validée en session 6 :

  Palier 0 (< TIER0_MAX_HANDS mains cumulées en DB pour ce joueur) :
      Prior de population par position (population_ranges.py). Ni
      complètement aveugle (prior uniforme), ni sur-engagé (un des 9
      archétypes RangeBot). Protège sans subir.

  Palier 1 (TIER0_MAX_HANDS ≤ mains < TIER1_MAX_HANDS) :
      Classification parmi les 9 archétypes existants (TAG/LAG/
      CALLING_STATION × 3 mutations) — réutilise VillainStats.infer_archetype()
      de range_estimator.py (phase 4), appliqué ici aux compteurs cumulés
      en DB plutôt qu'aux seules stats du match en cours. Bon point de
      départ discriminant à ce volume de données (décision validée phase 4).

  Palier 2 (mains ≥ TIER1_MAX_HANDS) :
      Profil empirique propre au joueur : distribution de range affinée à
      partir des combos réellement vus au showdown (record_showdown_combo),
      mélangée progressivement avec le prior d'archétype (le poids
      empirique croît avec le nombre de showdowns observés — pas de
      bascule brutale), ET fréquences fold/call/raise observées par bucket
      d'EHS (empirical_action_freq) — c'est le point structurel manquant
      identifié en phase 5 (section 9.1) : le Frequency Model n'est plus
      condamné aux seuils déterministes des 9 RangeBots.

Ces seuils (15 / 30 mains) sont volontairement alignés sur les constantes
déjà utilisées dans le projet (PlayerStats.is_reliable = 30 mains dans
game_state.py, min_hands_for_best_response = 30 en phase 5) plutôt que
d'introduire une nouvelle convention.

Ce module ne fait AUCUNE requête SQL directe — il consomme uniquement
l'API publique de PlayerDB (player_db.py), qui reste la seule couche à
connaître le détail du schéma SQLite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from core.bots.range_definitions import get_range, MUTATIONS
from .player_db import PlayerDB, PlayerRow
from .population_ranges import build_population_prior

# =============================================================================
# Constantes des paliers
# =============================================================================

TIER0_MAX_HANDS = 15   # < 15 mains : prior de population (Palier 0)
TIER1_MAX_HANDS = 30   # 15-29 mains : archétype (Palier 1) ; 30+ : Palier 2

# Confiance archétype : croît linéairement de 0.45 (à TIER0_MAX_HANDS mains)
# à 0.95 (plafond), atteint vers 100 mains. Volontairement progressive :
# pas de saut brutal à 15 mains pile.
_CONFIDENCE_FLOOR   = 0.45
_CONFIDENCE_CEILING = 0.95
_CONFIDENCE_HANDS_TO_CEILING = 100

# Nombre minimum d'observations dans un bucket (street, ehs_bucket,
# facing_bet) pour que sa fréquence empirique soit jugée exploitable.
# En dessous, le bucket est absent de empirical_action_freq et l'appelant
# (DBAwareFrequencyModel) doit retomber sur le modèle archétype.
MIN_OBS_PER_BUCKET = 5

# Nombre minimum de showdowns pour commencer à mélanger la distribution
# empirique de combos dans le prior preflop (cf. get_preflop_prior).
MIN_SHOWDOWNS_FOR_EMPIRICAL = 20
# Poids empirique maximal jamais atteint (on garde toujours un peu
# d'archétype en filet de sécurité, même avec des centaines de showdowns).
MAX_EMPIRICAL_WEIGHT = 0.70


# =============================================================================
# Profil adverse
# =============================================================================

@dataclass
class PlayerProfile:
    """
    Résultat de build_profile() — vue consolidée d'un adversaire, prête à
    être consommée par DBAwareRangeEstimator / DBAwareFrequencyModel.
    """
    player_id:  str
    tier:       int     # 0, 1, ou 2
    hands_seen: int
    vpip:       float
    pfr:        float
    af:         float

    archetype:            Optional[str] = None
    mutation:              Optional[int] = None
    archetype_confidence:  float = 0.0

    # Palier 0 uniquement : prior de combos prêt à l'emploi
    preflop_prior: Optional[Dict[str, float]] = None

    # Palier 2 uniquement : {(street, ehs_bucket, facing_bet): {action: proba, 'n_obs': int}}
    empirical_action_freq: Dict[Tuple[str, int, int], Dict[str, float]] = field(
        default_factory=dict
    )
    n_showdowns: int = 0

    def has_empirical_bucket(self, street: str, ehs_bucket: int, facing_bet: bool) -> bool:
        return (street, ehs_bucket, int(facing_bet)) in self.empirical_action_freq

    def get_empirical_bucket(
        self, street: str, ehs_bucket: int, facing_bet: bool
    ) -> Optional[Dict[str, float]]:
        return self.empirical_action_freq.get((street, ehs_bucket, int(facing_bet)))

    def summary(self) -> str:
        base = (f"PlayerProfile({self.player_id}, tier={self.tier}, "
                f"hands={self.hands_seen}, vpip={self.vpip:.2f}, "
                f"pfr={self.pfr:.2f}, af={self.af:.1f}")
        if self.tier >= 1:
            base += f", arch={self.archetype}({self.mutation}) conf={self.archetype_confidence:.2f}"
        if self.tier >= 2:
            base += f", showdowns={self.n_showdowns}, buckets={len(self.empirical_action_freq)}"
        return base + ")"


# =============================================================================
# Palier 1 — classification d'archétype depuis les compteurs cumulés
# =============================================================================

def _infer_archetype_and_mutation(row: PlayerRow) -> Tuple[str, int]:
    """
    Réutilise la même heuristique VPIP/PFR/AF que VillainStats.infer_archetype
    (range_estimator.py, phase 4), appliquée ici aux compteurs multi-sessions
    de la Player DB plutôt qu'au seul match en cours.

    On importe VillainStats localement pour construire l'objet attendu par
    infer_archetype() sans dupliquer sa logique de seuils.

    Correctif session 6 (bug trouvé via diag_vpip_trace.py, cf. schema.py
    pour le détail complet) : VillainStats.vpip/.pfr divisent par son
    propre champ hands_seen — on lui passe donc row.preflop_opportunities
    (mains avec une vraie décision), PAS row.hands_seen (qui inclut les
    "walks" sans décision et diluait artificiellement le VPIP mesuré,
    faisant passer des LAG/CALLING_STATION assez larges pour des TAG).
    """
    from core.range_estimator import VillainStats

    stats = VillainStats(
        hands_seen=row.preflop_opportunities,
        hands_vpip=row.hands_vpip,
        hands_pfr=row.hands_pfr,
        aggressive_acts=row.aggressive_acts,
        passive_acts=row.passive_acts,
    )
    archetype = stats.infer_archetype()

    best_mut, best_diff = 0, math.inf
    for mut in MUTATIONS:
        cfg  = get_range(archetype, mut)
        diff = abs(stats.vpip - cfg.range_pct)
        if diff < best_diff:
            best_diff = diff
            best_mut  = mut

    return archetype, best_mut


def _archetype_confidence(hands_seen: int) -> float:
    """Confiance croissante et plafonnée — cf. constantes en tête de module."""
    if hands_seen <= TIER0_MAX_HANDS:
        return _CONFIDENCE_FLOOR
    progress = (hands_seen - TIER0_MAX_HANDS) / _CONFIDENCE_HANDS_TO_CEILING
    conf = _CONFIDENCE_FLOOR + progress * (_CONFIDENCE_CEILING - _CONFIDENCE_FLOOR)
    return max(_CONFIDENCE_FLOOR, min(_CONFIDENCE_CEILING, conf))


# =============================================================================
# Palier 2 — table de fréquences empiriques (fold/call/raise par bucket EHS)
# =============================================================================

def _smooth_action_freq_table(
    raw: Dict[Tuple[str, int, int], Dict[str, int]],
    min_obs: int = MIN_OBS_PER_BUCKET,
) -> Dict[Tuple[str, int, int], Dict[str, float]]:
    """
    Lissage de Laplace (add-1) sur chaque bucket (street, ehs_bucket,
    facing_bet). Un bucket avec moins de min_obs observations totales est
    omis du résultat — pas assez fiable pour remplacer le modèle archétype,
    l'appelant doit alors retomber sur le comportement phase 5 pour ce
    bucket précis (fallback progressif, pas un tout-ou-rien global).
    """
    result: Dict[Tuple[str, int, int], Dict[str, float]] = {}
    for key, counts in raw.items():
        total = sum(counts.values())
        if total < min_obs:
            continue
        fold_n  = counts.get('fold', 0)
        call_n  = counts.get('call', 0)
        raise_n = counts.get('raise', 0)
        n = fold_n + call_n + raise_n + 3  # +3 : un pseudo-compte par action (Laplace)
        result[key] = {
            'fold':  (fold_n + 1) / n,
            'call':  (call_n + 1) / n,
            'raise': (raise_n + 1) / n,
            'n_obs': total,
        }
    return result


# =============================================================================
# Point d'entrée principal
# =============================================================================

def build_profile(
    db:         PlayerDB,
    player_id:  str,
    position:   Optional[str] = None,
) -> PlayerProfile:
    """
    Construit le PlayerProfile courant pour un adversaire, en appliquant
    la logique à 3 paliers.

    Args:
        db         : PlayerDB ouverte.
        player_id  : identifiant neutre du joueur (cf. player_db.py).
        position   : position adverse actuelle, utilisée pour le prior de
                     population au Palier 0 (None = mix des 6 positions).

    Returns:
        PlayerProfile — jamais None : un joueur totalement inconnu obtient
        un profil Palier 0 avec hands_seen=0.
    """
    row = db.get_player_row(player_id)
    if row is None:
        row = PlayerRow(player_id=player_id)

    hands_seen = row.hands_seen
    if hands_seen < TIER0_MAX_HANDS:
        tier = 0
    elif hands_seen < TIER1_MAX_HANDS:
        tier = 1
    else:
        tier = 2

    profile = PlayerProfile(
        player_id=player_id,
        tier=tier,
        hands_seen=hands_seen,
        vpip=row.vpip,
        pfr=row.pfr,
        af=row.af,
    )

    if tier == 0:
        profile.preflop_prior = build_population_prior(position)
        return profile

    archetype, mutation = _infer_archetype_and_mutation(row)
    profile.archetype            = archetype
    profile.mutation              = mutation
    profile.archetype_confidence  = _archetype_confidence(hands_seen)

    if tier == 1:
        return profile

    # ── Palier 2 ──────────────────────────────────────────────────────────
    raw_table = db.get_action_freq_table(player_id)
    profile.empirical_action_freq = _smooth_action_freq_table(raw_table)
    profile.n_showdowns           = row.showdowns_seen
    return profile


# =============================================================================
# Prior preflop combiné (utilisé par DBAwareRangeEstimator)
# =============================================================================

def get_preflop_prior(
    profile:   PlayerProfile,
    db:        Optional[PlayerDB] = None,
    smoothing: float = 0.05,
) -> Dict[str, float]:
    """
    Retourne le prior de combos (Dict[combo, proba], somme=1) correspondant
    au profil, en mélangeant progressivement archétype et données de
    showdown quand elles sont disponibles (Palier 2).
    """
    from core.range_estimator import ALL_COMBOS, PriorBuilder

    if profile.tier == 0:
        return profile.preflop_prior or build_population_prior(None)

    archetype_prior = PriorBuilder.build_prior(
        profile.archetype, profile.mutation, smoothing=smoothing
    )

    if profile.tier == 1 or db is None:
        return archetype_prior

    showdowns = db.get_showdown_combos(profile.player_id)
    if len(showdowns) < MIN_SHOWDOWNS_FOR_EMPIRICAL:
        return archetype_prior

    combo_counts: Dict[str, int] = {}
    for combo, _hand_class, _pos in showdowns:
        combo_counts[combo] = combo_counts.get(combo, 0) + 1
    total_sd = sum(combo_counts.values())
    if total_sd <= 0:
        return archetype_prior

    empirical = {c: combo_counts.get(c, 0) / total_sd for c in ALL_COMBOS}

    # Poids empirique croît avec le nombre de showdowns, plafonné à
    # MAX_EMPIRICAL_WEIGHT — on garde toujours une part d'archétype comme
    # filet de sécurité, même avec beaucoup de données.
    w_emp  = MAX_EMPIRICAL_WEIGHT * (total_sd / (total_sd + MIN_SHOWDOWNS_FOR_EMPIRICAL))
    w_arch = 1.0 - w_emp

    mixed = {
        c: w_arch * archetype_prior.get(c, 0.0) + w_emp * empirical.get(c, 0.0)
        for c in ALL_COMBOS
    }
    total = sum(mixed.values())
    if total <= 0:
        return archetype_prior
    return {c: p / total for c, p in mixed.items()}


# =============================================================================
# Tests intégrés
# =============================================================================

if __name__ == "__main__":
    import tempfile

    print("\n=== Tests profile_builder.py ===\n")
    passed = failed = 0

    def t(label, cond, detail=""):
        global passed, failed
        if cond:
            print(f"  ✓ {label}")
            passed += 1
        else:
            print(f"  ✗ {label}" + (f" : {detail}" if detail else ""))
            failed += 1

    with tempfile.TemporaryDirectory() as tmp:
        db = PlayerDB(f"{tmp}/test.sqlite3")

        # ── Palier 0 : joueur totalement inconnu ────────────────────────────
        p = build_profile(db, "unknown_player", position="BTN")
        t("joueur inconnu -> tier 0", p.tier == 0)
        t("joueur inconnu -> preflop_prior rempli", p.preflop_prior is not None)
        t("joueur inconnu -> archetype = None", p.archetype is None)
        total = sum(p.preflop_prior.values())
        t("prior tier 0 normalisé", abs(total - 1.0) < 1e-6, f"obtenu {total}")

        # ── Simuler 20 mains d'un joueur très serré (proche TAG) ────────────
        for i in range(20):
            db.new_hand_observed("tight_player")
            vpip = i % 5 == 0   # 20% VPIP
            pfr  = vpip
            db.record_preflop_action("tight_player", "CO", vpip=vpip, pfr=pfr)
            db.record_postflop_aggression("tight_player", aggressive=True)

        p2 = build_profile(db, "tight_player")
        t("20 mains -> tier 1", p2.tier == 1, f"obtenu tier={p2.tier}")
        t("20 mains -> archetype inféré = TAG",
          p2.archetype == "TAG", f"obtenu {p2.archetype}")
        t("confiance tier 1 dans [0.45, 0.95]",
          0.45 <= p2.archetype_confidence <= 0.95,
          f"obtenu {p2.archetype_confidence}")

        # ── Passer à 30+ mains + quelques showdowns ─────────────────────────
        for i in range(20, 35):
            db.new_hand_observed("tight_player")
            vpip = i % 5 == 0
            db.record_preflop_action("tight_player", "CO", vpip=vpip, pfr=vpip)

        for i in range(25):
            db.record_showdown_combo("tight_player", "AhKd", "AKo", "CO")
            db.record_ehs_bucket_action("tight_player", "flop", ehs=0.82,
                                         facing_bet=False, action="raise")
            db.record_ehs_bucket_action("tight_player", "flop", ehs=0.20,
                                         facing_bet=True, action="fold")

        p3 = build_profile(db, "tight_player")
        t("35 mains -> tier 2", p3.tier == 2, f"obtenu tier={p3.tier}")
        t("tier 2 -> empirical_action_freq non vide",
          len(p3.empirical_action_freq) > 0)
        t("tier 2 -> n_showdowns = 25", p3.n_showdowns == 25,
          f"obtenu {p3.n_showdowns}")

        bucket_high = p3.get_empirical_bucket("flop", 8, False)
        t("bucket EHS=0.82 (déciles) trouvé", bucket_high is not None)
        if bucket_high:
            t("bucket EHS élevé -> raise dominant",
              bucket_high['raise'] > bucket_high['fold'],
              f"obtenu {bucket_high}")

        # ── get_preflop_prior mélange archétype + showdowns ─────────────────
        prior_mixed = get_preflop_prior(p3, db)
        total_mixed = sum(prior_mixed.values())
        t("prior tier 2 mélangé normalisé", abs(total_mixed - 1.0) < 1e-6,
          f"obtenu {total_mixed}")
        # AhKd vu 25 fois au showdown -> masse largement supérieure au prior
        # d'archétype seul (qui répartit sur toute la range TAG)
        from .player_db import canonical_combo
        combo_key = canonical_combo("Ah", "Kd")
        prior_arch_only = get_preflop_prior(
            PlayerProfile(player_id="x", tier=1, hands_seen=30,
                          vpip=p3.vpip, pfr=p3.pfr, af=p3.af,
                          archetype=p3.archetype, mutation=p3.mutation),
            db=None,
        )
        t("combo vu 25x au showdown -> masse boostée vs archétype seul",
          prior_mixed.get(combo_key, 0) > prior_arch_only.get(combo_key, 0),
          f"mixed={prior_mixed.get(combo_key,0):.5f} arch={prior_arch_only.get(combo_key,0):.5f}")

        # ── Bucket sous le seuil MIN_OBS_PER_BUCKET reste absent ────────────
        db.record_ehs_bucket_action("tight_player", "turn", ehs=0.50,
                                     facing_bet=True, action="call")
        p4 = build_profile(db, "tight_player")
        t("bucket avec 1 seule observation absent (< MIN_OBS_PER_BUCKET)",
          not p4.has_empirical_bucket("turn", 5, True))

        # ── Persistance : réouvrir la DB (nouvelle session) ──────────────────
        db.close()
        db2 = PlayerDB(f"{tmp}/test.sqlite3")
        p5 = build_profile(db2, "tight_player")
        t("stats persistées après réouverture (nouvelle session)",
          p5.hands_seen == p4.hands_seen, f"{p5.hands_seen} vs {p4.hands_seen}")
        db2.close()

    # ── Régression : dilution du VPIP par les mains "walk" (session 6) ──────
    # Bug trouvé via diag_vpip_trace.py sur validate_phase6.py : un joueur
    # LAG (VPIP réel ~26%) se faisait classer TAG parce que hands_seen
    # comptait aussi les mains où il n'a JAMAIS eu de décision préflop à
    # prendre (walk en BB), diluant le VPIP mesuré à ~13%. Reproduit ici
    # avec un ratio de dilution de 50%, pire que ce qui est attendu en jeu
    # réel (où les walks dépendent de NOTRE taux de fold en SB, pas de
    # l'adversaire tracké — donc plus rares qu'ici en pratique).
    with tempfile.TemporaryDirectory() as tmp3:
        db3 = PlayerDB(f"{tmp3}/walk_dilution.sqlite3")

        # 100 mains "walk" : hands_seen s'incrémente, mais AUCUNE décision
        # préflop n'a eu lieu -> record_preflop_action n'est jamais appelé
        # (exactement ce que fait hand_recorder.py pour une main sans
        # aucune action du joueur suivi ce street).
        for _ in range(100):
            db3.new_hand_observed("lag_dilue")

        # 100 mains avec une VRAIE décision préflop, VPIP≈26% (LAG mut1)
        for i in range(100):
            db3.new_hand_observed("lag_dilue")
            vpip = (i % 4 == 0)  # 25%, proche de LAG mutation 1 (26.1%)
            db3.record_preflop_action("lag_dilue", "CO", vpip=vpip, pfr=vpip)
            db3.record_postflop_aggression("lag_dilue", aggressive=True)

        row_dilue = db3.get_player_row("lag_dilue")
        t("hands_seen compte les 200 mains (walks incluses)",
          row_dilue.hands_seen == 200, f"obtenu {row_dilue.hands_seen}")
        t("preflop_opportunities ne compte que les 100 décisions réelles",
          row_dilue.preflop_opportunities == 100,
          f"obtenu {row_dilue.preflop_opportunities}")
        t("VPIP correctement calculé sur les opportunités (~25%), pas dilué (~12.5%)",
          0.20 <= row_dilue.vpip <= 0.30, f"obtenu {row_dilue.vpip:.1%}")

        profile_dilue = build_profile(db3, "lag_dilue")
        t("archétype correctement classé LAG malgré 50% de mains walk",
          profile_dilue.archetype == "LAG", f"obtenu {profile_dilue.archetype}")
        db3.close()

    print(f"\n  {passed}/{passed+failed} tests passés")
    print("  ✅ OK\n" if failed == 0 else f"  ⚠ {failed} échec(s)\n")
