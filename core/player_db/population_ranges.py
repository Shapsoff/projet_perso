"""
population_ranges.py — Ranges "population" par position (Palier 0 du Cold Start)
Phase 6 — Bot Poker Académique

Contrairement à range_definitions.py (qui encode les 9 archétypes RangeBots,
utilisés comme vérité terrain de test), ce module encode des ranges
d'ouverture *génériques*, proches des charts standards 100BB 6-max qu'on
retrouve dans toute ressource de poker moderne. Ce ne sont ni des mesures
empiriques, ni les seuils d'un archétype — c'est un prior "raisonnable pour
un joueur inconnu", volontairement indépendant des 9 profils de test.

Rôle dans le pipeline Player DB (cf. profile_builder.py) :
    Palier 0 (< TIER0_MAX_HANDS mains observées sur CE joueur, cumulées
    sur toute la Player DB, pas seulement le match en cours) : le
    RangeEstimator ne peut pas encore distinguer un TAG d'un LAG d'un
    inconnu. Plutôt que de repartir d'un prior uniforme sur les 1326 combos
    (naïf et dangereux — cf. limitation phase 3) ou de forcer un des 9
    archétypes RangeBot (biaisé), on utilise ce prior de population par
    position. Il protège sans être passif : un joueur inconnu ouvrant UTG
    a structurellement plus de chances d'avoir une main forte qu'un joueur
    ouvrant BTN, même sans aucune donnée le concernant.

Ces charts sont volontairement resserrées côté "value" (on ne veut pas
sous-estimer un adversaire inconnu en early position) et suffisamment
larges aux positions tardives pour ne pas être exploité par une stratégie
naïvement passive à la BTN/SB.

Limite assumée (documentée, dans l'esprit des rapports de phases
précédentes) : un seul chart par position, pas de distinction ouverture /
3bet / call de 3bet à ce stade. Le Palier 0 est un prior de démarrage, pas
un modèle complet — il est remplacé dès que possible par le Palier 1
(archétype) puis le Palier 2 (profil empirique).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from core.bots.range_definitions import expand_range

# =============================================================================
# Charts d'ouverture "population" — 100BB 6-max, cash game
# =============================================================================
# Notation identique à range_definitions.py (expand_hand_class).
# Ces ranges sont des approximations pédagogiques standards, pas une donnée
# propriétaire — comparables à ce qu'on trouve dans n'importe quel article
# d'introduction au jeu préflop 6-max.

_POP_UTG: List[str] = [
    '77+', 'ATs+', 'KTs+', 'QTs+', 'JTs',
    'AJo+', 'KQo',
]  # ≈ 15%

_POP_MP: List[str] = [
    '55+', 'A9s+', 'KTs+', 'QTs+', 'J9s+', 'T9s',
    'ATo+', 'KJo+',
]  # ≈ 18%

_POP_CO: List[str] = [
    '22+', 'A2s+', 'K9s+', 'Q9s+', 'J8s+', 'T8s+', '98s',
    'A8o+', 'KTo+', 'QTo+',
]  # ≈ 27%

_POP_BTN: List[str] = [
    '22+', 'A2s+', 'K2s+', 'Q4s+', 'J6s+', 'T6s+', '96s+', '85s+', '74s+',
    'A2o+', 'K8o+', 'Q9o+', 'J9o+', 'T9o',
]  # ≈ 45%

_POP_SB: List[str] = [
    '22+', 'A2s+', 'K5s+', 'Q7s+', 'J7s+', 'T7s+', '97s+', '86s+',
    'A5o+', 'K9o+', 'QTo+', 'JTo',
]  # ≈ 35% (SB isolé, un peu plus tight que BTN car SB parle après)

_POP_BB: List[str] = [
    # Range de "défense" (call face à un open), volontairement plus large
    # qu'un chart d'ouverture — la BB ferme l'action et défend large en
    # pratique. Utilisée comme prior par défaut faute de mieux au Palier 0.
    '22+', 'A2s+', 'K4s+', 'Q6s+', 'J7s+', 'T7s+', '97s+', '86s+', '75s+',
    'A2o+', 'K8o+', 'Q9o+', 'J9o+', 'T9o',
]  # ≈ 42%

_CHARTS_BY_POSITION: Dict[str, List[str]] = {
    'UTG':  _POP_UTG,
    'UTG+1': _POP_UTG,   # alias si table 6-max avec UTG+1 nommé distinctement
    'MP':   _POP_MP,
    'CO':   _POP_CO,
    'BTN':  _POP_BTN,
    'SB':   _POP_SB,
    'BB':   _POP_BB,
}

# Prior par défaut si la position est inconnue / non fournie :
# moyenne des 6 positions, pondérée par leur fréquence à table (approximation
# uniforme — chaque position revient une fois par tour).
_DEFAULT_POSITIONS = ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB']


# =============================================================================
# Pondération des combos (identique à range_estimator._combo_weight)
# =============================================================================

def _combo_weight(card1: str, card2: str) -> float:
    r1, s1 = card1[0], card1[1]
    r2, s2 = card2[0], card2[1]
    if r1 == r2:
        return 1.0 / 6.0
    elif s1 == s2:
        return 1.0 / 4.0
    else:
        return 1.0 / 12.0


# =============================================================================
# API publique
# =============================================================================

def get_position_chart(position: Optional[str]) -> List[str]:
    """Retourne la liste de classes de mains (hand_classes) pour une position.

    Position inconnue ou None → chart MP (compromis raisonnable, ni trop
    tight ni trop large) plutôt qu'une exception : le Palier 0 doit rester
    utilisable même quand la position adverse n'est pas encore certaine.
    """
    if position is None:
        return _POP_MP
    return _CHARTS_BY_POSITION.get(position.upper(), _POP_MP)


def build_population_prior(
    position: Optional[str] = None,
    smoothing: float = 0.08,
) -> Dict[str, float]:
    """
    Construit un prior de combos (Dict[combo, proba], somme = 1) à partir
    du chart de population de la position donnée.

    Args:
        position  : 'UTG'/'MP'/'CO'/'BTN'/'SB'/'BB', ou None pour un mix
                    des 6 positions (utilisé quand la position adverse
                    n'est pas encore connue avec certitude).
        smoothing : masse assignée uniformément en dehors de la range
                    (même rôle que dans PriorBuilder.build_prior — évite
                    la dégénérescence bayésienne si l'adversaire agit hors
                    de son chart supposé, ce qui EST attendu pour un joueur
                    inconnu, d'où un smoothing un peu plus généreux que le
                    0.05 utilisé pour les archétypes RangeBot déterministes).

    Returns:
        Dict[combo, probabilité] normalisé sur les 1326 combos.
    """
    from core.range_estimator import ALL_COMBOS  # import tardif (cycle)

    if position is None:
        # Mix des 6 positions à parts égales — prior "aucune info de siège"
        charts = [get_position_chart(p) for p in _DEFAULT_POSITIONS]
    else:
        charts = [get_position_chart(position)]

    combo_weights: Dict[str, float] = {}
    for chart in charts:
        combos = expand_range(chart)
        for combo in combos:
            c1, c2 = combo[:2], combo[2:]
            w = _combo_weight(c1, c2) / len(charts)
            combo_weights[combo] = combo_weights.get(combo, 0.0) + w

    total_in = sum(combo_weights.values())
    n_combos = len(ALL_COMBOS)
    if total_in <= 0:
        return {c: 1.0 / n_combos for c in ALL_COMBOS}

    base_out = smoothing / n_combos
    scale_in = (1.0 - smoothing) / total_in

    result: Dict[str, float] = {}
    for combo in ALL_COMBOS:
        p_in = combo_weights.get(combo, 0.0) * scale_in
        result[combo] = p_in + base_out

    total = sum(result.values())
    return {c: p / total for c, p in result.items()}


def population_range_pct(position: Optional[str] = None) -> float:
    """Fraction approximative de combos jouée par le chart de population."""
    from core.range_estimator import ALL_COMBOS
    chart = get_position_chart(position)
    combos = expand_range(chart)
    return len(combos) / len(ALL_COMBOS)


# =============================================================================
# Tests intégrés (même convention que range_definitions.py)
# =============================================================================

if __name__ == "__main__":
    print("\n=== Tests population_ranges.py ===\n")
    passed = failed = 0

    def t(label, cond, detail=""):
        global passed, failed
        if cond:
            print(f"  ✓ {label}")
            passed += 1
        else:
            print(f"  ✗ {label}" + (f" : {detail}" if detail else ""))
            failed += 1

    # Charts croissants UTG < MP < CO < BTN
    pct_utg = population_range_pct('UTG')
    pct_mp  = population_range_pct('MP')
    pct_co  = population_range_pct('CO')
    pct_btn = population_range_pct('BTN')
    t("UTG < MP < CO < BTN (largeur croissante)",
      pct_utg < pct_mp < pct_co < pct_btn,
      f"UTG={pct_utg:.1%} MP={pct_mp:.1%} CO={pct_co:.1%} BTN={pct_btn:.1%}")

    # Prior normalisé
    for pos in [None, 'UTG', 'BTN', 'BB', 'inconnue']:
        prior = build_population_prior(pos)
        total = sum(prior.values())
        t(f"prior position={pos} somme à 1", abs(total - 1.0) < 1e-6,
          f"obtenu {total:.6f}")

    # Position insensible à la casse
    p1 = build_population_prior('btn')
    p2 = build_population_prior('BTN')
    t("position insensible à la casse", p1 == p2)

    # AA doit être plus probable que 72o dans n'importe quelle position
    prior_utg = build_population_prior('UTG')
    p_aa  = prior_utg.get('AcAd', 0) + prior_utg.get('AdAc', 0)
    p_72o = prior_utg.get('7c2d', 0) + prior_utg.get('2d7c', 0)
    t("AA plus probable que 72o (UTG)", p_aa > p_72o,
      f"AA={p_aa:.6f} 72o={p_72o:.6f}")

    print(f"\n  {passed}/{passed+failed} tests passés")
    print("  ✅ OK\n" if failed == 0 else f"  ⚠ {failed} échec(s)\n")
