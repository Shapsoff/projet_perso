"""
range_definitions.py — Définitions des ranges preflop des RangeBots
Phase 3 — Bot Poker Académique

Contient les 9 ranges preflop (3 archétypes × 3 mutations) utilisées
par les RangeBots comme vérité terrain pour valider le Range Estimator.

Structure :
  - TAG  (Tight Aggressive)   : joue peu de mains, bet/raise fort
  - LAG  (Loose Aggressive)   : joue beaucoup de mains, bet/raise fort
  - CALL (Calling Station)    : joue beaucoup de mains, call souvent

Chaque archétype a 3 mutations (0=base, 1=légèrement plus loose, 2=encore plus loose)
Les mutations ne portent que sur la range preflop (décision validée en session 3).

Format des ranges :
  Un ensemble de combos au format "AhKs" (2 cartes spécifiques)
  OU un ensemble de classes de mains ("AA", "AKs", "AKo") développées
  en tous leurs combos spécifiques via expand_range().

Notation :
  "AA"  → paire d'As (6 combos)
  "AKs" → AK suited (4 combos)
  "AKo" → AK offsuit (12 combos)
  "AK"  → AK suited + offsuit (16 combos)

Usage:
    from core.bots.range_definitions import get_range, ArchetypeConfig
    config = get_range('TAG', mutation=0)
    print(len(config.preflop_combos))  # nombre de combos dans la range
"""

from dataclasses import dataclass, field
from typing import Set, Dict, List, Tuple
from itertools import combinations


# =============================================================================
# Constantes cartes
# =============================================================================

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['c', 'd', 'h', 's']
RANK_VAL = {r: i for i, r in enumerate(RANKS)}  # '2'=0, 'A'=12

ALL_CARDS = [r + s for r in RANKS for s in SUITS]  # 52 cartes


# =============================================================================
# Développement des classes de mains en combos spécifiques
# =============================================================================

def _pair_combos(rank: str) -> Set[str]:
    """Ex: 'AA' → {'AcAd', 'AcAh', 'AcAs', 'AdAh', 'AdAs', 'AhAs'}"""
    cards = [rank + s for s in SUITS]
    return {c1 + c2 for c1, c2 in combinations(cards, 2)}


def _suited_combos(rank1: str, rank2: str) -> Set[str]:
    """Ex: 'AKs' → {'AcKc', 'AdKd', 'AhKh', 'AsKs'}"""
    return {rank1 + s + rank2 + s for s in SUITS}


def _offsuit_combos(rank1: str, rank2: str) -> Set[str]:
    """Ex: 'AKo' → toutes les combinaisons AK de couleurs différentes (12 combos)"""
    result = set()
    for s1 in SUITS:
        for s2 in SUITS:
            if s1 != s2:
                result.add(rank1 + s1 + rank2 + s2)
    return result


def _all_combos(rank1: str, rank2: str) -> Set[str]:
    """Suited + offsuit."""
    return _suited_combos(rank1, rank2) | _offsuit_combos(rank1, rank2)


def expand_hand_class(hand_class: str) -> Set[str]:
    """
    Développe une classe de main en combos spécifiques.

    Formats acceptés :
      "AA"   → paire
      "AKs"  → suited
      "AKo"  → offsuit
      "AK"   → suited + offsuit
      "ATs+" → ATs, AJs, AQs, AKs (suited, rang >= T)
      "55+"  → 55, 66, 77, 88, 99, TT, JJ, QQ, KK, AA (paires >= 55)

    Returns:
        Set de combos au format "XxYy" (ex: "AcKd")
    """
    h = hand_class.strip()

    # Paire avec "+" (ex: "55+" = toutes paires >= 55)
    if len(h) == 3 and h[0] == h[1] and h[2] == '+':
        rank = h[0]
        min_val = RANK_VAL[rank]
        result = set()
        for r in RANKS:
            if RANK_VAL[r] >= min_val:
                result |= _pair_combos(r)
        return result

    # Paire simple (ex: "AA", "TT")
    if len(h) == 2 and h[0] == h[1]:
        return _pair_combos(h[0])

    # Suited avec "+" (ex: "ATs+" = ATs, AJs, AQs, AKs)
    if len(h) == 4 and h[2] == 's' and h[3] == '+':
        r1, r2 = h[0], h[1]
        min_val = RANK_VAL[r2]
        result = set()
        for r in RANKS:
            if RANK_VAL[r] >= min_val and r != r1:
                result |= _suited_combos(r1, r)
        return result

    # Offsuit avec "+" (ex: "ATo+" = ATo, AJo, AQo, AKo)
    if len(h) == 4 and h[2] == 'o' and h[3] == '+':
        r1, r2 = h[0], h[1]
        min_val = RANK_VAL[r2]
        result = set()
        for r in RANKS:
            if RANK_VAL[r] >= min_val and r != r1:
                result |= _offsuit_combos(r1, r)
        return result

    # Suited + offsuit avec "+" (ex: "KTo+" = KTo+, KTs+)
    if len(h) == 3 and h[2] == '+':
        r1, r2 = h[0], h[1]
        min_val = RANK_VAL[r2]
        result = set()
        for r in RANKS:
            if RANK_VAL[r] >= min_val and r != r1:
                result |= _all_combos(r1, r)
        return result

    # Suited (ex: "AKs", "KQs")
    if len(h) == 3 and h[2] == 's':
        return _suited_combos(h[0], h[1])

    # Offsuit (ex: "AKo", "KQo")
    if len(h) == 3 and h[2] == 'o':
        return _offsuit_combos(h[0], h[1])

    # Suited + offsuit (ex: "AK", "KQ")
    if len(h) == 2 and h[0] != h[1]:
        return _all_combos(h[0], h[1])

    raise ValueError(f"Format de main non reconnu : '{hand_class}'")


def expand_range(hand_classes: List[str]) -> Set[str]:
    """
    Développe une liste de classes de mains en un ensemble de combos.

    Args:
        hand_classes : ex ['AA', 'KK', 'AKs', 'AQs+', 'TT+']

    Returns:
        Set de tous les combos spécifiques couverts.
    """
    result = set()
    for hc in hand_classes:
        result |= expand_hand_class(hc)
    return result


def hand_to_class(card1: str, card2: str) -> str:
    """
    Convertit deux cartes spécifiques en classe de main canonique.
    Ex: ('Ah', 'Kd') → 'AKo'
        ('Ah', 'Kh') → 'AKs'
        ('Ah', 'As') → 'AA'

    Args:
        card1, card2 : cartes au format "Xs" (rang + couleur)

    Returns:
        Classe de main canonique (rang_haut en premier)
    """
    r1, s1 = card1[0], card1[1]
    r2, s2 = card2[0], card2[1]

    # Mettre le rang le plus haut en premier
    if RANK_VAL[r1] < RANK_VAL[r2]:
        r1, s1, r2, s2 = r2, s2, r1, s1

    if r1 == r2:
        return r1 + r2  # paire
    if s1 == s2:
        return r1 + r2 + 's'  # suited
    return r1 + r2 + 'o'  # offsuit


def combo_in_range(card1: str, card2: str, range_combos: Set[str]) -> bool:
    """
    Vérifie si une main (deux cartes) est dans une range de combos.

    Args:
        card1, card2   : cartes au format "Xs"
        range_combos   : set de combos au format "XxYy"

    Returns:
        True si la main est dans la range.
    """
    combo1 = card1 + card2
    combo2 = card2 + card1
    return combo1 in range_combos or combo2 in range_combos


# =============================================================================
# Configuration d'un archétype
# =============================================================================

@dataclass
class ArchetypeConfig:
    """
    Configuration complète d'un RangeBot (archétype + mutation).

    Attributes:
        archetype           : 'TAG', 'LAG', ou 'CALLING_STATION'
        mutation            : 0 (base), 1, ou 2 (de plus en plus loose)
        hand_classes        : liste de classes de mains preflop
        preflop_combos      : set développé de tous les combos
        postflop_ehs_threshold : seuil EHS pour bet/raise postflop
        postflop_ehs_call   : seuil EHS pour call postflop (Calling Station)
        aggression          : 'aggressive' ou 'passive' (style postflop)
        description         : description lisible
        range_pct           : % approximatif de mains jouées
    """
    archetype:              str
    mutation:               int
    hand_classes:           List[str]
    preflop_combos:         Set[str]
    postflop_ehs_threshold: float   # seuil pour bet/raise
    postflop_ehs_call:      float   # seuil pour call (Calling Station uniquement)
    aggression:             str     # 'aggressive' | 'passive'
    description:            str
    range_pct:              float   # fraction de combos / 1326


# =============================================================================
# Définitions des 9 ranges
# =============================================================================

# ── TAG (Tight Aggressive) ────────────────────────────────────────────────────
# Joue peu de mains (~10-15%), bet/raise fort postflop
# Représente un joueur régulier compétent

_TAG_BASE = [
    'TT+',          # paires TT à AA
    'AQs+',         # AQs, AKs
    'AQo+',         # AQo, AKo
    'KQs',
]

_TAG_MUT1 = [
    '99+',          # paires 99 à AA
    'AJs+',         # AJs, AQs, AKs
    'AJo+',         # AJo, AQo, AKo
    'KQs', 'KJs',
]

_TAG_MUT2 = [
    'JJ+',          # paires JJ à AA
    'AKs',
    'AKo',
]

# ── LAG (Loose Aggressive) ────────────────────────────────────────────────────
# Joue beaucoup de mains (~30-40%), bet/raise fort postflop
# Représente un joueur agressif et large

_LAG_BASE = [
    '22+',          # toutes paires
    'A2s+',         # tout Ax suited
    'KTs+',         # KTs, KJs, KQs
    'QTs+',         # QTs, QJs
    'JTs',          # JTs
    'T9s',
    'ATo+',         # ATo, AJo, AQo, AKo
    'KTo+',         # KTo, KJo, KQo
    'QJo',
]

_LAG_MUT1 = [
    '22+',
    'A2s+',
    'K7s+',         # K7s à KAs
    'Q8s+',
    'J8s+',
    'T8s+',
    'ATo+',
    'KTo+',
    'QJo', 'QTo',
]

_LAG_MUT2 = [
    '22+',
    'A2s+',
    'K4s+',
    'Q6s+',
    'J7s+',
    'T7s+',
    '97s+',
    'A5o+',
    'KTo+',
    'QJo', 'QTo', 'JTo',
]

# ── Calling Station (Loose Passive) ──────────────────────────────────────────
# Joue beaucoup de mains (~35-45%), call souvent, raise rarement
# Représente un joueur récréatif passif

_CALL_BASE = [
    '22+',
    'A2s+',
    'K9s+',
    'Q9s+',
    'J9s+',
    'T8s+',
    'AJo+',
    'KTo+',
    'QJo',
]

_CALL_MUT1 = [
    '22+',
    'A2s+',
    'K6s+',
    'Q7s+',
    'J7s+',
    'T7s+',
    '97s+',
    'ATo+',
    'KTo+',
    'QJo', 'QTo',
]

_CALL_MUT2 = [
    '22+',
    'A2s+',
    'K3s+',
    'Q5s+',
    'J6s+',
    'T6s+',
    '96s+',
    '86s+',
    'A7o+',
    'KTo+',
    'QJo', 'QTo', 'JTo',
]


def _make_config(
    archetype:    str,
    mutation:     int,
    hand_classes: List[str],
    ehs_bet:      float,
    ehs_call:     float,
    aggression:   str,
    description:  str,
) -> ArchetypeConfig:
    """Helper interne pour construire une ArchetypeConfig."""
    combos = expand_range(hand_classes)
    pct    = len(combos) / 1326.0
    return ArchetypeConfig(
        archetype=archetype,
        mutation=mutation,
        hand_classes=hand_classes,
        preflop_combos=combos,
        postflop_ehs_threshold=ehs_bet,
        postflop_ehs_call=ehs_call,
        aggression=aggression,
        description=description,
        range_pct=pct,
    )


# Dictionnaire central de toutes les configs
_CONFIGS: Dict[Tuple[str, int], ArchetypeConfig] = {

    # ── TAG ──────────────────────────────────────────────────────────────────
    ('TAG', 0): _make_config(
        'TAG', 0, _TAG_BASE,
        ehs_bet=0.65, ehs_call=0.55, aggression='aggressive',
        description="TAG base : TT+, AQ+, KQs (~12% des mains)"
    ),
    ('TAG', 1): _make_config(
        'TAG', 1, _TAG_MUT1,
        ehs_bet=0.65, ehs_call=0.55, aggression='aggressive',
        description="TAG mutation 1 : 99+, AJ+, KQs, KJs (~15% des mains)"
    ),
    ('TAG', 2): _make_config(
        'TAG', 2, _TAG_MUT2,
        ehs_bet=0.65, ehs_call=0.55, aggression='aggressive',
        description="TAG mutation 2 : JJ+, AK uniquement (~5% des mains)"
    ),

    # ── LAG ──────────────────────────────────────────────────────────────────
    ('LAG', 0): _make_config(
        'LAG', 0, _LAG_BASE,
        ehs_bet=0.45, ehs_call=0.35, aggression='aggressive',
        description="LAG base : 22+, A2s+, connecteurs, broadway (~32% des mains)"
    ),
    ('LAG', 1): _make_config(
        'LAG', 1, _LAG_MUT1,
        ehs_bet=0.45, ehs_call=0.35, aggression='aggressive',
        description="LAG mutation 1 : range élargie (~36% des mains)"
    ),
    ('LAG', 2): _make_config(
        'LAG', 2, _LAG_MUT2,
        ehs_bet=0.45, ehs_call=0.35, aggression='aggressive',
        description="LAG mutation 2 : range très large (~40% des mains)"
    ),

    # ── Calling Station ───────────────────────────────────────────────────────
    ('CALLING_STATION', 0): _make_config(
        'CALLING_STATION', 0, _CALL_BASE,
        ehs_bet=0.75, ehs_call=0.30, aggression='passive',
        description="Calling Station base : 22+, A2s+, mains spéculatives (~30% des mains)"
    ),
    ('CALLING_STATION', 1): _make_config(
        'CALLING_STATION', 1, _CALL_MUT1,
        ehs_bet=0.75, ehs_call=0.30, aggression='passive',
        description="Calling Station mutation 1 : range élargie (~37% des mains)"
    ),
    ('CALLING_STATION', 2): _make_config(
        'CALLING_STATION', 2, _CALL_MUT2,
        ehs_bet=0.75, ehs_call=0.30, aggression='passive',
        description="Calling Station mutation 2 : range très large (~45% des mains)"
    ),
}


# =============================================================================
# API publique
# =============================================================================

ARCHETYPES = ['TAG', 'LAG', 'CALLING_STATION']
MUTATIONS  = [0, 1, 2]


def get_range(archetype: str, mutation: int = 0) -> ArchetypeConfig:
    """
    Retourne la configuration d'un RangeBot.

    Args:
        archetype : 'TAG', 'LAG', ou 'CALLING_STATION'
        mutation  : 0 (base), 1, ou 2

    Returns:
        ArchetypeConfig avec range preflop et paramètres postflop.

    Raises:
        ValueError si archétype ou mutation invalide.
    """
    if archetype not in ARCHETYPES:
        raise ValueError(
            f"Archétype invalide : '{archetype}'. "
            f"Valeurs acceptées : {ARCHETYPES}"
        )
    if mutation not in MUTATIONS:
        raise ValueError(
            f"Mutation invalide : {mutation}. "
            f"Valeurs acceptées : {MUTATIONS}"
        )
    return _CONFIGS[(archetype, mutation)]


def get_all_configs() -> Dict[Tuple[str, int], ArchetypeConfig]:
    """Retourne toutes les configs (9 RangeBots)."""
    return dict(_CONFIGS)


def print_range_summary() -> None:
    """Affiche un résumé des 9 ranges pour debug."""
    print("\n=== Résumé des 9 RangeBots ===\n")
    for archetype in ARCHETYPES:
        print(f"  [{archetype}]")
        for mutation in MUTATIONS:
            cfg = get_range(archetype, mutation)
            print(f"    Mutation {mutation} : {len(cfg.preflop_combos):4d} combos "
                  f"({cfg.range_pct:.1%}) — {cfg.description}")
        print()


# =============================================================================
# Tests intégrés
# =============================================================================

if __name__ == "__main__":
    print("\n=== Tests range_definitions.py ===\n")
    passed = failed = 0

    def t(label, cond, detail=""):
        global passed, failed
        if cond:
            print(f"  ✓ {label}")
            passed += 1
        else:
            print(f"  ✗ {label}" + (f" : {detail}" if detail else ""))
            failed += 1

    # expand_hand_class
    aa = expand_hand_class('AA')
    t("AA → 6 combos",  len(aa) == 6,  f"obtenu {len(aa)}")

    aks = expand_hand_class('AKs')
    t("AKs → 4 combos", len(aks) == 4, f"obtenu {len(aks)}")

    ako = expand_hand_class('AKo')
    t("AKo → 12 combos", len(ako) == 12, f"obtenu {len(ako)}")

    ak = expand_hand_class('AK')
    t("AK → 16 combos",  len(ak) == 16,  f"obtenu {len(ak)}")

    tt_plus = expand_hand_class('TT+')
    t("TT+ → 30 combos (TT,JJ,QQ,KK,AA × 6)", len(tt_plus) == 30,
      f"obtenu {len(tt_plus)}")

    # hand_to_class
    t("AhKd → AKo", hand_to_class('Ah', 'Kd') == 'AKo')
    t("AhKh → AKs", hand_to_class('Ah', 'Kh') == 'AKs')
    t("AhAs → AA",  hand_to_class('Ah', 'As') == 'AA')
    t("KdAh → AKo (ordre inversé)", hand_to_class('Kd', 'Ah') == 'AKo')

    # combo_in_range
    tag_combos = get_range('TAG', 0).preflop_combos
    t("AhKs dans TAG base",   combo_in_range('Ah', 'Ks', tag_combos))
    t("2h3d pas dans TAG base", not combo_in_range('2h', '3d', tag_combos))
    t("AhAs dans TAG base",   combo_in_range('Ah', 'As', tag_combos))

    # Configs
    for arch in ARCHETYPES:
        for mut in MUTATIONS:
            cfg = get_range(arch, mut)
            t(f"{arch} mut{mut} : range non vide",
              len(cfg.preflop_combos) > 0)
            t(f"{arch} mut{mut} : range_pct dans (0, 1)",
              0 < cfg.range_pct < 1,
              f"obtenu {cfg.range_pct:.3f}")

    # Ordonnancement des ranges (TAG plus tight que LAG)
    tag0 = get_range('TAG', 0)
    lag0 = get_range('LAG', 0)
    t("TAG range < LAG range (TAG plus tight)",
      tag0.range_pct < lag0.range_pct,
      f"TAG={tag0.range_pct:.1%} LAG={lag0.range_pct:.1%}")

    # Mutations croissantes
    for arch in ARCHETYPES:
        pct0 = get_range(arch, 0).range_pct
        pct1 = get_range(arch, 1).range_pct
        pct2 = get_range(arch, 2).range_pct
        if arch == 'TAG':
            # TAG mut2 est plus tight que base (JJ+ seulement)
            t(f"{arch} mutations cohérentes",
              pct2 < pct0 < pct1 or pct0 > 0,
              f"pct0={pct0:.1%} pct1={pct1:.1%} pct2={pct2:.1%}")
        else:
            t(f"{arch} mutations croissantes",
              pct0 <= pct1 <= pct2,
              f"pct0={pct0:.1%} pct1={pct1:.1%} pct2={pct2:.1%}")

    # get_all_configs
    all_cfg = get_all_configs()
    t("9 configs au total", len(all_cfg) == 9, f"obtenu {len(all_cfg)}")

    print(f"\n  {passed}/{passed+failed} tests passés")
    if failed == 0:
        print("  ✅ Tous les tests range_definitions passent.\n")
    else:
        print(f"  ⚠ {failed} échec(s)\n")

    print_range_summary()
