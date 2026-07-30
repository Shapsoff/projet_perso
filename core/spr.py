"""
spr.py — Bucketing Dimension 4 : Stack-to-Pot Ratio (SPR)
Phase 2 — Bot Poker Académique

Calcule le SPR et retourne une catégorie de commitment selon le tableau
défini en session 2, section 1.2.

SPR = stack_effectif / pot_actuel

Catégories :
  - TRES_BAS  : SPR < 2  → commitment quasi-obligatoire
  - BAS       : SPR 2–6  → jeu de sets et two-pair
  - MOYEN     : SPR 6–15 → draws et value bets multicouches
  - ELEVE     : SPR > 15 → bluffs, floats, positional play

Usage:
    from core.spr import calculate_spr, SPRCategory, get_spr_category
    spr = calculate_spr(stack_effectif=1200, pot=320)
    cat = get_spr_category(spr)
    print(cat)  # SPRCategory.MOYEN
"""

from enum import Enum
from typing import Optional
from dataclasses import dataclass


class SPRCategory(Enum):
    """
    Catégories SPR définissant l'engagement recommandé.
    Correspond à la Dimension 4 du bucketing dynamique.
    """
    TRES_BAS = "tres_bas"   # SPR < 2  : all-in quasi-obligatoire
    BAS      = "bas"        # SPR 2–6  : jeu value, bluffs chers
    MOYEN    = "moyen"      # SPR 6–15 : jeu de draws et layered bets
    ELEVE    = "eleve"      # SPR > 15 : jeu bluff, float, position

    def __str__(self):
        return self.value

    def description(self) -> str:
        descriptions = {
            SPRCategory.TRES_BAS:
                "SPR très bas (<2) — Commitment quasi-obligatoire avec tout draw ou top pair",
            SPRCategory.BAS:
                "SPR bas (2–6) — Jeu de sets et two-pair ; bluffs moins rentables",
            SPRCategory.MOYEN:
                "SPR moyen (6–15) — Jeu de draws et de value bets multicouches",
            SPRCategory.ELEVE:
                "SPR élevé (>15) — Jeu de bluffs, de floats et de positional play",
        }
        return descriptions[self]

    def fold_equity_matters(self) -> bool:
        """True si la fold equity est un facteur de décision important."""
        return self in (SPRCategory.MOYEN, SPRCategory.ELEVE)

    def commitment_threshold(self) -> float:
        """
        Seuil EHS à partir duquel on est en mode commitment (jouer pour le tout).
        Plus le SPR est bas, plus le seuil s'abaisse (on commit avec des mains moins fortes).
        """
        thresholds = {
            SPRCategory.TRES_BAS: 0.30,  # Commit avec n'importe quelle equity raisonnable
            SPRCategory.BAS:      0.50,  # Commit avec top pair+
            SPRCategory.MOYEN:    0.65,  # Commit avec main forte seulement
            SPRCategory.ELEVE:    0.80,  # Commit uniquement avec les nuts ou quasi-nuts
        }
        return thresholds[self]


# ---------------------------------------------------------------------------
# Calcul SPR
# ---------------------------------------------------------------------------

def calculate_spr(stack_effectif: float, pot: float) -> float:
    """
    Calcule le Stack-to-Pot Ratio.

    Le stack_effectif est le plus petit stack entre nous et l'adversaire principal
    (pour une table multiway, utiliser le plus petit stack actif).

    Args:
        stack_effectif : Stack en chips du joueur le plus court (ou notre stack si heads-up)
        pot            : Pot actuel en chips (avant notre action)

    Returns:
        SPR (float). Retourne float('inf') si pot == 0.

    Raises:
        ValueError si les valeurs sont négatives.
    """
    if stack_effectif < 0:
        raise ValueError(f"stack_effectif ne peut pas être négatif : {stack_effectif}")
    if pot < 0:
        raise ValueError(f"pot ne peut pas être négatif : {pot}")
    if pot == 0:
        return float('inf')  # Pas encore de pot (rare en pratique)
    return stack_effectif / pot


def get_spr_category(spr: float) -> SPRCategory:
    """
    Retourne la catégorie SPR pour un SPR donné.

    Seuils session 2 :
        < 2   → TRES_BAS
        2–6   → BAS
        6–15  → MOYEN
        > 15  → ELEVE
    """
    if spr < 2.0:
        return SPRCategory.TRES_BAS
    elif spr < 6.0:
        return SPRCategory.BAS
    elif spr < 15.0:
        return SPRCategory.MOYEN
    else:
        return SPRCategory.ELEVE


def get_spr_from_game_state(game_state: dict) -> float:
    """
    Extrait le SPR depuis un game_state normalisé (format session 2, section 4.4).

    Le stack_effectif est calculé comme le minimum entre :
      - Notre stack
      - Le plus petit stack des adversaires actifs

    Args:
        game_state: dict au format défini dans le document (section 4.4)

    Returns:
        SPR calculé.
    """
    our_stack = game_state.get('stack', 0)
    pot       = game_state.get('pot', 1)
    players   = game_state.get('players', [])

    # Stack effectif = min(notre stack, plus petit stack adverse actif)
    if players:
        opp_stacks = [p['stack'] for p in players if p.get('stack', 0) > 0]
        if opp_stacks:
            stack_effectif = min(our_stack, min(opp_stacks))
        else:
            stack_effectif = our_stack
    else:
        stack_effectif = our_stack

    return calculate_spr(stack_effectif, pot)


# ---------------------------------------------------------------------------
# Dataclass résumé SPR
# ---------------------------------------------------------------------------

@dataclass
class SPRInfo:
    """
    Résumé complet du contexte SPR pour une décision.
    Utilisé par l'EHSBot pour modular sa stratégie.
    """
    spr:              float
    category:         SPRCategory
    stack_effectif:   float
    pot:              float

    # Flags stratégiques dérivés
    force_commit:     bool   = False  # True si SPR < 2 (commitment forcé)
    bluff_viable:     bool   = False  # True si fold equity raisonnable
    protect_equity:   bool   = False  # True si SPR moyen (draws adverses rentables)

    def __post_init__(self):
        self.force_commit   = self.category == SPRCategory.TRES_BAS
        self.bluff_viable   = self.category.fold_equity_matters()
        self.protect_equity = self.category in (SPRCategory.BAS, SPRCategory.MOYEN)

    def __str__(self):
        return (f"SPR={self.spr:.2f} [{self.category.value}] "
                f"| commit={self.force_commit} bluff={self.bluff_viable} "
                f"protect={self.protect_equity}")

    def sizing_modifier(self) -> float:
        """
        Modificateur de sizing selon le SPR.
        SPR bas → bets relatifs plus grands (pour maximiser EV commitment)
        SPR élevé → sizing variable (bluff + value)
        """
        modifiers = {
            SPRCategory.TRES_BAS: 1.50,   # Overbet / shove
            SPRCategory.BAS:      1.20,   # Bets larges pour value
            SPRCategory.MOYEN:    1.00,   # Sizing standard
            SPRCategory.ELEVE:    0.85,   # Bets plus petits (bluff cheaper)
        }
        return modifiers[self.category]


def get_spr_info(stack_effectif: float, pot: float) -> SPRInfo:
    """
    Retourne un SPRInfo complet depuis les valeurs brutes.
    Raccourci pratique pour l'EHSBot.
    """
    spr = calculate_spr(stack_effectif, pot)
    cat = get_spr_category(spr)
    return SPRInfo(spr=spr, category=cat, stack_effectif=stack_effectif, pot=pot)


def get_spr_info_from_game_state(game_state: dict) -> SPRInfo:
    """Version depuis game_state normalisé."""
    our_stack = game_state.get('stack', 0)
    pot       = game_state.get('pot', 1)
    players   = game_state.get('players', [])

    if players:
        opp_stacks = [p['stack'] for p in players if p.get('stack', 0) > 0]
        stack_effectif = min(our_stack, min(opp_stacks)) if opp_stacks else our_stack
    else:
        stack_effectif = our_stack

    return get_spr_info(stack_effectif, pot)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _run_tests():
    print("\n=== Tests SPR ===")

    # Test calculs SPR
    assert calculate_spr(1200, 320) == pytest_approx(3.75, 0.01), "SPR 1200/320"
    assert calculate_spr(100, 200) == 0.5,  "SPR 100/200 = 0.5"
    assert calculate_spr(0,   100) == 0.0,  "SPR 0/100 = 0 (allin)"
    assert calculate_spr(100, 0)   == float('inf'), "SPR /0 = inf"

    # Test catégories
    assert get_spr_category(0.5)  == SPRCategory.TRES_BAS
    assert get_spr_category(1.9)  == SPRCategory.TRES_BAS
    assert get_spr_category(2.0)  == SPRCategory.BAS
    assert get_spr_category(5.9)  == SPRCategory.BAS
    assert get_spr_category(6.0)  == SPRCategory.MOYEN
    assert get_spr_category(14.9) == SPRCategory.MOYEN
    assert get_spr_category(15.0) == SPRCategory.ELEVE
    assert get_spr_category(100)  == SPRCategory.ELEVE

    print("  ✓ Tous les calculs SPR corrects")

    # Test SPRInfo
    info = get_spr_info(100, 200)
    assert info.force_commit == True,  "SPR 0.5 → force_commit"
    assert info.bluff_viable == False, "SPR très bas → pas de bluff viable"

    info2 = get_spr_info(1500, 100)
    assert info2.force_commit == False, "SPR 15 → pas de force_commit"
    assert info2.bluff_viable == True,  "SPR élevé → bluff viable"
    print("  ✓ Flags SPRInfo corrects")

    # Test depuis game_state
    gs = {
        'stack': 1200,
        'pot':   320,
        'players': [
            {'id': 1, 'stack': 900},
            {'id': 2, 'stack': 1400},
        ]
    }
    info3 = get_spr_info_from_game_state(gs)
    # stack_effectif = min(1200, 900, 1400) = 900
    # SPR = 900/320 ≈ 2.81 → BAS
    assert info3.category == SPRCategory.BAS, f"Attendu BAS, obtenu {info3.category}"
    print(f"  ✓ SPRInfo depuis game_state : {info3}")

    print("  ✅ Tous les tests SPR passent.\n")


def pytest_approx(val, rel):
    """Micro-helper pour tests sans pytest."""
    class _Approx:
        def __init__(self, v, r): self.v, self.r = v, r
        def __eq__(self, other): return abs(other - self.v) / max(abs(self.v), 1e-10) <= self.r
    return _Approx(val, rel)


if __name__ == "__main__":
    _run_tests()

    # Exemples
    print("Exemples SPR :")
    for stack, pot in [(100, 200), (600, 200), (1200, 200), (3000, 200)]:
        info = get_spr_info(stack, pot)
        print(f"  stack={stack:5d} pot={pot} → {info}")
