"""
board_texture.py — Bucketing Dimension 3 : Texture du Board
Phase 2 — Bot Poker Académique

Classifie le board en 4 catégories selon la session 2 (section 1.2) :
  - SEC         : board sans draws (ex: A72 rainbow)
  - SEMI_CONNECTE: draws straights + flush possibles (ex: 89J two-tone)
  - MONOTONE    : flush draws omniprésents (ex: Qh9h4h)
  - PAIRE       : board pairé, ranges condensées (ex: KK7)

Usage:
    from core.board_texture import classify_board, BoardTexture
    texture = classify_board(["As", "7d", "2c"])
    print(texture)  # BoardTexture.SEC
"""

from enum import Enum, auto
from typing import List, Tuple


class BoardTexture(Enum):
    """Catégories de texture de board (Dimension 3 du bucketing dynamique)."""
    SEC          = "sec"           # Rainbow, peu de connectivité
    SEMI_CONNECTE = "semi_connecte" # Two-tone ou 2+ cartes connectées
    MONOTONE     = "monotone"      # 3+ cartes de même couleur
    PAIRE        = "paire"         # Board contient une paire ou brelan

    def __str__(self):
        return self.value

    def sizing_modifier(self) -> float:
        """
        Modificateur de sizing suggéré selon la texture.
        Retourne un multiplicateur appliqué au sizing de base de l'EHSBot.

        Rationale :
          - Boards drawy (semi-connecté, monotone) → bets plus gros pour denial of equity
          - Board sec → bets plus petits suffisent (range adversaire moins forte)
          - Board pairé → sizing modéré (ranges condensées des deux côtés)
        """
        modifiers = {
            BoardTexture.SEC:          0.90,
            BoardTexture.SEMI_CONNECTE: 1.10,
            BoardTexture.MONOTONE:      1.20,
            BoardTexture.PAIRE:         1.00,
        }
        return modifiers[self]

    def description(self) -> str:
        """Description lisible pour les logs."""
        descriptions = {
            BoardTexture.SEC:
                "Board sec (rainbow, peu de draws) — ranges polarisées, peu de draws",
            BoardTexture.SEMI_CONNECTE:
                "Board semi-connecté (two-tone ou cartes proches) — draws straights + flush présents",
            BoardTexture.MONOTONE:
                "Board monotone (3+ cartes même couleur) — flush draws omniprésents",
            BoardTexture.PAIRE:
                "Board pairé (paire ou brelan visible) — ranges condensées, moins de nuts pour l'agresseur",
        }
        return descriptions[self]


# ---------------------------------------------------------------------------
# Parsing des cartes
# ---------------------------------------------------------------------------

RANK_ORDER = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
              '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}


def parse_card(card_str: str) -> Tuple[int, str]:
    """
    Parse une carte du format "As" -> (14, 's').
    Retourne (rang_int, couleur_char).
    """
    if len(card_str) < 2:
        raise ValueError(f"Carte invalide : '{card_str}'")
    rank_char = card_str[0].upper()
    suit_char = card_str[1].lower()
    if rank_char not in RANK_ORDER:
        raise ValueError(f"Rang invalide : '{rank_char}' dans '{card_str}'")
    if suit_char not in ('c', 'd', 'h', 's'):
        raise ValueError(f"Couleur invalide : '{suit_char}' dans '{card_str}'")
    return RANK_ORDER[rank_char], suit_char


def parse_board(board: List[str]) -> Tuple[List[int], List[str]]:
    """
    Parse une liste de cartes.
    Retourne (rangs: List[int], couleurs: List[str]).
    """
    ranks, suits = [], []
    for card in board:
        r, s = parse_card(card)
        ranks.append(r)
        suits.append(s)
    return ranks, suits


# ---------------------------------------------------------------------------
# Détecteurs de propriétés du board
# ---------------------------------------------------------------------------

def _has_pair_or_better(ranks: List[int]) -> bool:
    """True si le board contient au moins une paire."""
    return len(ranks) != len(set(ranks))


def _suit_counts(suits: List[str]) -> dict:
    """Compte le nombre de cartes par couleur."""
    counts = {}
    for s in suits:
        counts[s] = counts.get(s, 0) + 1
    return counts


def _is_monotone(suits: List[str]) -> bool:
    """True si 3+ cartes de même couleur (flush draw possible ou complet)."""
    for count in _suit_counts(suits).values():
        if count >= 3:
            return True
    return False


def _is_two_tone(suits: List[str]) -> bool:
    """True si exactement 2 cartes de même couleur (sur flop de 3 cartes)."""
    for count in _suit_counts(suits).values():
        if count >= 2:
            return True
    return False


def _connectivity_score(ranks: List[int]) -> int:
    """
    Score de connectivité du board.
    Mesure le nombre de paires adjacentes (écart <= 4) entre les rangs.
    Score > 0 → board connecté.
    """
    sorted_ranks = sorted(set(ranks))  # dédupliqué pour ignorer les paires
    score = 0
    for i in range(len(sorted_ranks) - 1):
        gap = sorted_ranks[i + 1] - sorted_ranks[i]
        if gap <= 4:
            score += (5 - gap)  # plus l'écart est petit, plus le score est élevé
    return score


def _has_straight_draw(ranks: List[int]) -> bool:
    """
    True si le board contient au moins un tirage quinte (OESD ou gutshot).
    Détecte 3+ cartes dans une fenêtre de 5 rangs consécutifs.
    """
    unique_ranks = sorted(set(ranks))
    # Ajouter le cas As (rang 1) pour les tirages roue
    ranks_with_low_ace = unique_ranks[:]
    if 14 in ranks_with_low_ace:
        ranks_with_low_ace = [1] + ranks_with_low_ace

    for start in range(1, 11):  # fenêtres de 5 (A-5 à T-A)
        window = set(range(start, start + 5))
        overlap = len([r for r in ranks_with_low_ace if r in window])
        if overlap >= 3:
            return True
    return False


# ---------------------------------------------------------------------------
# Classificateur principal
# ---------------------------------------------------------------------------

def classify_board(board: List[str]) -> BoardTexture:
    """
    Classifie le board en une des 4 textures.

    Ordre de priorité (du plus restrictif au plus général) :
      1. PAIRE     : board pairé/triplé — condition la plus forte
      2. MONOTONE  : 3+ cartes de même couleur
      3. SEMI_CONNECTE : 2 cartes de même couleur OU connectivité élevée
      4. SEC       : défaut

    Args:
        board: Liste de cartes ex: ["As", "7d", "2c"]
               Peut être flop (3), turn (4) ou river (5).

    Returns:
        BoardTexture correspondant.

    Raises:
        ValueError si le board est vide ou une carte est invalide.
        ValueError si le board contient des doublons.

    Exemples:
        classify_board(["As", "7d", "2c"])          -> SEC
        classify_board(["8s", "9h", "Jd"])          -> SEMI_CONNECTE
        classify_board(["Qh", "9h", "4h"])          -> MONOTONE
        classify_board(["Ks", "Kd", "7h"])          -> PAIRE
    """
    if not board:
        raise ValueError("Le board ne peut pas être vide")
    if len(board) > 5:
        raise ValueError(f"Board trop long : {len(board)} cartes (max 5)")

    ranks, suits = parse_board(board)

    # Vérification doublons
    card_ints = [(r, s) for r, s in zip(ranks, suits)]
    if len(card_ints) != len(set(card_ints)):
        raise ValueError("Board contient des cartes dupliquées")

    # Priorité 1 : Board pairé
    if _has_pair_or_better(ranks):
        return BoardTexture.PAIRE

    # Priorité 2 : Monotone (3+ cartes même couleur)
    if _is_monotone(suits):
        return BoardTexture.MONOTONE

    # Priorité 3 : Semi-connecté
    # Conditions : two-tone OU straight draw OU connectivité élevée
    has_two_tone    = _is_two_tone(suits)
    has_str_draw    = _has_straight_draw(ranks)
    connectivity    = _connectivity_score(ranks)

    if has_two_tone or has_str_draw or connectivity >= 3:
        return BoardTexture.SEMI_CONNECTE

    # Défaut : Sec
    return BoardTexture.SEC


def get_board_texture_details(board: List[str]) -> dict:
    """
    Retourne des métriques détaillées sur la texture du board.
    Utile pour le debug et le tuning des seuils.

    Returns:
        dict avec texture, métriques brutes, et modificateur de sizing.
    """
    ranks, suits = parse_board(board)
    texture = classify_board(board)
    suit_count = _suit_counts(suits)
    max_suit_count = max(suit_count.values()) if suit_count else 0

    return {
        "texture":          texture,
        "texture_name":     texture.value,
        "description":      texture.description(),
        "sizing_modifier":  texture.sizing_modifier(),
        "has_pair":         _has_pair_or_better(ranks),
        "is_monotone":      _is_monotone(suits),
        "is_two_tone":      _is_two_tone(suits),
        "has_straight_draw": _has_straight_draw(ranks),
        "connectivity_score": _connectivity_score(ranks),
        "max_suit_count":   max_suit_count,
        "ranks_sorted":     sorted(ranks, reverse=True),
    }


# ---------------------------------------------------------------------------
# Tests unitaires intégrés
# ---------------------------------------------------------------------------

def _run_tests():
    """Tests de régression pour classify_board."""
    print("\n=== Tests BoardTexture ===")

    test_cases = [
        # (board, expected_texture, description)
        (["As", "7d", "2c"],       BoardTexture.SEC,           "Board sec classique"),
        (["Kh", "4d", "2c"],       BoardTexture.SEC,           "Board sec bas"),
        (["8s", "9h", "Jd"],       BoardTexture.SEMI_CONNECTE, "Straight draw (OESD)"),
        (["Td", "8h", "6s"],       BoardTexture.SEMI_CONNECTE, "Deux gaps (gutshots)"),
        (["Qh", "9h", "4h"],       BoardTexture.MONOTONE,      "Flush monotone"),
        (["Ah", "Kh", "2h", "7h"], BoardTexture.MONOTONE,      "4 cartes flush draw"),
        (["Ks", "Kd", "7h"],       BoardTexture.PAIRE,         "Board pairé"),
        (["7s", "7d", "7h"],       BoardTexture.PAIRE,         "Board triplé"),
        (["As", "Ac", "2h", "Kd"], BoardTexture.PAIRE,         "Paire sur turn"),
        (["Jh", "Td", "9c"],       BoardTexture.SEMI_CONNECTE, "Très connecté"),
        (["As", "7h", "2d", "Ks"], BoardTexture.SEMI_CONNECTE, "Two-tone turn sec"),
        (["Qh", "9h", "4h", "2s"], BoardTexture.MONOTONE,      "Monotone turn"),
    ]

    passed = 0
    failed = 0
    for board, expected, desc in test_cases:
        result = classify_board(board)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
            print(f"  {status} ÉCHEC [{desc}] : {board}")
            print(f"       Attendu={expected.value}, Obtenu={result.value}")
        if result == expected:
            print(f"  {status} {desc:45s} → {result.value}")

    # Tests des détails
    details = get_board_texture_details(["Jh", "Td", "9c"])
    assert details["connectivity_score"] >= 3, "Board JT9 très connecté"
    assert details["has_straight_draw"],        "Board JT9 a un straight draw"

    print(f"\n  Résultat : {passed}/{passed+failed} tests passés")
    if failed == 0:
        print("  ✅ Tous les tests BoardTexture passent.\n")
    else:
        print(f"  ⚠ {failed} test(s) en échec.\n")


if __name__ == "__main__":
    _run_tests()

    # Exemple d'affichage détaillé
    board = ["Jh", "Td", "9c"]
    details = get_board_texture_details(board)
    print(f"Board {board} :")
    for k, v in details.items():
        print(f"  {k:25s}: {v}")
