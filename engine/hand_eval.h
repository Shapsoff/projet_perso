#pragma once
#include <array>
#include <string>
#include <vector>
#include <cstdint>

// =============================================================================
// hand_eval.h — Evaluateur de main poker
//
// Principe : on encode chaque carte sur un uint32 (bits de rang + suit),
// puis on identifie la meilleure combinaison de 5 cartes parmi celles
// disponibles. Le score retourne est un entier comparable :
// plus il est eleve, plus la main est forte.
//
// Categories (bits 28-31 du score) :
//   8 = Quinte flush / Quinte flush royale
//   7 = Carre
//   6 = Full house
//   5 = Couleur (flush)
//   4 = Quinte (straight)
//   3 = Brelan
//   2 = Deux paires
//   1 = Paire
//   0 = Hauteur
//
// Nouveaute phase 2 :
//   evaluate(vector<Card>) — accepte 5, 6 ou 7 cartes sans padding.
//   Utilise par EHSCalculator pour evaluer flop (5), turn (6), river (7)
//   sans biaiser les calculs PPot/NPot.
// =============================================================================

namespace poker {

// ── Constantes ────────────────────────────────────────────────────────────────
static constexpr int NUM_RANKS = 13;
static constexpr int NUM_SUITS = 4;
static constexpr int DECK_SIZE = 52;

// Categories de mains
enum HandRank : uint32_t {
    HIGH_CARD       = 0,
    ONE_PAIR        = 1,
    TWO_PAIR        = 2,
    THREE_OF_A_KIND = 3,
    STRAIGHT        = 4,
    FLUSH           = 5,
    FULL_HOUSE      = 6,
    FOUR_OF_A_KIND  = 7,
    STRAIGHT_FLUSH  = 8,
};

// ── Encodage d'une carte ──────────────────────────────────────────────────────
// Bits [0..3]  = couleur (0-3)
// Bits [4..7]  = rang (0-12)
// Bits [8..20] = bitmask du rang (1 << rang) pour detection quinte/flush
using Card = uint32_t;

inline Card make_card(int rank, int suit) {
    return static_cast<uint32_t>((rank << 4) | suit | ((1 << rank) << 8));
}
inline int card_rank(Card c)     { return (c >> 4) & 0xF; }
inline int card_suit(Card c)     { return c & 0xF; }
inline int card_rank_bit(Card c) { return (c >> 8); }

// Parse une carte depuis une string : "As", "Kd", "Th", "2c"...
Card parse_card(const std::string& s);

// ── Score de main ─────────────────────────────────────────────────────────────
// bits [28..31] = categorie, bits [0..27] = kickers pour departage
using HandScore = uint32_t;

// ── HandEvaluator ─────────────────────────────────────────────────────────────

class HandEvaluator {
public:
    HandEvaluator() = default;

    // ── API phase 1 (inchangee) ───────────────────────────────────────────────

    // Evalue la meilleure main parmi 7 cartes (C(7,5) = 21 combinaisons)
    HandScore evaluate(const std::array<Card, 7>& cards) const;

    // Surcharge strings — requiert exactement 7 cartes
    HandScore evaluate(const std::vector<std::string>& cards) const;

    // ── API phase 2 (nouvelle) ────────────────────────────────────────────────

    // Evalue la meilleure main parmi N cartes (5, 6 ou 7).
    // Dispatche automatiquement :
    //   5 cartes (flop  : 2 main + 3 board) -> eval5  : C(5,5) = 1  combinaison
    //   6 cartes (turn  : 2 main + 4 board) -> eval6  : C(6,5) = 6  combinaisons
    //   7 cartes (river : 2 main + 5 board) -> eval7  : C(7,5) = 21 combinaisons
    // Leve std::invalid_argument si n < 5 ou n > 7.
    HandScore evaluate(const std::vector<Card>& cards) const;

    // Utilitaires
    static std::string rank_name(HandScore score);
    static HandRank    hand_rank(HandScore score);

private:
    // Evalue exactement 5 cartes
    HandScore eval5(const std::array<Card, 5>& c) const;

    // Evalue la meilleure main parmi 6 cartes (C(6,5) = 6 combinaisons)
    HandScore eval6(const std::array<Card, 6>& c) const;

    // Detecte flush dans N cartes — retourne la couleur dominante ou -1
    int detect_flush_suit(const std::vector<Card>& cards) const;

    // Detecte une quinte dans un bitmask de rangs
    // Retourne le rang le plus haut de la quinte, ou -1
    int detect_straight(uint32_t rank_bits) const;

    // Construit le score final depuis categorie + kickers
    static HandScore make_score(HandRank rank, uint32_t kickers);
};

} // namespace poker
