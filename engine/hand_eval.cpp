#include "hand_eval.h"
#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace poker {

// =============================================================================
// Parse carte depuis string
// =============================================================================

Card parse_card(const std::string& s) {
    if (s.size() != 2)
        throw std::invalid_argument("Carte invalide : " + s);

    int rank;
    switch (s[0]) {
        case '2': rank = 0;  break;
        case '3': rank = 1;  break;
        case '4': rank = 2;  break;
        case '5': rank = 3;  break;
        case '6': rank = 4;  break;
        case '7': rank = 5;  break;
        case '8': rank = 6;  break;
        case '9': rank = 7;  break;
        case 'T': rank = 8;  break;
        case 'J': rank = 9;  break;
        case 'Q': rank = 10; break;
        case 'K': rank = 11; break;
        case 'A': rank = 12; break;
        default:
            throw std::invalid_argument("Rang invalide : " + s);
    }

    int suit;
    switch (s[1]) {
        case 'c': suit = 0; break;
        case 'd': suit = 1; break;
        case 'h': suit = 2; break;
        case 's': suit = 3; break;
        default:
            throw std::invalid_argument("Couleur invalide : " + s);
    }

    return make_card(rank, suit);
}

// =============================================================================
// Helpers internes
// =============================================================================

HandScore HandEvaluator::make_score(HandRank rank, uint32_t kickers) {
    return (static_cast<uint32_t>(rank) << 28) | (kickers & 0x0FFFFFFF);
}

HandRank HandEvaluator::hand_rank(HandScore score) {
    return static_cast<HandRank>(score >> 28);
}

std::string HandEvaluator::rank_name(HandScore score) {
    switch (hand_rank(score)) {
        case HIGH_CARD:       return "Hauteur";
        case ONE_PAIR:        return "Paire";
        case TWO_PAIR:        return "Deux paires";
        case THREE_OF_A_KIND: return "Brelan";
        case STRAIGHT:        return "Quinte";
        case FLUSH:           return "Couleur";
        case FULL_HOUSE:      return "Full house";
        case FOUR_OF_A_KIND:  return "Carre";
        case STRAIGHT_FLUSH:  return "Quinte flush";
        default:              return "Inconnu";
    }
}

// =============================================================================
// detect_straight — identique a l'original, factorise ici pour eval5/eval6/eval7
// =============================================================================

int HandEvaluator::detect_straight(uint32_t rank_bits) const {
    for (int top = 12; top >= 4; top--) {
        uint32_t mask = 0x1F << (top - 4);
        if ((rank_bits & mask) == mask)
            return top;
    }
    // Cas roue : A(12) + 2(0) + 3(1) + 4(2) + 5(3)
    if ((rank_bits & 0x100F) == 0x100F)
        return 3;
    return -1;
}

// =============================================================================
// detect_flush_suit — version generique sur vector<Card>
// Retourne la couleur si >= 5 cartes de meme couleur, sinon -1
// =============================================================================

int HandEvaluator::detect_flush_suit(const std::vector<Card>& cards) const {
    int counts[4] = {0, 0, 0, 0};
    for (auto& c : cards) counts[card_suit(c)]++;
    for (int s = 0; s < 4; s++)
        if (counts[s] >= 5) return s;
    return -1;
}

// =============================================================================
// eval5 — evalue exactement 5 cartes
// Noyau commun utilise par eval6 et evaluate(array<Card,7>)
// =============================================================================

HandScore HandEvaluator::eval5(const std::array<Card, 5>& c) const {
    int counts[13] = {};
    uint32_t rank_bits = 0;
    bool all_same_suit = true;
    int first_suit = card_suit(c[0]);

    for (auto& card : c) {
        counts[card_rank(card)]++;
        rank_bits |= (1u << card_rank(card));
        if (card_suit(card) != first_suit) all_same_suit = false;
    }

    std::array<std::pair<int,int>, 13> rc;
    int n = 0;
    for (int r = 12; r >= 0; r--)
        if (counts[r] > 0) rc[n++] = {r, counts[r]};

    std::sort(rc.begin(), rc.begin() + n, [](const auto& a, const auto& b){
        return a.second != b.second ? a.second > b.second : a.first > b.first;
    });

    int  straight_top = detect_straight(rank_bits);
    bool is_flush     = all_same_suit;
    bool is_straight  = (straight_top >= 0);

    if (is_straight && is_flush) {
        return make_score(STRAIGHT_FLUSH, static_cast<uint32_t>(straight_top));
    }
    if (rc[0].second == 4) {
        return make_score(FOUR_OF_A_KIND,
            (static_cast<uint32_t>(rc[0].first) << 4) | static_cast<uint32_t>(rc[1].first));
    }
    if (rc[0].second == 3 && rc[1].second == 2) {
        return make_score(FULL_HOUSE,
            (static_cast<uint32_t>(rc[0].first) << 4) | static_cast<uint32_t>(rc[1].first));
    }
    if (is_flush) {
        uint32_t kickers = 0;
        for (int i = 0; i < n; i++)
            kickers = (kickers << 4) | static_cast<uint32_t>(rc[i].first);
        return make_score(FLUSH, kickers);
    }
    if (is_straight) {
        return make_score(STRAIGHT, static_cast<uint32_t>(straight_top));
    }
    if (rc[0].second == 3) {
        return make_score(THREE_OF_A_KIND,
            (static_cast<uint32_t>(rc[0].first) << 8) |
            (static_cast<uint32_t>(rc[1].first) << 4) |
             static_cast<uint32_t>(rc[2].first));
    }
    if (rc[0].second == 2 && rc[1].second == 2) {
        return make_score(TWO_PAIR,
            (static_cast<uint32_t>(rc[0].first) << 8) |
            (static_cast<uint32_t>(rc[1].first) << 4) |
             static_cast<uint32_t>(rc[2].first));
    }
    if (rc[0].second == 2) {
        return make_score(ONE_PAIR,
            (static_cast<uint32_t>(rc[0].first) << 12) |
            (static_cast<uint32_t>(rc[1].first) << 8)  |
            (static_cast<uint32_t>(rc[2].first) << 4)  |
             static_cast<uint32_t>(rc[3].first));
    }
    uint32_t kickers = 0;
    for (int i = 0; i < n; i++)
        kickers = (kickers << 4) | static_cast<uint32_t>(rc[i].first);
    return make_score(HIGH_CARD, kickers);
}

// =============================================================================
// eval6 — meilleure main parmi C(6,5) = 6 combinaisons
// Utilise pour evaluer la turn (2 main + 4 board) sans padding
// =============================================================================

HandScore HandEvaluator::eval6(const std::array<Card, 6>& cards) const {
    // Les 6 combinaisons de 5 parmi 6 : on exclut chaque carte une fois
    static const int combos[6][5] = {
        {1,2,3,4,5},  // exclut index 0
        {0,2,3,4,5},  // exclut index 1
        {0,1,3,4,5},  // exclut index 2
        {0,1,2,4,5},  // exclut index 3
        {0,1,2,3,5},  // exclut index 4
        {0,1,2,3,4},  // exclut index 5
    };

    HandScore best = 0;
    for (auto& combo : combos) {
        std::array<Card, 5> hand = {
            cards[combo[0]], cards[combo[1]], cards[combo[2]],
            cards[combo[3]], cards[combo[4]]
        };
        HandScore s = eval5(hand);
        if (s > best) best = s;
    }
    return best;
}

// =============================================================================
// evaluate(array<Card,7>) — API phase 1, inchangee
// Meilleure main parmi C(7,5) = 21 combinaisons
// =============================================================================

HandScore HandEvaluator::evaluate(const std::array<Card, 7>& cards) const {
    static const int combos[21][5] = {
        {0,1,2,3,4}, {0,1,2,3,5}, {0,1,2,3,6},
        {0,1,2,4,5}, {0,1,2,4,6}, {0,1,2,5,6},
        {0,1,3,4,5}, {0,1,3,4,6}, {0,1,3,5,6},
        {0,1,4,5,6}, {0,2,3,4,5}, {0,2,3,4,6},
        {0,2,3,5,6}, {0,2,4,5,6}, {0,3,4,5,6},
        {1,2,3,4,5}, {1,2,3,4,6}, {1,2,3,5,6},
        {1,2,4,5,6}, {1,3,4,5,6}, {2,3,4,5,6},
    };

    HandScore best = 0;
    for (auto& combo : combos) {
        std::array<Card, 5> hand = {
            cards[combo[0]], cards[combo[1]], cards[combo[2]],
            cards[combo[3]], cards[combo[4]]
        };
        HandScore s = eval5(hand);
        if (s > best) best = s;
    }
    return best;
}

// =============================================================================
// evaluate(vector<string>) — API phase 1, inchangee, requiert 7 cartes
// =============================================================================

HandScore HandEvaluator::evaluate(const std::vector<std::string>& cards) const {
    if (cards.size() != 7)
        throw std::invalid_argument("evaluate(strings) requiert exactement 7 cartes");
    std::array<Card, 7> c;
    for (int i = 0; i < 7; i++) c[i] = parse_card(cards[i]);
    return evaluate(c);
}

// =============================================================================
// evaluate(vector<Card>) — API phase 2 : 5, 6 ou 7 cartes
//
// Dispatche selon la taille :
//   5 -> eval5  directement (flop : 2 main + 3 board)
//   6 -> eval6  C(6,5) = 6  (turn : 2 main + 4 board)
//   7 -> eval7  C(7,5) = 21 (river : 2 main + 5 board)
// =============================================================================

HandScore HandEvaluator::evaluate(const std::vector<Card>& cards) const {
    switch (cards.size()) {
        case 5: {
            std::array<Card, 5> c = {cards[0], cards[1], cards[2], cards[3], cards[4]};
            return eval5(c);
        }
        case 6: {
            std::array<Card, 6> c = {cards[0], cards[1], cards[2],
                                     cards[3], cards[4], cards[5]};
            return eval6(c);
        }
        case 7: {
            std::array<Card, 7> c = {cards[0], cards[1], cards[2], cards[3],
                                     cards[4], cards[5], cards[6]};
            return evaluate(c);
        }
        default:
            throw std::invalid_argument(
                "evaluate(vector) : nombre de cartes invalide (attendu 5, 6 ou 7)");
    }
}

} // namespace poker
