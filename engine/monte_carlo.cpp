#include "monte_carlo.h"
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <bitset>

// Remplacement portable de __builtin_popcountll
static inline int popcount64(uint64_t x) {
    return static_cast<int>(std::bitset<64>(x).count());
}

namespace poker {

MonteCarloEngine::MonteCarloEngine(uint64_t seed) : rng_(seed) {}

// ── Deck disponible ───────────────────────────────────────────────────────────
std::vector<Card> MonteCarloEngine::build_available_deck(
    const std::vector<Card>& known_cards) const
{
    uint64_t mask = 0;
    for (auto& k : known_cards)
        mask |= (1ULL << (card_rank(k) * 4 + card_suit(k)));

    std::vector<Card> deck;
    deck.reserve(52 - popcount64(mask));
    for (int r = 0; r < 13; r++)
        for (int s = 0; s < 4; s++)
            if (!((mask >> (r*4+s)) & 1))
                deck.push_back(make_card(r, s));
    return deck;
}

// ── Structure combo pré-compilé ───────────────────────────────────────────────
struct CompiledCombo { Card c1, c2; float weight; };

static std::vector<CompiledCombo> compile_range(const PlayerRange& range) {
    std::vector<CompiledCombo> out;
    out.reserve(range.size());
    for (auto& [s, w] : range) {
        if (s.size() != 4 || w <= 0.0) continue;
        try {
            Card c1 = parse_card(s.substr(0,2));
            Card c2 = parse_card(s.substr(2,2));
            out.push_back({c1, c2, static_cast<float>(w)});
        } catch(...) {}
    }
    return out;
}

// ── Tire un combo depuis une range ────────────────────────────────────────────
std::pair<Card,Card> MonteCarloEngine::sample_combo(
    const PlayerRange& range,
    const std::vector<Card>& used_cards,
    std::mt19937_64& rng) const
{
    static thread_local std::vector<std::pair<std::pair<Card,Card>,float>> valid;
    valid.clear();

    uint64_t used_mask = 0;
    for (auto& u : used_cards)
        used_mask |= (1ULL << (card_rank(u)*4 + card_suit(u)));

    auto compiled = compile_range(range);
    float total = 0.f;
    for (auto& cc : compiled) {
        if ((used_mask >> (card_rank(cc.c1)*4+card_suit(cc.c1))) & 1) continue;
        if ((used_mask >> (card_rank(cc.c2)*4+card_suit(cc.c2))) & 1) continue;
        valid.push_back({{cc.c1, cc.c2}, cc.weight});
        total += cc.weight;
    }

    if (valid.empty() || total <= 0.f)
        return {INVALID_CARD, INVALID_CARD};

    std::uniform_real_distribution<float> dist(0.f, total);
    float pick = dist(rng);
    float cumul = 0.f;
    for (auto& [combo, w] : valid) {
        cumul += w;
        if (pick <= cumul) return combo;
    }
    return valid.back().first;
}

// ── Calcul d'équité multiway ──────────────────────────────────────────────────
EquityResult MonteCarloEngine::compute(
    const std::vector<std::string>& our_hand_str,
    const std::vector<std::string>& board_str,
    const std::vector<PlayerRange>&  opp_ranges,
    int n_sims)
{
    if (our_hand_str.size() != 2)
        throw std::invalid_argument("our_hand doit contenir exactement 2 cartes");
    if (board_str.size() > 5)
        throw std::invalid_argument("board ne peut pas depasser 5 cartes");

    Card our_c1 = parse_card(our_hand_str[0]);
    Card our_c2 = parse_card(our_hand_str[1]);
    std::vector<Card> board_fixed;
    for (auto& s : board_str) board_fixed.push_back(parse_card(s));

    int n_board_missing = 5 - static_cast<int>(board_str.size());

    EquityResult result;
    result.n = 0;

    for (int sim = 0; sim < n_sims; sim++) {
        std::vector<Card> used = {our_c1, our_c2};
        used.insert(used.end(), board_fixed.begin(), board_fixed.end());

        std::vector<std::pair<Card,Card>> opp_hands;
        bool valid_sim = true;

        for (auto& range : opp_ranges) {
            auto [c1, c2] = sample_combo(range, used, rng_);
            if (c1 == INVALID_CARD) { valid_sim = false; break; }
            opp_hands.push_back({c1, c2});
            used.push_back(c1);
            used.push_back(c2);
        }
        if (!valid_sim) continue;

        std::vector<Card> avail = build_available_deck(used);
        if (static_cast<int>(avail.size()) < n_board_missing) continue;

        std::vector<Card> board_complete = board_fixed;
        for (int i = 0; i < n_board_missing; i++) {
            std::uniform_int_distribution<int> d(i, static_cast<int>(avail.size()) - 1);
            int j = d(rng_);
            std::swap(avail[i], avail[j]);
            board_complete.push_back(avail[i]);
        }

        std::array<Card, 7> our_7 = {
            our_c1, our_c2,
            board_complete[0], board_complete[1], board_complete[2],
            board_complete[3], board_complete[4]
        };
        HandScore our_score = evaluator_.evaluate(our_7);

        std::vector<HandScore> opp_scores;
        for (auto& [c1, c2] : opp_hands) {
            std::array<Card, 7> opp_7 = {
                c1, c2,
                board_complete[0], board_complete[1], board_complete[2],
                board_complete[3], board_complete[4]
            };
            opp_scores.push_back(evaluator_.evaluate(opp_7));
        }

        HandScore best_opp = *std::max_element(opp_scores.begin(), opp_scores.end());

        result.n++;
        if (our_score > best_opp) {
            result.win += 1.0;
        } else if (our_score == best_opp) {
            int tied = 1;
            for (auto s : opp_scores) if (s == our_score) tied++;
            result.tie += 1.0 / tied;
        } else {
            result.loss += 1.0;
        }
    }

    if (result.n > 0) {
        double inv = 1.0 / result.n;
        result.win  *= inv;
        result.tie  *= inv;
        result.loss *= inv;
        double n_opp = static_cast<double>(opp_ranges.size());
        result.ev = result.win + (n_opp > 0 ? result.tie / n_opp : result.tie);
    }

    return result;
}

// ── Version optimisée vs range aléatoire ──────────────────────────────────────
//
// OPTIMISATION PRINCIPALE :
// L'ancienne version construisait une PlayerRange de 1326 combos pour chaque
// adversaire, puis appelait sample_combo() qui recompilait et filtrait ces 1326
// entrées à chaque simulation → O(1326 × n_sims) allocations et itérations.
//
// La nouvelle version tire directement 2 cartes dans le deck disponible via un
// Fisher-Yates partiel sur un tableau pré-alloué. C'est O(1) par tirage.
// On évite aussi toute allocation heap dans la boucle chaude.
//
EquityResult MonteCarloEngine::compute_vs_random(
    const std::vector<std::string>& our_hand,
    const std::vector<std::string>& board,
    int n_opponents,
    int n_sims)
{
    if (our_hand.size() != 2)
        throw std::invalid_argument("our_hand doit contenir exactement 2 cartes");
    if (board.size() > 5)
        throw std::invalid_argument("board ne peut pas depasser 5 cartes");

    Card our_c1 = parse_card(our_hand[0]);
    Card our_c2 = parse_card(our_hand[1]);

    std::vector<Card> board_fixed;
    board_fixed.reserve(board.size());
    for (auto& s : board) board_fixed.push_back(parse_card(s));

    int n_board_missing = 5 - static_cast<int>(board_fixed.size());

    // Bitmask des cartes connues au départ (our hand + board fixe)
    uint64_t base_mask = 0;
    base_mask |= (1ULL << (card_rank(our_c1)*4 + card_suit(our_c1)));
    base_mask |= (1ULL << (card_rank(our_c2)*4 + card_suit(our_c2)));
    for (auto& c : board_fixed)
        base_mask |= (1ULL << (card_rank(c)*4 + card_suit(c)));

    // Deck de base : toutes les cartes sauf les nôtres et le board
    // On le construit une seule fois et on le réutilise par copie à chaque sim
    std::vector<Card> base_deck;
    base_deck.reserve(52 - popcount64(base_mask));
    for (int r = 0; r < 13; r++)
        for (int s = 0; s < 4; s++)
            if (!((base_mask >> (r*4+s)) & 1))
                base_deck.push_back(make_card(r, s));

    // Buffers pré-alloués pour éviter toute allocation dans la boucle chaude
    // On a besoin de : n_opponents*2 + n_board_missing cartes dans le deck
    int cards_needed = n_opponents * 2 + n_board_missing;

    // Tableau de travail : copie du deck réutilisée par Fisher-Yates
    std::vector<Card> work_deck(base_deck);

    // Résultats par adversaire pré-alloués
    std::vector<HandScore> opp_scores(n_opponents);

    EquityResult result;
    result.n = 0;

    for (int sim = 0; sim < n_sims; sim++) {
        // Réinitialise le deck de travail (copie rapide)
        work_deck = base_deck;
        int deck_size = static_cast<int>(work_deck.size());

        if (deck_size < cards_needed) continue; // ne devrait pas arriver

        // Fisher-Yates partiel : tire (cards_needed) cartes sans remise
        // Les premières n_opponents*2 cartes → mains adversaires
        // Les suivantes n_board_missing → complétion du board
        for (int i = 0; i < cards_needed; i++) {
            std::uniform_int_distribution<int> d(i, deck_size - 1);
            int j = d(rng_);
            std::swap(work_deck[i], work_deck[j]);
        }

        // Mains adversaires : work_deck[0..2*n_opp-1]
        // Board complet : board_fixed + work_deck[2*n_opp .. 2*n_opp+n_board_missing-1]

        // Board complet (5 cartes)
        std::array<Card, 5> board5;
        for (int i = 0; i < static_cast<int>(board_fixed.size()); i++)
            board5[i] = board_fixed[i];
        int board_base = n_opponents * 2;
        for (int i = 0; i < n_board_missing; i++)
            board5[static_cast<int>(board_fixed.size()) + i] = work_deck[board_base + i];

        // Évaluation de notre main
        std::array<Card, 7> our_7 = {
            our_c1, our_c2,
            board5[0], board5[1], board5[2], board5[3], board5[4]
        };
        HandScore our_score = evaluator_.evaluate(our_7);

        // Évaluation des adversaires
        HandScore best_opp = 0;
        for (int p = 0; p < n_opponents; p++) {
            Card oc1 = work_deck[p * 2];
            Card oc2 = work_deck[p * 2 + 1];
            std::array<Card, 7> opp_7 = {
                oc1, oc2,
                board5[0], board5[1], board5[2], board5[3], board5[4]
            };
            opp_scores[p] = evaluator_.evaluate(opp_7);
            if (opp_scores[p] > best_opp) best_opp = opp_scores[p];
        }

        result.n++;
        if (our_score > best_opp) {
            result.win += 1.0;
        } else if (our_score == best_opp) {
            // Compter combien d'adversaires sont à égalité avec nous
            int tied = 1;
            for (int p = 0; p < n_opponents; p++)
                if (opp_scores[p] == our_score) tied++;
            result.tie += 1.0 / tied;
        } else {
            result.loss += 1.0;
        }
    }

    if (result.n > 0) {
        double inv = 1.0 / result.n;
        result.win  *= inv;
        result.tie  *= inv;
        result.loss *= inv;
        double n_opp = static_cast<double>(n_opponents);
        result.ev = result.win + (n_opp > 0 ? result.tie / n_opp : result.tie);
    }

    return result;
}

} // namespace poker
