#pragma once
#include "hand_eval.h"
#include <vector>
#include <map>
#include <string>
#include <random>
#include <numeric>

namespace poker {

// ─────────────────────────────────────────────────────────────────────────────
//  MonteCarloEngine — calcul d'équité multiway
//
//  Pour chaque simulation :
//    1. On tire un combo pour chaque adversaire selon sa distribution de range
//    2. On complète le board avec les cartes restantes
//    3. On évalue toutes les mains, on désigne le(s) gagnant(s)
//    4. On accumule les résultats
//
//  Retourne P(win), P(tie), P(loss) pour notre main.
// ─────────────────────────────────────────────────────────────────────────────

struct EquityResult {
    double win  = 0.0;  // fraction de mains gagnées
    double tie  = 0.0;  // fraction de mains partagées
    double loss = 0.0;  // fraction de mains perdues
    double ev   = 0.0;  // win + tie/nb_adversaires (equity nette)
    int    n    = 0;    // nombre de simulations effectuées
};

// Range d'un joueur : combo (ex "AsKd") → probabilité relative
// Les probabilités n'ont pas besoin d'être normalisées (on normalise en interne)
using PlayerRange = std::map<std::string, double>;

class MonteCarloEngine {
public:
    explicit MonteCarloEngine(uint64_t seed = 42);

    // Calcul d'équité multiway
    // our_hand   : nos 2 cartes, ex {"As", "Kd"}
    // board      : cartes communes déjà visibles (0 à 5 cartes)
    // opp_ranges : ranges de chaque adversaire actif (peut être vide → range aléatoire)
    // n_sims     : nombre de simulations (50 000 recommandé)
    EquityResult compute(
        const std::vector<std::string>& our_hand,
        const std::vector<std::string>& board,
        const std::vector<PlayerRange>&  opp_ranges,
        int n_sims = 50000
    );

    // Version simplifiée : adversaires avec range uniforme (pour tests)
    EquityResult compute_vs_random(
        const std::vector<std::string>& our_hand,
        const std::vector<std::string>& board,
        int n_opponents,
        int n_sims = 50000
    );

private:
    HandEvaluator  evaluator_;
    std::mt19937_64 rng_;

    // Construit le deck complet et retire les cartes déjà connues
    std::vector<Card> build_available_deck(
        const std::vector<Card>& known_cards
    ) const;

    // Tire un combo depuis une range, en évitant les cartes déjà utilisées
    // Retourne {card1, card2} ou {INVALID, INVALID} si impossible
    std::pair<Card,Card> sample_combo(
        const PlayerRange& range,
        const std::vector<Card>& used_cards,
        std::mt19937_64& rng
    ) const;

    static constexpr Card INVALID_CARD = 0xFFFFFFFF;
};

} // namespace poker
