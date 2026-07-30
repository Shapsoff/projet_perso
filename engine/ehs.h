#pragma once

// =============================================================================
// ehs.h — EHS Calculator (Effective Hand Strength)
// Phase 2 — Bot Poker Académique
//
// Dépend de hand_eval.h et monte_carlo.h (phase 1).
// =============================================================================

#include "hand_eval.h"   // Card, make_card, card_rank, card_suit, parse_card, HandEvaluator
#include "monte_carlo.h" // build_available_deck (via MonteCarloEngine)
#include <vector>
#include <string>
#include <cstdint>

namespace poker {

// ---------------------------------------------------------------------------
// Résultat EHS
// ---------------------------------------------------------------------------

/**
 * EHS = HS * (1 - NPot) + (1 - HS) * PPot
 *
 * HS   : Hand Strength   — P(gagner le showdown maintenant)
 * PPot : Positive Potential — P(améliorer et dépasser l'adversaire)
 * NPot : Negative Potential — P(l'adversaire améliore et nous dépasse)
 * EHS  : score combiné utilisé par l'EHSBot
 */
struct EHSResult {
    double HS   = 0.0;
    double PPot = 0.0;
    double NPot = 0.0;
    double EHS  = 0.0;

    int    simulations_run = 0;
    double elapsed_ms      = 0.0;
};

// ---------------------------------------------------------------------------
// EHSCalculator
// ---------------------------------------------------------------------------

class EHSCalculator {
public:
    /**
     * @param seed       Graine RNG (0 = aléatoire)
     * @param use_openmp Activer OpenMP si compilé avec USE_OPENMP
     */
    explicit EHSCalculator(uint64_t seed = 0, bool use_openmp = true);

    /**
     * Calcule l'EHS contre 1 adversaire (range uniforme).
     *
     * @param our_hand  Nos 2 cartes, ex: {"As", "Kd"}
     * @param board     Board actuel (0–5 cartes)
     * @param n_sims    Nombre de simulations Monte Carlo
     */
    EHSResult calculate(
        const std::vector<std::string>& our_hand,
        const std::vector<std::string>& board,
        int n_sims = 10000
    ) const;

    /**
     * Calcule l'EHS contre plusieurs adversaires.
     * HS = P(battre TOUS les adversaires actifs).
     *
     * @param n_opponents Nombre d'adversaires actifs (1–8)
     */
    EHSResult calculate_multiway(
        const std::vector<std::string>& our_hand,
        const std::vector<std::string>& board,
        int n_opponents,
        int n_sims = 10000
    ) const;

private:
    uint64_t seed_;
    bool     use_openmp_;
    mutable uint64_t rng_state_;

    uint64_t xorshift64() const;

    void run_simulations(
        const std::array<Card, 2>&  our_cards,
        const std::vector<Card>&    board_cards,
        const std::vector<Card>&    remaining_deck,
        int                         n_opponents,
        int                         n_sims,
        double& out_hs,
        double& out_ppot,
        double& out_npot
    ) const;
};

// ---------------------------------------------------------------------------
// Utilitaires standalone
// ---------------------------------------------------------------------------

/** Calcule la Hand Strength uniquement — plus rapide, utile sur river. */
double calculate_hs_only(
    const std::vector<std::string>& our_hand,
    const std::vector<std::string>& board,
    int n_sims = 10000
);

/** Description textuelle d'un EHSResult pour les logs. */
std::string describe_ehs(const EHSResult& result);

} // namespace poker
