#include "hand_eval.h"
#include "monte_carlo.h"
#include <iostream>
#include <chrono>
#include <cassert>
#include <iomanip>
#include <random>
#include <algorithm>

using namespace poker;
using namespace std::chrono;

// ── Helpers d'affichage ───────────────────────────────────────────────────────
void print_sep(const std::string& title = "") {
    std::cout << "\n" << std::string(60, '-') << "\n";
    if (!title.empty()) std::cout << "  " << title << "\n" << std::string(60, '-') << "\n";
}

void check(bool cond, const std::string& msg) {
    if (cond) std::cout << "  ✅ " << msg << "\n";
    else      { std::cout << "  ❌ ÉCHEC : " << msg << "\n"; std::exit(1); }
}

// ── Tests de correction ───────────────────────────────────────────────────────
void test_correction() {
    print_sep("TESTS DE CORRECTION — HandEvaluator");
    HandEvaluator ev;

    // Quinte flush royale vs carré d'as
    auto qfr   = ev.evaluate({"As","Ks","Qs","Js","Ts","2h","3d"});
    auto carre = ev.evaluate({"As","Ah","Ad","Ac","Ks","Qs","Js"});
    check(qfr > carre, "Quinte flush royale > Carré d'as");

    // Full house > couleur
    auto full    = ev.evaluate({"As","Ah","Ad","Ks","Kh","2c","3d"});
    auto couleur = ev.evaluate({"As","Ks","Qs","Js","9s","2h","3d"});
    check(full > couleur, "Full house > Couleur");

    // Couleur > quinte
    auto quinte = ev.evaluate({"As","Kh","Qd","Jc","Ts","2h","3d"});
    check(couleur > quinte, "Couleur > Quinte");

    // Quinte > brelan
    auto brelan = ev.evaluate({"As","Ah","Ad","Ks","Qh","2c","3d"});
    check(quinte > brelan, "Quinte > Brelan");

    // Brelan > deux paires
    auto deux_p = ev.evaluate({"As","Ah","Ks","Kh","Qd","2c","3d"});
    check(brelan > deux_p, "Brelan > Deux paires");

    // Deux paires > paire
    auto paire = ev.evaluate({"As","Ah","Ks","Qd","Jc","2c","3d"});
    check(deux_p > paire, "Deux paires > Paire");

    // Kickers : paire d'as avec K > paire d'as avec Q
    auto pair_ak = ev.evaluate({"As","Ah","Kd","Qc","Jh","2s","3d"});
    auto pair_aq = ev.evaluate({"As","Ah","Qd","Jc","9h","2s","3d"});
    check(pair_ak > pair_aq, "Paire As+K kicker > Paire As+Q kicker");

    // Quinte roue A-2-3-4-5
    auto roue    = ev.evaluate({"As","2h","3d","4c","5s","Kh","Qd"});
    auto quinte6 = ev.evaluate({"2s","3h","4d","5c","6s","Kh","Qd"});
    check(quinte6 > roue, "Quinte 6-high > Quinte roue (A-5)");

    // Égalité
    auto hand_a = ev.evaluate({"As","Kd","Qh","Jc","9s","2h","3d"});
    auto hand_b = ev.evaluate({"As","Kh","Qd","Jc","9h","2s","3d"});
    // Même hauteur AKQJx → égalité
    // (board identique, main différente mais même force)
    auto shared = ev.evaluate({"As","Kd","Qh","Jc","9s","2h","3d"});
    auto shared2= ev.evaluate({"As","Kd","Qh","Jc","9s","3h","2d"});
    check(shared == shared2, "Égalité détectée sur deux mains identiques en force");

    std::cout << "\n  Tous les tests de correction passés ✅\n";
}

// ── Benchmark ─────────────────────────────────────────────────────────────────
void benchmark_hand_eval(int n = 1'000'000) {
    print_sep("BENCHMARK — Hand Evaluator (" + std::to_string(n) + " évaluations)");
    HandEvaluator ev;

    // Génère des mains aléatoires
    std::mt19937 rng(12345);
    std::vector<std::array<Card,7>> hands(n);

    // Générer des decks aléatoires valides
    std::vector<Card> deck;
    for (int r = 0; r < 13; r++)
        for (int s = 0; s < 4; s++)
            deck.push_back(make_card(r, s));

    for (int i = 0; i < n; i++) {
        std::shuffle(deck.begin(), deck.end(), rng);
        for (int j = 0; j < 7; j++) hands[i][j] = deck[j];
    }

    auto t0 = high_resolution_clock::now();
    volatile HandScore dummy = 0;
    for (int i = 0; i < n; i++)
        dummy = ev.evaluate(hands[i]);
    auto t1 = high_resolution_clock::now();
    (void)dummy;

    double total_ms = duration_cast<microseconds>(t1 - t0).count() / 1000.0;
    double per_eval_ns = duration_cast<nanoseconds>(t1 - t0).count() / static_cast<double>(n);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Total         : " << total_ms << " ms\n";
    std::cout << "  Par évaluation: " << per_eval_ns << " ns\n";
    std::cout << "  Throughput    : " << static_cast<int>(n / (total_ms/1000.0)) << " éval/sec\n";

    if (per_eval_ns < 1000.0)
        std::cout << "  ✅ Objectif < 1 µs atteint (" << per_eval_ns << " ns)\n";
    else
        std::cout << "  ⚠️  Objectif < 1 µs non atteint (" << per_eval_ns << " ns)\n";
}

// ── Benchmark Monte Carlo ─────────────────────────────────────────────────────
void benchmark_monte_carlo() {
    print_sep("BENCHMARK — Monte Carlo (50k simulations)");
    MonteCarloEngine mc(42);

    struct TestCase {
        std::string name;
        std::vector<std::string> hand;
        std::vector<std::string> board;
        int n_opp;
        double expected_ev_min;
        double expected_ev_max;
    };

    std::vector<TestCase> cases = {
        {"AA préflop vs 1",   {"As","Ah"}, {},               1, 0.80, 0.90},
        {"AA préflop vs 3",   {"As","Ah"}, {},               3, 0.60, 0.75},
        {"72o préflop vs 1",  {"7c","2d"}, {},               1, 0.30, 0.40},
        {"AK flop top pair",  {"As","Kd"}, {"Ah","7c","2d"}, 1, 0.70, 0.90},
    };

    for (auto& tc : cases) {
        auto t0 = high_resolution_clock::now();
        auto result = mc.compute_vs_random(tc.hand, tc.board, tc.n_opp, 50000);
        auto t1 = high_resolution_clock::now();
        double ms = duration_cast<microseconds>(t1 - t0).count() / 1000.0;

        std::cout << "\n  [" << tc.name << "]\n";
        std::cout << "    EV     : " << std::setprecision(1) << result.ev*100 << "%"
                  << "  (attendu " << tc.expected_ev_min*100 << "–" << tc.expected_ev_max*100 << "%)\n";
        std::cout << "    win=" << result.win*100 << "% tie=" << result.tie*100
                  << "% loss=" << result.loss*100 << "%\n";
        std::cout << "    Temps  : " << ms << " ms (" << result.n << " sims valides)\n";

        bool ev_ok = result.ev >= tc.expected_ev_min && result.ev <= tc.expected_ev_max;
        check(ev_ok, "EV dans la plage attendue");

        if (ms < 50.0) std::cout << "    ✅ Objectif < 50 ms atteint\n";
        else           std::cout << "    ⚠️  Objectif < 50 ms non atteint (" << ms << " ms)\n";
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "\n╔══════════════════════════════════════════════════╗\n";
    std::cout <<   "║      POKER BOT — TESTS MOTEUR C++               ║\n";
    std::cout <<   "╚══════════════════════════════════════════════════╝\n";

    test_correction();
    benchmark_hand_eval(1'000'000);
    benchmark_monte_carlo();

    print_sep("RÉSUMÉ");
    std::cout << "  Tous les tests passés. Le moteur est prêt pour l'intégration Python.\n\n";
    return 0;
}
