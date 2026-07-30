#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "hand_eval.h"
#include "monte_carlo.h"
#include "ehs.h"

namespace py = pybind11;
using namespace poker;

PYBIND11_MODULE(poker_engine, m) {
    m.doc() = "Moteur de calcul poker — Hand Evaluator + Monte Carlo + EHS (C++)";

    // ── HandScore ─────────────────────────────────────────────────────────────
    m.def("evaluate_hand",
        [](const std::vector<std::string>& cards) -> uint32_t {
            HandEvaluator ev;
            return ev.evaluate(cards);
        },
        py::arg("cards"),
        R"doc(
Évalue la meilleure main à 5 cartes parmi 7 cartes données.
Retourne un entier : plus il est élevé, plus la main est forte.
Comparable directement : evaluate_hand(a) > evaluate_hand(b) <=> a bat b.

Args:
    cards: liste de 7 strings, ex ["As","Kd","7h","8c","9s","Th","2d"]

Returns:
    uint32 — score de main
        )doc"
    );

    m.def("hand_rank_name",
        [](uint32_t score) -> std::string {
            return HandEvaluator::rank_name(score);
        },
        py::arg("score"),
        "Retourne le nom de la catégorie (ex: 'Quinte flush', 'Paire', ...)"
    );

    m.def("compare_hands",
        [](const std::vector<std::string>& hand_a,
           const std::vector<std::string>& hand_b) -> int {
            HandEvaluator ev;
            uint32_t sa = ev.evaluate(hand_a);
            uint32_t sb = ev.evaluate(hand_b);
            if (sa > sb) return  1;  // A gagne
            if (sa < sb) return -1;  // B gagne
            return 0;                // Égalité
        },
        py::arg("hand_a"), py::arg("hand_b"),
        "Compare deux mains de 7 cartes. Retourne 1 (A gagne), -1 (B gagne), 0 (égalité)."
    );

    // ── EquityResult ──────────────────────────────────────────────────────────
    py::class_<EquityResult>(m, "EquityResult")
        .def_readonly("win",  &EquityResult::win,  "Fraction de mains gagnées")
        .def_readonly("tie",  &EquityResult::tie,  "Fraction de mains partagées (equity)")
        .def_readonly("loss", &EquityResult::loss, "Fraction de mains perdues")
        .def_readonly("ev",   &EquityResult::ev,   "Equity nette = win + tie/n_opp")
        .def_readonly("n",    &EquityResult::n,    "Nombre de simulations effectuées")
        .def("__repr__", [](const EquityResult& r) {
            char buf[128];
            snprintf(buf, sizeof(buf),
                "EquityResult(win=%.1f%%, tie=%.1f%%, loss=%.1f%%, ev=%.1f%%, n=%d)",
                r.win*100, r.tie*100, r.loss*100, r.ev*100, r.n);
            return std::string(buf);
        });

    // ── MonteCarloEngine ──────────────────────────────────────────────────────
    py::class_<MonteCarloEngine>(m, "MonteCarloEngine")
        .def(py::init<uint64_t>(), py::arg("seed") = 42)

        .def("compute",
            &MonteCarloEngine::compute,
            py::arg("our_hand"),
            py::arg("board"),
            py::arg("opp_ranges"),
            py::arg("n_sims") = 50000,
            R"doc(
Calcule l'équité multiway de notre main contre les ranges des adversaires actifs.

Args:
    our_hand  : ["As", "Kd"] — nos 2 cartes
    board     : ["7h", "8c", "9s"] — board actuel (0 à 5 cartes)
    opp_ranges: liste de dicts {combo_str: poids} pour chaque adversaire actif
                ex: [{"AsKd": 0.8, "QhQs": 1.0, ...}, {...}]
    n_sims    : nombre de simulations (défaut: 50 000)

Returns:
    EquityResult avec win, tie, loss, ev, n
            )doc"
        )

        .def("compute_vs_random",
            &MonteCarloEngine::compute_vs_random,
            py::arg("our_hand"),
            py::arg("board"),
            py::arg("n_opponents"),
            py::arg("n_sims") = 50000,
            "Équité vs N adversaires avec range uniforme (pour tests et benchmarks)"
        );

    // ── EHSResult ─────────────────────────────────────────────────────────────
    py::class_<EHSResult>(m, "EHSResult")
        .def_readonly("HS",              &EHSResult::HS,
                      "Hand Strength : P(gagner le showdown maintenant)")
        .def_readonly("PPot",            &EHSResult::PPot,
                      "Positive Potential : P(améliorer et gagner)")
        .def_readonly("NPot",            &EHSResult::NPot,
                      "Negative Potential : P(adversaire améliore et gagne)")
        .def_readonly("EHS",             &EHSResult::EHS,
                      "Effective Hand Strength = HS*(1-NPot) + (1-HS)*PPot")
        .def_readonly("simulations_run", &EHSResult::simulations_run,
                      "Nombre de simulations effectuées")
        .def_readonly("elapsed_ms",      &EHSResult::elapsed_ms,
                      "Temps de calcul en millisecondes")
        .def("__repr__", [](const EHSResult& r) {
            return describe_ehs(r);
        });

    // ── EHSCalculator ─────────────────────────────────────────────────────────
    py::class_<EHSCalculator>(m, "EHSCalculator")
        .def(py::init<uint64_t, bool>(),
             py::arg("seed") = 0,
             py::arg("use_openmp") = true,
             R"doc(
Calculateur EHS (Effective Hand Strength) par Monte Carlo.

Args:
    seed (int)       : Graine RNG — 0 = aléatoire, fixer pour reproductibilité.
    use_openmp (bool): Activer OpenMP si compilé avec USE_OPENMP. Défaut: True.
             )doc")

        .def("calculate", &EHSCalculator::calculate,
             py::arg("our_hand"),
             py::arg("board"),
             py::arg("n_sims") = 10000,
             R"doc(
Calcule l'EHS contre 1 adversaire avec range uniforme.

Args:
    our_hand (list[str]) : Nos 2 cartes, ex: ["As", "Kd"]
    board    (list[str]) : Board actuel (0–5 cartes), ex: ["7h", "8c", "9s"]
    n_sims   (int)       : Nombre de simulations (défaut: 10 000)

Returns:
    EHSResult avec HS, PPot, NPot, EHS, simulations_run, elapsed_ms

Exemple:
    calc = poker_engine.EHSCalculator(seed=42)
    r = calc.calculate(["As", "Kd"], ["7h", "8c", "9s"])
    print(f"EHS={r.EHS:.3f}  HS={r.HS:.3f}  PPot={r.PPot:.3f}  NPot={r.NPot:.3f}")
             )doc")

        .def("calculate_multiway", &EHSCalculator::calculate_multiway,
             py::arg("our_hand"),
             py::arg("board"),
             py::arg("n_opponents"),
             py::arg("n_sims") = 10000,
             R"doc(
Calcule l'EHS contre plusieurs adversaires (range uniforme).
La HS représente P(battre TOUS les adversaires actifs).

Args:
    our_hand    (list[str]) : Nos 2 cartes
    board       (list[str]) : Board actuel (0–5 cartes)
    n_opponents (int)       : Nombre d'adversaires actifs (1–8)
    n_sims      (int)       : Nombre de simulations (défaut: 10 000)

Returns:
    EHSResult
             )doc");

    // ── Fonctions utilitaires EHS ─────────────────────────────────────────────
    m.def("calculate_hs_only", &calculate_hs_only,
          py::arg("our_hand"),
          py::arg("board"),
          py::arg("n_sims") = 10000,
          R"doc(
Calcule la Hand Strength uniquement (sans PPot/NPot).
Plus rapide que calculate() — utile sur river ou en préflop rapide.

Returns:
    float : P(gagner le showdown) entre 0.0 et 1.0
          )doc");

    m.def("describe_ehs",
          [](const EHSResult& r) { return describe_ehs(r); },
          py::arg("result"),
          "Retourne une description textuelle de l'EHSResult (pour debug/logs).");
}
