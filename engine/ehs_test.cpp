// =============================================================================
// ehs_test.cpp — Tests standalone EHS Calculator (sans Python)
//
// Compilation depuis engine/build/ :
//   cmake --build . --config Release --target ehs_test
//
// Ou directement avec MSVC :
//   cl /EHsc /std:c++17 /O2 /I.. ehs_test.cpp ..\ehs.cpp ..\hand_eval.cpp /Fe:ehs_test.exe
//
// Execution :
//   .\Release\ehs_test.exe
// =============================================================================

#include "ehs.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>

using namespace poker;

// Helper : verifie qu'une valeur est dans [min, max]
static void check_range(double val, double min, double max, const char* label) {
    if (val < min || val > max) {
        std::cerr << "ECHEC " << label << " : " << val
                  << " hors de [" << min << ", " << max << "]\n";
        std::exit(1);
    }
}

// Helper : verifie qu'une condition est vraie
static void check(bool cond, const char* label) {
    if (!cond) {
        std::cerr << "ECHEC : " << label << "\n";
        std::exit(1);
    }
}

int main() {
    std::cout << "\n=== Tests EHS Calculator (standalone C++) ===\n\n";

    EHSCalculator calc(42, false);  // seed fixe, sans OpenMP pour reproductibilite

    // -------------------------------------------------------------------------
    // T1 : Valeurs dans [0, 1]
    // -------------------------------------------------------------------------
    {
        auto r = calc.calculate({"As", "Kd"}, {"7h", "2c", "3d"}, 5000);
        check_range(r.HS,   0.0, 1.0, "T1 HS");
        check_range(r.PPot, 0.0, 1.0, "T1 PPot");
        check_range(r.NPot, 0.0, 1.0, "T1 NPot");
        check_range(r.EHS,  0.0, 1.0, "T1 EHS");
        std::cout << "[T1] Valeurs dans [0,1] : OK\n";
        std::cout << "     " << describe_ehs(r) << "\n";
    }

    // -------------------------------------------------------------------------
    // T2 : Flush draw nut -> PPot eleve, EHS > HS
    // As2s sur Ks7s3d : flush draw nut (9 outs sur 47)
    // -------------------------------------------------------------------------
    {
        auto r = calc.calculate({"As", "2s"}, {"Ks", "7s", "3d"}, 10000);
        check(r.PPot > 0.18, "T2 PPot flush draw > 0.18");
        check(r.EHS  > r.HS, "T2 EHS > HS pour draw fort");
        std::cout << "[T2] Flush draw nut    : OK\n";
        std::cout << "     " << describe_ehs(r) << "\n";
    }

    // -------------------------------------------------------------------------
    // T3 : Set sur board sec -> HS tres elevee
    // 7h7d sur 7sAdKc
    // -------------------------------------------------------------------------
    {
        auto r = calc.calculate({"7h", "7d"}, {"7s", "Ad", "Kc"}, 10000);
        check(r.HS > 0.85, "T3 HS set > 0.85");
        check(r.EHS > 0.75, "T3 EHS set > 0.75");
        std::cout << "[T3] Set sur board sec : OK\n";
        std::cout << "     " << describe_ehs(r) << "\n";
    }

    // -------------------------------------------------------------------------
    // T4 : River -> PPot = NPot = 0 (plus de cartes a venir)
    // -------------------------------------------------------------------------
    {
        auto r = calc.calculate({"Ah", "Kh"}, {"2s", "7d", "Qc", "Jh", "3s"}, 5000);
        check(r.PPot == 0.0, "T4 PPot river == 0");
        check(r.NPot == 0.0, "T4 NPot river == 0");
        std::cout << "[T4] River (PPot=NPot=0): OK\n";
        std::cout << "     " << describe_ehs(r) << "\n";
    }

    // -------------------------------------------------------------------------
    // T5 : Multiway -> EHS 1v3 < EHS 1v1
    // -------------------------------------------------------------------------
    {
        auto r1 = calc.calculate_multiway({"Ah", "Kh"}, {"7d", "2s", "3c"}, 1, 10000);
        auto r3 = calc.calculate_multiway({"Ah", "Kh"}, {"7d", "2s", "3c"}, 3, 10000);
        check(r3.EHS < r1.EHS, "T5 EHS 1v3 < EHS 1v1");
        std::cout << "[T5] Multiway          : OK\n";
        std::cout << "     1v1 " << describe_ehs(r1) << "\n";
        std::cout << "     1v3 " << describe_ehs(r3) << "\n";
    }

    // -------------------------------------------------------------------------
    // T6 : Ordonnancement Set > Top pair
    // -------------------------------------------------------------------------
    {
        auto r_set = calc.calculate({"7h", "7d"}, {"7s", "Ad", "Kc"}, 10000);
        auto r_tp  = calc.calculate({"Ah", "Qd"}, {"As", "7d", "2c"}, 10000);
        check(r_set.EHS > r_tp.EHS, "T6 Set EHS > Top pair EHS");
        std::cout << "[T6] Set > Top pair    : OK\n";
        std::cout << "     Set      EHS=" << r_set.EHS << "\n";
        std::cout << "     Top pair EHS=" << r_tp.EHS  << "\n";
    }

    // -------------------------------------------------------------------------
    // T7 : Formule EHS = HS*(1-NPot) + (1-HS)*PPot
    // -------------------------------------------------------------------------
    {
        auto r = calc.calculate({"Jh", "Th"}, {"9h", "2d", "Ac"}, 10000);
        double expected = r.HS * (1.0 - r.NPot) + (1.0 - r.HS) * r.PPot;
        check(std::fabs(r.EHS - expected) < 0.001, "T7 formule EHS coherente");
        std::cout << "[T7] Formule EHS       : OK\n";
        std::cout << "     " << describe_ehs(r) << "\n";
    }

    // -------------------------------------------------------------------------
    // T8 : Carte invalide -> exception
    // -------------------------------------------------------------------------
    {
        bool threw = false;
        try { calc.calculate({"Xx", "Kd"}, {}, 100); }
        catch (const std::exception&) { threw = true; }
        check(threw, "T8 carte invalide -> exception");
        std::cout << "[T8] Carte invalide    : OK (exception levee)\n";
    }

    // -------------------------------------------------------------------------
    // T9 : Doublon -> exception
    // -------------------------------------------------------------------------
    {
        bool threw = false;
        try { calc.calculate({"As", "As"}, {}, 100); }
        catch (const std::exception&) { threw = true; }
        check(threw, "T9 doublon -> exception");
        std::cout << "[T9] Doublon           : OK (exception levee)\n";
    }

    // -------------------------------------------------------------------------
    // T10 : Performance (indicatif)
    // -------------------------------------------------------------------------
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        calc.calculate({"Ah", "Kh"}, {"7d", "2s", "3c"}, 10000);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "[T10] Performance      : " << ms << " ms pour 10k sims\n";
        std::cout << "      (cible : < 50ms sans OpenMP, < 10ms avec OpenMP)\n";
    }

    std::cout << "\nTous les tests passent.\n\n";
    return 0;
}
