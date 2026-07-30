// =============================================================================
// ehs.cpp — EHS Calculator (Effective Hand Strength)
// Phase 2 — Bot Poker Academique
//
// Utilise evaluate(vector<Card>) de hand_eval.h pour evaluer exactement
// les cartes disponibles selon la street, sans aucun padding :
//
//   Preflop : HS via Monte Carlo, PPot=NPot=0 (pas de board)
//   Flop    : idx_current via eval(5 cartes), idx_future via eval(7 cartes)
//   Turn    : idx_current via eval(6 cartes), idx_future via eval(7 cartes)
//   River   : HS exacte via eval(7 cartes),   PPot=NPot=0 par definition
// =============================================================================

#include "ehs.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <bitset>

namespace poker {

// =============================================================================
// popcount portable (MSVC ne supporte pas __builtin_popcountll)
// =============================================================================

static inline int popcount64(uint64_t x) {
    return static_cast<int>(std::bitset<64>(x).count());
}

// =============================================================================
// Constructeur
// =============================================================================

EHSCalculator::EHSCalculator(uint64_t seed, bool use_openmp)
    : seed_(seed)
    , use_openmp_(use_openmp)
    , rng_state_(seed != 0 ? seed : 0xDEADBEEFCAFEBABEULL)
{}

uint64_t EHSCalculator::xorshift64() const {
    rng_state_ ^= rng_state_ << 13;
    rng_state_ ^= rng_state_ >> 7;
    rng_state_ ^= rng_state_ << 17;
    return rng_state_;
}

// =============================================================================
// Construction du deck restant
// =============================================================================

static std::vector<Card> build_remaining_deck(const std::vector<Card>& known_cards) {
    uint64_t used_mask = 0;
    for (Card c : known_cards)
        used_mask |= (1ULL << (card_rank(c) * 4 + card_suit(c)));

    std::vector<Card> deck;
    deck.reserve(52 - popcount64(used_mask));
    for (int r = 0; r < 13; r++)
        for (int s = 0; s < 4; s++)
            if (!((used_mask >> (r * 4 + s)) & 1))
                deck.push_back(make_card(r, s));
    return deck;
}

// =============================================================================
// eval_current — evalue l'etat actuel sans padding
//
// Flop  (board_size=3) : 2 main + 3 board = 5 cartes -> eval5 via evaluate(vec5)
// Turn  (board_size=4) : 2 main + 4 board = 6 cartes -> eval6 via evaluate(vec6)
// River (board_size=5) : traite separement (PPot=NPot=0)
// Preflop (board_size=0) : pas d'evaluation possible, retourne 0
// =============================================================================

static HandScore eval_current(
    const HandEvaluator&     evaluator,
    const std::array<Card,2>& hand,
    const std::vector<Card>&  board)
{
    int n = static_cast<int>(board.size());
    if (n < 3) return 0; // preflop : pas d'evaluation

    // Construire le vecteur main + board (5 ou 6 cartes)
    std::vector<Card> cards;
    cards.reserve(2 + n);
    cards.push_back(hand[0]);
    cards.push_back(hand[1]);
    for (Card c : board) cards.push_back(c);

    // evaluate(vector<Card>) dispatche sur eval5 (n=3) ou eval6 (n=4)
    return evaluator.evaluate(cards);
}

// =============================================================================
// Noyau de simulation Monte Carlo
//
// Matrice HP[etat_actuel][etat_futur] aplatie en 9 scalaires (compat MSVC OpenMP)
// etat : 0=behind, 1=tied, 2=ahead
//
// PPot = (HP[0][2] + HP[1][2]) / (total behind+tied)
// NPot = (HP[2][0] + HP[1][0]) / (total ahead+tied)
// =============================================================================

void EHSCalculator::run_simulations(
    const std::array<Card, 2>&  our_cards,
    const std::vector<Card>&    board_cards,
    const std::vector<Card>&    remaining_deck,
    int                         n_opponents,
    int                         n_sims,
    double& out_hs,
    double& out_ppot,
    double& out_npot
) const {
    const int board_size    = static_cast<int>(board_cards.size());
    const int cards_to_draw = 5 - board_size;
    const bool is_river     = (board_size == 5);
    const bool has_board    = (board_size >= 3);

    long long hp00=0, hp01=0, hp02=0;
    long long hp10=0, hp11=0, hp12=0;
    long long hp20=0, hp21=0, hp22=0;
    long long HS_win = 0, HS_total = 0;

    HandEvaluator evaluator;

#ifdef USE_OPENMP
    #pragma omp parallel if(use_openmp_) \
        reduction(+: HS_win, HS_total,   \
                     hp00, hp01, hp02,   \
                     hp10, hp11, hp12,   \
                     hp20, hp21, hp22)
#endif
    {
        std::vector<Card> local_deck = remaining_deck;
        int local_size = static_cast<int>(local_deck.size());

#ifdef USE_OPENMP
        uint64_t local_rng = rng_state_
            ^ (static_cast<uint64_t>(omp_get_thread_num() + 1) * 0x9E3779B97F4A7C15ULL);
#else
        uint64_t local_rng = rng_state_;
#endif

        auto xorshift = [&]() -> uint64_t {
            local_rng ^= local_rng << 13;
            local_rng ^= local_rng >> 7;
            local_rng ^= local_rng << 17;
            return local_rng;
        };

        auto draw = [&](int& size) -> Card {
            int idx = static_cast<int>(xorshift() % static_cast<uint64_t>(size));
            Card c = local_deck[idx];
            local_deck[idx] = local_deck[--size];
            return c;
        };

        auto restore = [&](const std::vector<Card>& drawn, int& size) {
            for (Card c : drawn) local_deck[size++] = c;
        };

#ifdef USE_OPENMP
        #pragma omp for schedule(static)
#endif
        for (int sim = 0; sim < n_sims; ++sim) {
            int size = local_size;
            std::vector<Card> drawn;
            drawn.reserve(2 * n_opponents + cards_to_draw);

            // ── 1. Tirer les mains adverses ───────────────────────────────────
            std::vector<std::array<Card, 2>> opp_hands(n_opponents);
            for (int p = 0; p < n_opponents; ++p) {
                opp_hands[p][0] = draw(size); drawn.push_back(opp_hands[p][0]);
                opp_hands[p][1] = draw(size); drawn.push_back(opp_hands[p][1]);
            }

            // ── 2. Etat ACTUEL — sans padding, cartes exactes disponibles ─────
            // Flop  : evaluate(5 cartes) = eval5
            // Turn  : evaluate(6 cartes) = eval6
            // River : traite plus bas (PPot=NPot=0)
            // Preflop : idx_current = 1 (tied, pas d'eval possible)
            int idx_current = 1;
            if (has_board && !is_river) {
                HandScore our_cur = eval_current(evaluator, our_cards, board_cards);

                int best_opp_cur = -1;
                for (int p = 0; p < n_opponents; ++p) {
                    HandScore sc = eval_current(evaluator, opp_hands[p], board_cards);
                    if (static_cast<int>(sc) > best_opp_cur)
                        best_opp_cur = static_cast<int>(sc);
                }

                if      (static_cast<int>(our_cur) > best_opp_cur) idx_current = 2;
                else if (static_cast<int>(our_cur) < best_opp_cur) idx_current = 0;
            }

            // ── 3. Completer le board jusqu'a 5 cartes ────────────────────────
            std::vector<Card> full_board(board_cards.begin(), board_cards.end());
            for (int i = 0; i < cards_to_draw; ++i) {
                Card c = draw(size);
                drawn.push_back(c);
                full_board.push_back(c);
            }

            // ── 4. Etat FINAL — board complet (7 cartes) ─────────────────────
            std::array<Card, 7> our_final = {
                our_cards[0], our_cards[1],
                full_board[0], full_board[1], full_board[2],
                full_board[3], full_board[4]
            };
            int our_sc_final = static_cast<int>(evaluator.evaluate(our_final));

            int best_opp_final = -1;
            for (int p = 0; p < n_opponents; ++p) {
                std::array<Card, 7> opp_final = {
                    opp_hands[p][0], opp_hands[p][1],
                    full_board[0], full_board[1], full_board[2],
                    full_board[3], full_board[4]
                };
                int sc = static_cast<int>(evaluator.evaluate(opp_final));
                if (sc > best_opp_final) best_opp_final = sc;
            }

            int idx_future = 1;
            if      (our_sc_final > best_opp_final) idx_future = 2;
            else if (our_sc_final < best_opp_final) idx_future = 0;

            // ── 5. Accumuler ──────────────────────────────────────────────────
            switch (idx_current * 3 + idx_future) {
                case 0: hp00++; break;
                case 1: hp01++; break;
                case 2: hp02++; break;
                case 3: hp10++; break;
                case 4: hp11++; break;
                case 5: hp12++; break;
                case 6: hp20++; break;
                case 7: hp21++; break;
                case 8: hp22++; break;
            }

            HS_total++;
            if (idx_future == 2) HS_win++;

            restore(drawn, size);
        }
    }

    // ── Metriques finales ─────────────────────────────────────────────────────
    out_hs = (HS_total > 0) ? static_cast<double>(HS_win) / HS_total : 0.0;

    if (is_river || !has_board) {
        // River : PPot/NPot = 0 par definition (plus de cartes a venir)
        // Preflop : PPot/NPot = 0 (pas de board pour evaluer l'etat actuel)
        out_ppot = 0.0;
        out_npot = 0.0;
        return;
    }

    long long behind_or_tied = hp00 + hp01 + hp02 + hp10 + hp11 + hp12;
    out_ppot = (behind_or_tied > 0)
        ? static_cast<double>(hp02 + hp12) / behind_or_tied
        : 0.0;

    long long ahead_or_tied = hp20 + hp21 + hp22 + hp10 + hp11 + hp12;
    out_npot = (ahead_or_tied > 0)
        ? static_cast<double>(hp20 + hp10) / ahead_or_tied
        : 0.0;
}

// =============================================================================
// calculate() / calculate_multiway()
// =============================================================================

EHSResult EHSCalculator::calculate(
    const std::vector<std::string>& our_hand,
    const std::vector<std::string>& board,
    int n_sims
) const {
    return calculate_multiway(our_hand, board, 1, n_sims);
}

EHSResult EHSCalculator::calculate_multiway(
    const std::vector<std::string>& our_hand,
    const std::vector<std::string>& board,
    int n_opponents,
    int n_sims
) const {
    if (our_hand.size() != 2)
        throw std::invalid_argument("our_hand doit contenir exactement 2 cartes");
    if (board.size() > 5)
        throw std::invalid_argument("board ne peut pas contenir plus de 5 cartes");
    if (n_opponents < 1 || n_opponents > 8)
        throw std::invalid_argument("n_opponents doit etre entre 1 et 8");
    if (n_sims < 100)
        throw std::invalid_argument("n_sims minimum : 100");

    auto t_start = std::chrono::high_resolution_clock::now();

    std::array<Card, 2> our_cards = {
        parse_card(our_hand[0]),
        parse_card(our_hand[1])
    };
    std::vector<Card> board_cards;
    board_cards.reserve(board.size());
    for (const auto& s : board)
        board_cards.push_back(parse_card(s));

    // Verification doublons
    std::vector<Card> all_known = { our_cards[0], our_cards[1] };
    all_known.insert(all_known.end(), board_cards.begin(), board_cards.end());
    std::vector<Card> sorted_known = all_known;
    std::sort(sorted_known.begin(), sorted_known.end());
    for (size_t i = 1; i < sorted_known.size(); ++i)
        if (sorted_known[i] == sorted_known[i-1])
            throw std::invalid_argument("Carte dupliquee detectee");

    std::vector<Card> remaining = build_remaining_deck(all_known);

    double hs = 0.0, ppot = 0.0, npot = 0.0;
    run_simulations(our_cards, board_cards, remaining, n_opponents, n_sims, hs, ppot, npot);

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    double ehs = hs * (1.0 - npot) + (1.0 - hs) * ppot;
    ehs = std::max(0.0, std::min(1.0, ehs));

    EHSResult result;
    result.HS              = hs;
    result.PPot            = ppot;
    result.NPot            = npot;
    result.EHS             = ehs;
    result.simulations_run = n_sims;
    result.elapsed_ms      = elapsed;
    return result;
}

// =============================================================================
// Utilitaires standalone
// =============================================================================

double calculate_hs_only(
    const std::vector<std::string>& our_hand,
    const std::vector<std::string>& board,
    int n_sims
) {
    EHSCalculator calc;
    return calc.calculate(our_hand, board, n_sims).HS;
}

std::string describe_ehs(const EHSResult& result) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "EHS=" << result.EHS
        << " (HS="  << result.HS
        << " PPot=" << result.PPot
        << " NPot=" << result.NPot << ")";

    if      (result.EHS >= 0.80) oss << " -> Tres forte (value/raise)";
    else if (result.EHS >= 0.70) oss << " -> Forte (bet/raise)";
    else if (result.EHS >= 0.55) {
        if (result.PPot >= 0.20) oss << " -> Draw fort (semi-bluff)";
        else                     oss << " -> Marginale haute (bet33)";
    }
    else if (result.EHS >= 0.45) oss << " -> Marginale basse (call si rentable)";
    else {
        if (result.PPot >= 0.15) oss << " -> Draw faible (fold probable)";
        else                     oss << " -> Faible (check/fold)";
    }

    if (result.NPot >= 0.30 && result.HS >= 0.70)
        oss << " /!\\ NPot eleve";

    oss << " [" << result.simulations_run << " sims "
        << std::setprecision(1) << result.elapsed_ms << "ms]";
    return oss.str();
}

} // namespace poker
