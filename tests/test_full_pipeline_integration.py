"""
test_full_pipeline_integration.py — Tests bout-en-bout, pipeline complet
Phase 6 — Bot Poker Académique

Écrit après réception de ev_calculator.py, sizing_optimizer.py,
bluff_layer.py, range_bot.py et deck.py (le vrai) — permet enfin de
tester BestResponseEngine.decide() avec un vrai EVCalculator/
SizingOptimizer/BluffLayer, chose impossible dans les passes précédentes.

Note sur poker_engine : toujours absent (extension C++ compilée,
spécifique à la machine de l'utilisateur). EVCalculator._compute_equity
et RangeBot._compute_ehs dégradent tous deux gracieusement vers une
équité stub neutre (0.5) quand le moteur C++ est indisponible — ce qui
permet de valider tout le CÂBLAGE et la LOGIQUE de décision, mais pas les
valeurs d'EV réalistes (qui nécessitent le vrai moteur Monte Carlo).
simulator.py, lui, N'A PAS de repli gracieux (RuntimeError si
poker_engine manque) — impossible de tester PokerTable/run_match ici.
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.bots.ehs_bot import EHSBot, EHSBotConfig, _normalize_game_state
from core.player_db.player_db import PlayerDB
from core.best_response.frequency_model import FrequencyModel
from core.player_db.db_frequency_model import DBAwareFrequencyModel
from core.range_estimator import RangeEstimator
from core.player_db.db_range_estimator import DBAwareRangeEstimator
from core.spr import get_spr_info_from_game_state


def t(label, cond, detail=""):
    global _passed, _failed
    if cond:
        print(f"  ✓ {label}")
        _passed += 1
    else:
        print(f"  ✗ {label}" + (f" : {detail}" if detail else ""))
        _failed += 1


POSTFLOP_STATE = {
    "hand": ["Ah", "Kd"], "board": ["Qh", "7d", "2c"], "position": "BTN",
    "pot": 20, "to_call": 0, "stack": 180,
    "players": [
        {"id": 0, "position": "BTN", "status": "active", "stack": 180},
        {"id": 1, "position": "BB",  "status": "active", "stack": 150},
    ],
    "our_id": 0, "street": "flop",
}


def _seed_tag(db: PlayerDB, player_id: str, n_hands: int = 50, dense_buckets: bool = False):
    for i in range(n_hands):
        db.new_hand_observed(player_id)
        vpip = (i % 7 == 0)
        db.record_preflop_action(player_id, "CO", vpip=vpip, pfr=vpip)
        db.record_postflop_aggression(player_id, aggressive=True)
    if dense_buckets:
        for bucket_ehs in [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
            action = 'fold' if bucket_ehs < 0.35 else ('call' if bucket_ehs < 0.65 else 'raise')
            for _ in range(8):
                db.record_ehs_bucket_action(player_id, "flop", ehs=bucket_ehs,
                                             facing_bet=True, action=action)


def _direct_br_decision(bot: EHSBot, state: dict):
    """Appelle BestResponseEngine.decide() directement pour inspecter la décision complète."""
    gs = _normalize_game_state(dict(state))
    ehs_result = bot._compute_ehs(gs['hand'], gs['board'], 1)
    spr_info = get_spr_info_from_game_state(gs)
    villain_id = bot._get_villain_id(gs['players'], gs.get('player_id', 0))
    bot._estimator.update_from_game_state(gs, villain_id)
    return bot._br_engine.decide(
        game_state=gs, estimator=bot._estimator, ehs_result=ehs_result,
        spr_info=spr_info, texture_modifier=1.0, bucket_modifier=0.0,
        hand_count=bot._effective_hand_count(),
    )


def main() -> int:
    global _passed, _failed
    _passed = _failed = 0

    print("\n=== Tests pipeline complet (EVCalculator + SizingOptimizer + BluffLayer) ===\n")

    import core.bots.ehs_bot as ehs_mod
    t("_BEST_RESPONSE_AVAILABLE = True avec tous les fichiers présents",
      ehs_mod._BEST_RESPONSE_AVAILABLE is True)

    # ── Régression : player_db=False -> classes phase 5 pures, zéro DB-aware ──
    bot_off = EHSBot(EHSBotConfig(use_player_db=False, use_best_response=True))
    t("player_db=False -> estimator est un RangeEstimator pur (pas DBAware)",
      type(bot_off._estimator) is RangeEstimator,
      f"obtenu {type(bot_off._estimator).__name__}")
    t("player_db=False -> freq_model est un FrequencyModel pur (pas DBAware)",
      type(bot_off._br_engine._freq_model) is FrequencyModel,
      f"obtenu {type(bot_off._br_engine._freq_model).__name__}")

    # ── player_db=True -> classes DB-aware correctement utilisées ──
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.sqlite3")
        bot_on = EHSBot(EHSBotConfig(use_player_db=True, player_db_path=db_path,
                                      use_best_response=True))
        t("player_db=True -> estimator est un DBAwareRangeEstimator",
          isinstance(bot_on._estimator, DBAwareRangeEstimator))
        t("player_db=True -> freq_model est un DBAwareFrequencyModel",
          isinstance(bot_on._br_engine._freq_model, DBAwareFrequencyModel))

        # ── Seuil de confiance effectif : documente la désynchronisation
        # trouvée entre EHSBotConfig.best_response_min_confidence (0.55) et
        # best_response_engine._MIN_ARCHETYPE_CONFIDENCE (0.75, commentaire
        # "durci de 0.55 à 0.75"). EHSBotConfig gagne car passé explicitement
        # au constructeur de BestResponseConfig — donc 0.55 est bien LA
        # valeur effective aujourd'hui. Ce test verrouille ce comportement
        # ACTUEL ; s'il échoue après une synchro des deux fichiers par
        # l'utilisateur, c'est attendu et ce test devra être mis à jour en
        # conséquence (35→75 mains pour l'activation BR).
        t("seuil de confiance BR effectif = celui d'EHSBotConfig (0.55), pas celui du module (0.75)",
          bot_on._br_engine._config.min_archetype_confidence == 0.55,
          f"obtenu {bot_on._br_engine._config.min_archetype_confidence}")

        _seed_tag(db_on := PlayerDB(db_path), "tag_35", n_hands=35)
        db_on.close()
        bot_on.full_reset(opponent_id="tag_35")
        decision, action = _direct_br_decision(bot_on, POSTFLOP_STATE)
        t("35 mains DB (conf 0.55 franchie) -> Best-Response actif",
          decision.used_fallback is False,
          f"used_fallback={decision.used_fallback}, conf={decision.confidence:.3f}")

        bot_on.full_reset(opponent_id="tag_29")
        db29 = PlayerDB(db_path)
        _seed_tag(db29, "tag_29", n_hands=29)  # sous le plancher des 30 mains
        db29.close()
        bot_on.full_reset(opponent_id="tag_29")
        decision29, _ = _direct_br_decision(bot_on, POSTFLOP_STATE)
        t("29 mains DB (< min_hands_for_best_response=30) -> fallback v3",
          decision29.used_fallback is True, f"used_fallback={decision29.used_fallback}")

        # ── Chemin empirique vs repli, dans le pipeline COMPLET (pas isolé) ──
        _seed_tag(db3 := PlayerDB(db_path), "tag_dense", n_hands=50, dense_buckets=True)
        db3.close()
        bot_on.full_reset(opponent_id="tag_dense")
        decision_dense, _ = _direct_br_decision(bot_on, POSTFLOP_STATE)
        t("buckets densément peuplés -> 100% des combos passent par l'empirique",
          bot_on._br_engine._freq_model.n_empirical_combos > 1000
          and bot_on._br_engine._freq_model.n_fallback_combos == 0,
          f"empirique={bot_on._br_engine._freq_model.n_empirical_combos} "
          f"repli={bot_on._br_engine._freq_model.n_fallback_combos}")

        # ── Aucune exception à travers tout EVCalculator+SizingOptimizer+BluffLayer ──
        t("decision complète retournée sans exception (all_results non vide)",
          len(decision_dense.all_results) > 0, f"obtenu {len(decision_dense.all_results)} résultats")
        t("chaque EVResult a une EV numérique valide",
          all(isinstance(r.ev, (int, float)) for r in decision_dense.all_results))

        bot_on.close()

    print(f"\n  {_passed}/{_passed + _failed} tests passés")
    ok = _failed == 0
    print("  ✅ OK\n" if ok else f"  ⚠ {_failed} échec(s)\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
