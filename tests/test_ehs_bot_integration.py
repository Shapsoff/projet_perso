"""
test_ehs_bot_integration.py — Tests de non-régression EHSBot × Player DB
Phase 6 — Bot Poker Académique

Verrouille 3 problèmes trouvés lors de la relecture post-livraison :

  1. full_reset() sans opponent_id doit générer une identité jetable
     DISTINCTE à chaque appel — jamais réutiliser "unknown" ni conserver
     l'adversaire précédent. Sans ce garde-fou, un harness non modifié
     (ex: validate_phase5.py, qui appelle bot.full_reset() sans argument
     entre CHAQUE match contre un archétype différent) fusionnerait
     silencieusement les stats de plusieurs adversaires distincts dans
     une seule entrée DB.

  2. _effective_hand_count() doit refléter le nombre de mains CUMULÉ en
     DB pour l'adversaire suivi, pas seulement le compteur local au
     match — sinon un adversaire recroisé après une pause, pourtant déjà
     au Palier 2 en DB, redémarrerait à tort en fallback jusqu'à rejouer
     tous les seuils d'activation localement.

  3. _villain_position ne doit jamais "fuiter" d'un adversaire au
     suivant : changer d'adversaire (set_player) doit effacer la
     position mémorisée, et chaque main doit la ré-alimenter depuis le
     game_state courant.

N'a besoin QUE de core.range_estimator / core.bots.range_definitions /
core.player_db — pas de poker_engine, ev_calculator, sizing_optimizer ni
bluff_layer (utilise use_best_response=False, chemin v3 uniquement).
Nécessite des stubs locaux pour core.board_texture / core.spr / core.deck
si ces fichiers ne sont pas encore présents dans l'environnement de test
(le vrai projet les a déjà).
"""

import sys
import os
import tempfile
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.bots.ehs_bot import EHSBot, EHSBotConfig
from core.player_db.player_db import PlayerDB


def t(label, cond, detail=""):
    global _passed, _failed
    if cond:
        print(f"  ✓ {label}")
        _passed += 1
    else:
        print(f"  ✗ {label}" + (f" : {detail}" if detail else ""))
        _failed += 1


def main() -> int:
    global _passed, _failed
    _passed = _failed = 0

    print("\n=== Tests de non-régression EHSBot × Player DB ===\n")

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.sqlite3")

        # ── Bug #1 : full_reset() sans opponent_id (pattern validate_phase5.py) ──
        cfg = EHSBotConfig(use_player_db=True, player_db_path=db_path,
                            use_best_response=False)
        bot = EHSBot(cfg)

        opponents = ["TAG_0", "LAG_1", "CALLING_STATION_2", "TAG_0"]  # "TAG_0" répété !
        seen_ids = []
        for name in opponents:
            bot.full_reset()  # tel qu'appelé par validate_phase5.py, ligne 428
            seen_ids.append(bot._estimator.player_id)

        t("4 appels full_reset() sans opponent_id -> 4 identités distinctes",
          len(set(seen_ids)) == 4, f"obtenu {seen_ids}")
        t("aucune identité n'est le literal 'unknown'",
          all(pid != "unknown" for pid in seen_ids), f"obtenu {seen_ids}")

        db_check = PlayerDB(db_path)
        t("aucune pollution DB tant que hand_recorder n'est pas appelé",
          db_check.list_players() == [], f"obtenu {db_check.list_players()}")
        db_check.close()

        # ── Bug #2 : hand_count DB cumulé doit primer sur le compteur local ──
        seed_db = PlayerDB(db_path)
        for i in range(50):
            seed_db.new_hand_observed("returning_player")
            seed_db.record_preflop_action("returning_player", "CO",
                                           vpip=(i % 4 == 0), pfr=(i % 4 == 0))
        seed_db.close()

        bot.full_reset(opponent_id="returning_player")
        t("hand_count local repart à 0 après full_reset()", bot._hand_count == 0)
        t("effective_hand_count reflète les 50 mains DB malgré hand_count local=0",
          bot._effective_hand_count() == 50, f"obtenu {bot._effective_hand_count()}")

        # Un adversaire réellement neuf ne doit PAS hériter de ce total
        bot.full_reset(opponent_id="jamais_vu_avant")
        t("nouvel adversaire jamais vu -> effective_hand_count = 0",
          bot._effective_hand_count() == 0, f"obtenu {bot._effective_hand_count()}")

        # ── Bug #3 : la position ne doit pas fuiter d'un adversaire à l'autre ──
        bot.full_reset(opponent_id="joueur_A")
        bot.set_opponent("joueur_A", position="UTG")
        t("position mémorisée pour joueur_A", bot._estimator._villain_position == "UTG")

        bot.full_reset(opponent_id="joueur_B")  # nouvel adversaire, pas de position fournie
        t("position remise à None pour un nouvel adversaire (pas de fuite UTG)",
          bot._estimator._villain_position is None,
          f"obtenu {bot._estimator._villain_position!r}")

        # Simuler une main : decide() doit ré-alimenter automatiquement la position
        # à partir du game_state (sans appel manuel à set_opponent).
        game_state = {
            "hand": ["Ah", "Kd"], "board": [], "position": "BTN", "pot": 3,
            "to_call": 2, "stack": 100,
            "players": [
                {"id": 0, "position": "BTN", "status": "active"},
                {"id": 1, "position": "BB",  "status": "active"},
            ],
            "our_id": 0,
        }
        bot.decide(game_state)
        t("decide() alimente automatiquement la position adverse (BB) depuis game_state",
          bot._estimator._villain_position == "BB",
          f"obtenu {bot._estimator._villain_position!r}")

        # ── Pipeline postflop complet (vrai classify_board + vrai SPRInfo) ──
        # Board non vide -> exerce classify_board() et get_spr_info_from_game_state()
        # réels (pas de stub) — vérifie qu'aucune exception n'est levée et
        # que la confiance archétype se comporte comme attendu de bout en
        # bout à travers tout le pipeline Dimension 3/4/5.
        seed = PlayerDB(db_path)
        for i in range(40):
            seed.new_hand_observed("tag_postflop_test")
            vpip = (i % 7 == 0)
            seed.record_preflop_action("tag_postflop_test", "CO", vpip=vpip, pfr=vpip)
            seed.record_postflop_aggression("tag_postflop_test", aggressive=True)
        seed.close()

        postflop_state = {
            "hand": ["Ah", "Kd"], "board": ["Qh", "7d", "2c"], "position": "BTN",
            "pot": 20, "to_call": 0, "stack": 180,
            "players": [
                {"id": 0, "position": "BTN", "status": "active", "stack": 180},
                {"id": 1, "position": "BB",  "status": "active", "stack": 150},
            ],
            "our_id": 0, "street": "flop",
        }

        bot.full_reset()  # adversaire inconnu, Palier 0
        bot.decide(dict(postflop_state))
        probs_unknown = bot._estimator.get_archetype_probabilities()
        t("postflop réel, adversaire inconnu -> confiance plafonnée ≤ 0.40",
          all(p <= 0.40 + 1e-9 for p in probs_unknown.values()),
          f"obtenu {probs_unknown}")

        bot.full_reset(opponent_id="tag_postflop_test")
        bot.decide(dict(postflop_state))  # ne doit lever aucune exception
        probs_known = bot._estimator.get_archetype_probabilities()
        t("postflop réel, adversaire TAG connu (40 mains DB) -> confiance > seuil d'activation (0.50)",
          probs_known["TAG"] > 0.50, f"obtenu {probs_known}")
        t("confiance sur adversaire connu > confiance sur inconnu",
          probs_known["TAG"] > max(probs_unknown.values()),
          f"connu={probs_known['TAG']:.2f} inconnu max={max(probs_unknown.values()):.2f}")

        bot.close()

    print(f"\n  {_passed}/{_passed + _failed} tests passés")
    ok = _failed == 0
    print("  ✅ OK\n" if ok else f"  ⚠ {_failed} échec(s)\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
