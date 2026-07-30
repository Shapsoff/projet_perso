"""
test_db_range_estimator.py — Tests DBAwareRangeEstimator
Phase 6 — Bot Poker Académique

Vérifie :
  1. Compatibilité "drop-in" avec l'interface RangeEstimator (mêmes
     méthodes publiques, même comportement observable pour un appelant
     comme BestResponseEngine.decide()).
  2. Progression correcte à travers les 3 paliers au fil des mains.
  3. Persistance effective : deux instances successives (nouvelle
     session) partagent le même profil pour le même player_id.
  4. Confiance archétype plafonnée au Palier 0 (cf. section 6.1 rapport
     phase 5 — évite l'activation du Best-Response sur un adversaire
     inconnu par artefact de softmax).
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.player_db.player_db import PlayerDB
from core.player_db.db_range_estimator import DBAwareRangeEstimator
from core.bots.range_definitions import ARCHETYPES


def t(label, cond, detail=""):
    global _passed, _failed
    if cond:
        print(f"  ✓ {label}")
        _passed += 1
    else:
        print(f"  ✗ {label}" + (f" : {detail}" if detail else ""))
        _failed += 1


def _play_n_hands_tag(db: PlayerDB, player_id: str, n: int, start: int = 0) -> None:
    """Simule n mains d'un joueur TAG (VPIP≈15%, PFR≈13%, agressif)."""
    for i in range(start, start + n):
        db.new_hand_observed(player_id)
        vpip = (i % 7 == 0)  # ≈ 14%
        db.record_preflop_action(player_id, "CO", vpip=vpip, pfr=vpip)
        db.record_postflop_aggression(player_id, aggressive=True)


def main() -> int:
    global _passed, _failed
    _passed = _failed = 0

    print("\n=== Tests DBAwareRangeEstimator ===\n")

    with tempfile.TemporaryDirectory() as tmp:
        db = PlayerDB(f"{tmp}/db.sqlite3")

        # ── Interface drop-in ────────────────────────────────────────────────
        est = DBAwareRangeEstimator(player_db=db, player_id="villain_1")
        t("get_distribution() disponible", callable(getattr(est, "get_distribution", None)))
        t("get_archetype_probabilities() disponible",
          callable(getattr(est, "get_archetype_probabilities", None)))
        t("observe_action() disponible", callable(getattr(est, "observe_action", None)))
        t("reset()/new_hand()/full_reset() disponibles",
          all(callable(getattr(est, m, None)) for m in ("reset", "new_hand", "full_reset")))

        dist = est.get_distribution()
        total = sum(dist.values())
        t("distribution initiale normalisée (Palier 0)", abs(total - 1.0) < 1e-6,
          f"obtenu {total}")

        # ── Palier 0 : confiance plafonnée ──────────────────────────────────
        probs = est.get_archetype_probabilities()
        t("Palier 0 : toutes les probas d'archétype ≤ 0.40 (cap anti-artefact)",
          all(p <= 0.40 + 1e-9 for p in probs.values()), f"obtenu {probs}")

        # ── Progression Palier 0 → 1 ─────────────────────────────────────────
        _play_n_hands_tag(db, "villain_1", 20)
        est.new_hand()  # force la relecture du profil DB
        t("après 20 mains DB -> tier 1", est.last_profile.tier == 1,
          f"obtenu tier={est.last_profile.tier}")
        t("après 20 mains DB -> archétype inféré = TAG",
          est._archetype == "TAG", f"obtenu {est._archetype}")

        probs1 = est.get_archetype_probabilities()
        t("Palier 1 : TAG a la proba dominante",
          probs1["TAG"] == max(probs1.values()), f"obtenu {probs1}")
        t("Palier 1 : confiance TAG > cap Palier 0 (0.40)",
          probs1["TAG"] > 0.40, f"obtenu {probs1['TAG']:.3f}")

        # ── Progression Palier 1 → 2 + persistance ──────────────────────────
        _play_n_hands_tag(db, "villain_1", 15, start=20)
        for _ in range(22):
            db.record_showdown_combo("villain_1", "AsKs", "AKs", "CO")
        est.new_hand()
        t("après 35 mains DB -> tier 2", est.last_profile.tier == 2,
          f"obtenu tier={est.last_profile.tier}")

        # Le prior doit maintenant surpondérer AsKs par rapport à un
        # estimateur "neuf" sur le même DB (persistance effective)
        dist2 = est.get_distribution()
        est_fresh = DBAwareRangeEstimator(player_db=db, player_id="villain_1")
        dist_fresh = est_fresh.get_distribution()
        t("distribution après reconstruction identique (persistance DB)",
          abs(dist2.get("AsKs", 0) - dist_fresh.get("AsKs", 0)) < 1e-6,
          f"reconstruite={dist2.get('AsKs',0):.5f} fraiche={dist_fresh.get('AsKs',0):.5f}")

        # ── Nouvelle session (réouverture DB) : profil conservé ─────────────
        db.close()
        db2  = PlayerDB(f"{tmp}/db.sqlite3")
        est2 = DBAwareRangeEstimator(player_db=db2, player_id="villain_1")
        t("nouvelle session -> tier 2 conservé", est2.last_profile.tier == 2,
          f"obtenu tier={est2.last_profile.tier}")

        # ── set_player() change bien d'adversaire (sans fuite de profil) ────
        est2.set_player("nouveau_joueur_jamais_vu")
        est2.full_reset()
        t("set_player() + full_reset() -> nouveau joueur au Palier 0",
          est2.last_profile.tier == 0, f"obtenu tier={est2.last_profile.tier}")
        db2.close()

    # ── Régression : _archetype doit être correct DÈS full_reset(), avant
    # tout appel à new_hand() (sinon la 1ère main d'un match utiliserait
    # 'LAG' par défaut même face à un adversaire connu de longue date) ────
    with tempfile.TemporaryDirectory() as tmp2:
        db3 = PlayerDB(f"{tmp2}/db3.sqlite3")
        _play_n_hands_tag(db3, "vieux_tag", 40)  # bien au Palier 2

        est3 = DBAwareRangeEstimator(player_db=db3, player_id="vieux_tag")
        t("archétype correct DÈS la construction, sans new_hand() préalable",
          est3._archetype == "TAG", f"obtenu {est3._archetype}")

        # Simuler un changement de match vers un AUTRE adversaire déjà connu,
        # sans jamais appeler new_hand() entre les deux
        est3.set_player("vieux_tag")  # même joueur, mais on force le chemin full_reset
        est3.full_reset()
        t("archétype correct DÈS full_reset(), avant le premier new_hand()",
          est3._archetype == "TAG", f"obtenu {est3._archetype}")
        db3.close()

    print(f"\n  {_passed}/{_passed + _failed} tests passés")
    ok = _failed == 0
    print("  ✅ OK\n" if ok else f"  ⚠ {_failed} échec(s)\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
