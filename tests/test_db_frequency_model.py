"""
test_db_frequency_model.py — Tests DBAwareFrequencyModel
Phase 6 — Bot Poker Académique

Utilise un calculateur EHS stub (déterministe, sans poker_engine C++) —
même convention que les tests légers déjà présents dans le projet
(ex: benchmark_frequency_model.py, test_bluff_layer.py mentionnés dans
l'arborescence). L'EHS stub retourne directement des valeurs choisies pour
piloter précisément quels combos tombent dans quels buckets.
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.player_db.player_db import PlayerDB, ehs_to_bucket
from core.player_db.profile_builder import build_profile, PlayerProfile
from core.player_db.db_frequency_model import DBAwareFrequencyModel


def t(label, cond, detail=""):
    global _passed, _failed
    if cond:
        print(f"  ✓ {label}")
        _passed += 1
    else:
        print(f"  ✗ {label}" + (f" : {detail}" if detail else ""))
        _failed += 1


class _StubEHSResult:
    def __init__(self, ehs):
        self.EHS = ehs


class _StubEHSCalculator:
    """
    Calculateur EHS stub : renvoie un EHS déterministe selon la carte
    haute du combo, pour piloter précisément les buckets dans les tests
    sans dépendre du moteur C++.
    """
    _RANK_VAL = {r: i for i, r in enumerate(
        ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    )}

    def calculate_multiway(self, hand, board, n_opponents, n_sims):
        r1, r2 = hand[0][0], hand[1][0]
        high = max(self._RANK_VAL.get(r1, 6), self._RANK_VAL.get(r2, 6))
        # Mappe la carte haute (0..12) sur un EHS grossier 0.1..0.95
        ehs = 0.1 + (high / 12.0) * 0.85
        return _StubEHSResult(ehs)


def main() -> int:
    global _passed, _failed
    _passed = _failed = 0

    print("\n=== Tests DBAwareFrequencyModel ===\n")

    with tempfile.TemporaryDirectory() as tmp:
        db = PlayerDB(f"{tmp}/db.sqlite3")
        stub_ehs = _StubEHSCalculator()

        # ── Sans profil actif -> comportement phase 5 exact (fallback) ──────
        model = DBAwareFrequencyModel(ehs_calculator=stub_ehs, n_sims=10, player_db=db)
        distribution = {"AhAd": 0.5, "7c2d": 0.5}
        result = model.compute(
            distribution=distribution, archetype="TAG", board=["Kh", "5d", "2s"],
            bet_sizing=0.75, pot=100, n_opponents=1, street="flop",
        )
        t("sans profil actif -> résultat cohérent (fallback phase 5)",
          abs(result.p_fold + result.p_call + result.p_raise - 1.0) < 0.01,
          f"obtenu {result.summary()}")
        t("sans profil actif -> aucun combo empirique",
          model.n_empirical_combos == 0, f"obtenu {model.n_empirical_combos}")

        # ── Construire un joueur Palier 2 avec des fréquences data-driven ───
        # Bucket EHS élevé (As haute -> ehs≈0.928 -> bucket 9) : le joueur
        # RAISE quasi systématiquement (contrairement à un TAG standard qui
        # raise mais foldrait moins que ça face à lui-même...). On choisit
        # des fréquences délibérément DIFFÉRENTES du modèle archétype pour
        # vérifier que c'est bien la donnée empirique qui est utilisée.
        for _ in range(30):
            db.record_ehs_bucket_action("villain_data", "flop", ehs=0.93,
                                         facing_bet=True, action="fold")
        for _ in range(10):
            db.record_ehs_bucket_action("villain_data", "flop", ehs=0.93,
                                         facing_bet=True, action="call")
        # -> à ce bucket, un joueur "nitty" qui fold même des mains fortes
        #    face à une mise (comportement improbable pour un TAG standard,
        #    volontairement choisi pour être détectable dans le test)

        for i in range(35):
            db.new_hand_observed("villain_data")
            db.record_preflop_action("villain_data", "CO", vpip=(i % 6 == 0),
                                      pfr=(i % 6 == 0))

        profile = build_profile(db, "villain_data")
        t("profil de test au Palier 2", profile.tier == 2, f"obtenu tier={profile.tier}")

        bucket_check = ehs_to_bucket(0.93)
        t("bucket EHS=0.93 présent dans le profil",
          profile.has_empirical_bucket("flop", bucket_check, True))

        model.set_active_profile(profile)
        distribution2 = {"AhAd": 1.0}  # As haute -> ehs≈0.928 avec le stub
        result2 = model.compute(
            distribution=distribution2, archetype=profile.archetype,
            board=["Kh", "5d", "2s"], bet_sizing=0.75, pot=100,
            n_opponents=1, street="flop",
        )
        t("profil Palier 2 actif -> combo classé en empirique",
          model.n_empirical_combos == 1 and model.n_fallback_combos == 0,
          f"empirique={model.n_empirical_combos} fallback={model.n_fallback_combos}")
        # 30 fold / 10 call / 0 raise (+ Laplace) -> fold très dominant,
        # à l'opposé du modèle archétype TAG standard (qui raiserait une
        # main aussi forte face à une mise)
        t("fréquence empirique -> fold dominant (comportement réel observé)",
          result2.p_fold > result2.p_raise,
          f"obtenu {result2.summary()}")

        # ── Bucket sans données -> repli archétype (in_range doit être correct) ──
        distribution3 = {"7c2d": 1.0}  # carte basse -> bucket sans données
        result3 = model.compute(
            distribution=distribution3, archetype=profile.archetype,
            board=["Kh", "5d", "2s"], bet_sizing=0.75, pot=100,
            n_opponents=1, street="flop",
        )
        t("bucket sans données -> repli archétype (pas empirique)",
          model.n_fallback_combos == 1 and model.n_empirical_combos == 0,
          f"empirique={model.n_empirical_combos} fallback={model.n_fallback_combos}")

        # ── Régression ciblée : in_range doit venir de la vraie range figée
        # de l'archétype, PAS d'un proxy sur la masse de distribution.
        # Avec un prior réellement lissé (get_preflop_prior, celui utilisé
        # en prod), TOUT combo porte une masse non nulle même hors range
        # — un check naïf "masse >= seuil" serait donc presque toujours
        # vrai et rendrait in_range inopérant. On le vérifie avec K7o : hors
        # range TAG, mais que le calculateur EHS stub (basé sur la carte
        # haute uniquement) score artificiellement haut à cause du Roi.
        from core.player_db.profile_builder import get_preflop_prior
        from core.player_db.player_db import canonical_combo

        real_dist = get_preflop_prior(profile, db)
        combo_k7o = canonical_combo("Kc", "7d")
        dist_k7o  = {combo_k7o: real_dist[combo_k7o]}

        t("K7o porte une masse non nulle dans le prior lissé (piège du proxy)",
          real_dist[combo_k7o] > 1e-6, f"obtenu {real_dist[combo_k7o]:.2e}")

        result_k7o = model.compute(
            distribution=dist_k7o, archetype=profile.archetype,
            board=["2h", "5d", "8s"], bet_sizing=0.75, pot=100,
            n_opponents=1, street="turn",  # bucket sans donnée -> repli garanti
        )
        t("K7o hors-range TAG en repli -> fold garanti malgré EHS stub élevé",
          result_k7o.p_fold == 1.0, f"obtenu {result_k7o.summary()}")

        db.close()

    print(f"\n  {_passed}/{_passed + _failed} tests passés")
    ok = _failed == 0
    print("  ✅ OK\n" if ok else f"  ⚠ {_failed} échec(s)\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
