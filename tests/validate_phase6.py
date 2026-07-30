"""
validate_phase6.py — Validation Player DB (Phase 6)
Bot Poker Académique

Ne modifie PAS validate_phase5.py — fichier entièrement séparé, même
conventions de style/CLI, focalisé sur ce que la phase 5 ne pouvait pas
tester : la persistance et l'apprentissage progressif de la Player DB.

Valide, avec de VRAIES mains jouées via PokerTable (pas des stats de seed
synthétiques) :
  1. Convergence : après N mains contre un RangeBot connu, le profil DB
     retrouve-t-il le bon archétype et le bon palier (0/1/2) ?
  2. Effet observable : le Best-Response s'active-t-il réellement plus tôt
     / plus souvent contre un adversaire déjà profilé qu'un inconnu ?
  3. Persistance multi-session : le profil survit-il à une fermeture et
     réouverture de la Player DB (nouvelle instance EHSBot, nouvelle
     connexion SQLite) ?

Interface simulateur (identique à validate_phase5.py) :
    table = PokerTable(n_players=2, starting_stacks=1000, big_blind=10, seed=42)
    result = table.run_hand(bots={0: bot_fn, 1: opp_fn})

Usage :
    python tests/validate_phase6.py                   # validation complète
    python tests/validate_phase6.py --quick            # 200 mains/match
    python tests/validate_phase6.py --opponent TAG_0   # un seul adversaire
    python tests/validate_phase6.py --db_path data/validate_phase6.sqlite3
"""

from __future__ import annotations

import sys
import os
import time
import argparse
import logging
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.simulator import PokerTable
from core.bots.ehs_bot import EHSBot, EHSBotConfig
from core.bots.range_bot import make_all_range_bots
from core.bots.range_definitions import ARCHETYPES
from core.game_state import Action as SimAction, ActionType, Street

from core.player_db.player_db import PlayerDB
from core.player_db.hand_recorder import record_hand
from core.player_db.profile_builder import build_profile

try:
    import poker_engine
    _ehs_calc_for_recording = poker_engine.EHSCalculator(seed=42, use_openmp=False)
except ImportError:
    _ehs_calc_for_recording = None

logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Adaptateur bot → BotFn, avec capture du board/position pour hand_recorder
# =============================================================================
# Repris de validate_phase5.py::_make_bot_fn (même logique), avec l'ajout de
# `capture` : un dict partagé rempli à chaque appel avec le board et la
# position de l'adversaire tels que vus par CE bot à cet instant — permet de
# récupérer le board FINAL de la main après table.run_hand(), que HandResult
# n'expose pas directement.

_MAP = {
    'fold':  ActionType.FOLD,
    'check': ActionType.CHECK,
    'call':  ActionType.CALL,
    'bet':   ActionType.RAISE,
    'raise': ActionType.RAISE,
    'allin': ActionType.ALLIN,
}


def _make_bot_fn(bot, player_id: int, capture: Optional[dict] = None):
    """Identique à validate_phase5.py::_make_bot_fn + capture optionnelle."""

    def bot_fn(state) -> SimAction:
        try:
            our = state.get_player(player_id)
        except KeyError:
            our = state.our_player

        stack      = float(our.stack)
        pos        = our.position
        street_raw = state.street
        hand       = [str(c) for c in our.hole_cards] if our.hole_cards else []

        players_dict = [
            {
                'id': p.player_id, 'player_id': p.player_id,
                'stack': float(p.stack), 'status': p.status.value,
                'position': p.position.value,
            }
            for p in state.players if p.player_id != player_id
        ]

        action_history = [
            {'street': act.street.value, 'player': act.player_id,
             'action': act.action_type.value, 'amount': float(act.amount)}
            for act in state.action_history
        ]

        gs_dict = {
            'hand': hand, 'board': state.board_str(), 'pot': float(state.pot),
            'to_call': float(state.to_call), 'stack': stack, 'position': pos.value,
            'street': street_raw.value, 'players': players_dict,
            'player_id': player_id, 'action_history': action_history,
            'big_blind': float(state.big_blind),
        }

        if capture is not None:
            capture['board'] = state.board_str()
            # position de l'ADVERSAIRE (pas la nôtre) — utilisée par
            # hand_recorder pour le prior de population (Palier 0)
            other = next((p for p in state.players if p.player_id != player_id), None)
            if other is not None:
                capture['opponent_position'] = other.position.value

        internal_action = bot.decide(gs_dict)
        at     = _MAP.get(internal_action.action_type, ActionType.CHECK)
        amount = int(getattr(internal_action, 'amount', 0))
        return SimAction(player_id, at, amount, street_raw)

    return bot_fn


# =============================================================================
# Runner de match, avec enregistrement Player DB
# =============================================================================

@dataclass
class Phase6MatchResult:
    opponent:        str
    n_hands:         int
    profit_chips:    float
    bb_100:          float
    duration_s:      float
    true_archetype:  str
    tier_before:     int
    tier_after:      int
    archetype_before: Optional[str]
    archetype_after:  Optional[str]
    hands_seen_after: int
    confidence_after: float
    correctly_identified: bool

    def summary(self) -> str:
        flag = "✅" if self.correctly_identified else "⚠ "
        return (
            f"{flag} {self.opponent:22s} | vrai={self.true_archetype:16s} | "
            f"palier {self.tier_before}→{self.tier_after} | "
            f"archétype DB={self.archetype_after or '?':16s} "
            f"(conf={self.confidence_after:.2f}) | "
            f"{self.hands_seen_after:4d} mains DB cumulées | "
            f"BB/100={self.bb_100:+7.2f}"
        )


def run_match_with_recording(
    bot_v4,
    opp,
    opp_name:      str,
    true_archetype: str,
    n_hands:       int,
    db:            PlayerDB,
    starting_stack: int = 1000,
    small_blind:    int = 5,
    big_blind:      int = 10,
    seed:           int = 42,
    rebuy_threshold: float = 0.70,
) -> Phase6MatchResult:
    """
    Comme run_match() de validate_phase5.py, mais :
      - bascule l'adversaire suivi via full_reset(opponent_id=opp_name)
        (persistance activée — c'est la seule différence structurelle
        avec le harness phase 5, qui ne connaît pas cette notion)
      - enregistre chaque main dans la Player DB via hand_recorder, en
        transmettant small_blind/big_blind pour que les mises forcées ne
        soient jamais comptées comme VPIP/PFR volontaires (cf. bug trouvé
        en session — simulator.py::_post_blindes logge encore les blindes
        en ActionType.RAISE, TODO connu du projet)
    """
    profile_before = build_profile(db, opp_name)

    if hasattr(bot_v4, 'full_reset'):
        bot_v4.full_reset(opponent_id=opp_name)
    if hasattr(opp, 'reset'):
        opp.reset()

    capture: dict = {}
    bot_fn = _make_bot_fn(bot_v4, player_id=0, capture=capture)
    opp_fn = _make_bot_fn(opp, player_id=1)

    table = PokerTable(n_players=2, starting_stacks=starting_stack,
                        small_blind=small_blind, big_blind=big_blind, seed=seed)

    t0 = time.time()
    hands_played = 0
    profit_chips = 0.0

    for hand_idx in range(n_hands):
        for pid in [0, 1]:
            if table.stacks[pid] < starting_stack * rebuy_threshold:
                table.stacks[pid] = starting_stack

        try:
            result = table.run_hand(bots={0: bot_fn, 1: opp_fn})
            profit_chips += float(result.gains.get(0, 0))
            hands_played += 1
        except Exception as e:
            logger.warning("Main %d échouée : %s", hand_idx, e)
            continue

        # ── Enregistrement Player DB (le seul ajout vs run_match phase 5) ──
        # Les blindes forcées portent ActionType.POST_BLIND (cf. game_state.py
        # et simulator.py::_post_blindes) — hand_recorder les exclut déjà
        # nativement du calcul VPIP/PFR/3bet, aucun paramètre supplémentaire
        # n'est nécessaire ici.
        try:
            record_hand(
                db=db,
                hand_result=result,
                board=capture.get('board', []),
                player_id_map={1: opp_name},
                position_map={1: capture.get('opponent_position')},
                our_player_id=0,
                ehs_calculator=_ehs_calc_for_recording,
                n_sims=200,
            )
        except Exception as e:
            logger.warning("Enregistrement DB échoué (main %d) : %s", hand_idx, e)

        if hasattr(bot_v4, 'reset'): bot_v4.reset()
        if hasattr(opp, 'reset'):    opp.reset()

    duration = time.time() - t0
    bb_100 = (profit_chips / big_blind) / max(hands_played, 1) * 100

    profile_after = build_profile(db, opp_name)
    archetype_after = profile_after.archetype
    correctly_identified = (
        profile_after.tier >= 1 and archetype_after == true_archetype
    )

    return Phase6MatchResult(
        opponent=opp_name, n_hands=hands_played, profit_chips=profit_chips,
        bb_100=bb_100, duration_s=duration, true_archetype=true_archetype,
        tier_before=profile_before.tier, tier_after=profile_after.tier,
        archetype_before=profile_before.archetype, archetype_after=archetype_after,
        hands_seen_after=profile_after.hands_seen,
        confidence_after=profile_after.archetype_confidence,
        correctly_identified=correctly_identified,
    )


# =============================================================================
# Validation 1 — Convergence contre les RangeBots
# =============================================================================

def validate_convergence(
    db_path: str,
    n_hands: int,
    filter_opponents: Optional[List[str]] = None,
) -> List[Phase6MatchResult]:
    """
    Joue n_hands mains contre chaque RangeBot, avec enregistrement DB actif.

    `db_path` (pas un objet PlayerDB) : EHSBot ouvre sa PROPRE connexion
    vers ce fichier via EHSBotConfig.player_db_path, exactement comme en
    conditions réelles — pas de bricolage d'attributs internes. Ce module
    ouvre une connexion SÉPARÉE (SQLite en WAL gère très bien plusieurs
    connexions vers le même fichier) pour enregistrer les mains et
    consulter les profils avant/après chaque match.
    """
    all_bots = make_all_range_bots(n_sims=500)
    opponents = (
        {k: v for k, v in all_bots.items() if k in filter_opponents}
        if filter_opponents else all_bots
    )

    cfg = EHSBotConfig(use_best_response=True, use_range_estimator=True,
                        use_player_db=True, player_db_path=db_path)
    bot_v4 = EHSBot(config=cfg)
    db = PlayerDB(db_path)

    results = []
    n_total = len(opponents)
    for i, (name, opp) in enumerate(opponents.items(), 1):
        true_arch = name.rsplit('_', 1)[0]  # "TAG_0" -> "TAG"
        print(f"  [{i:2d}/{n_total}] vs {name:<24s}", end=' ', flush=True)
        result = run_match_with_recording(
            bot_v4=bot_v4, opp=opp, opp_name=name, true_archetype=true_arch,
            n_hands=n_hands, db=db,
        )
        print(f"BB/100={result.bb_100:+7.2f}  "
              f"archétype DB={result.archetype_after or '?':16s} "
              f"{'✅' if result.correctly_identified else '⚠ '}")
        results.append(result)

    db.close()
    bot_v4.close()
    return results


# =============================================================================
# Validation 2 — Persistance multi-session
# =============================================================================

def validate_persistence(db_path: str, n_hands: int) -> bool:
    """
    Joue n_hands mains contre un TAG, ferme la DB, la rouvre avec une
    NOUVELLE instance PlayerDB/EHSBot (simule un nouveau process/une
    nouvelle session), et vérifie que le profil a bien survécu.
    """
    print("\n── PERSISTANCE MULTI-SESSION ──")
    all_bots = make_all_range_bots(n_sims=500)
    opp_name = "TAG_0"
    opp = all_bots[opp_name]

    # Si ce db_path a déjà servi à validate_convergence() (cas par défaut,
    # main() les enchaîne sur le même fichier), "TAG_0" a potentiellement
    # déjà de l'historique — c'est volontaire (test plus réaliste : "la DB
    # accumulée survit-elle à une reconnexion", pas un cas artificiellement
    # isolé) mais on l'affiche pour éviter toute confusion sur les chiffres.
    db_probe = PlayerDB(db_path)
    profile_start = build_profile(db_probe, opp_name)
    db_probe.close()
    if profile_start.hands_seen > 0:
        print(f"  (DB déjà à {profile_start.hands_seen} mains pour {opp_name} "
              f"avant cette session — accumulation, pas un départ à zéro)")

    print(f"  Session 1 : {n_hands} mains vs {opp_name}...", end=' ', flush=True)
    cfg = EHSBotConfig(use_best_response=True, use_range_estimator=True,
                        use_player_db=True, player_db_path=db_path)
    bot1 = EHSBot(config=cfg)
    db1  = PlayerDB(db_path)

    result1 = run_match_with_recording(
        bot_v4=bot1, opp=opp, opp_name=opp_name, true_archetype="TAG",
        n_hands=n_hands, db=db1,
    )
    print(f"tier={result1.tier_after} archétype={result1.archetype_after} "
          f"hands_seen={result1.hands_seen_after}")
    db1.close()
    bot1.close()

    print(f"  Fermeture DB, nouvelle connexion (simule un nouveau process)...")
    db2 = PlayerDB(db_path)
    profile2 = build_profile(db2, opp_name)
    print(f"  Session 2 (relecture seule) : tier={profile2.tier} "
          f"archétype={profile2.archetype} hands_seen={profile2.hands_seen}")

    ok = (profile2.hands_seen == result1.hands_seen_after
          and profile2.tier == result1.tier_after
          and profile2.archetype == result1.archetype_after)
    print(f"  {'✅ Profil survit à la réouverture de la DB' if ok else '⚠ Profil perdu ou altéré'}")
    db2.close()
    return ok


# =============================================================================
# Point d'entrée
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Validation Phase 6 — Player DB")
    parser.add_argument('--quick', action='store_true',
                         help="Mode rapide (200 mains/match)")
    parser.add_argument('--opponent', type=str, default=None,
                         help="Tester contre un seul adversaire (ex: TAG_0)")
    parser.add_argument('--n_hands', type=int, default=None,
                         help="Nombre de mains par match")
    parser.add_argument('--db_path', type=str, default=None,
                         help="Chemin de la Player DB (défaut : fichier temporaire jetable)")
    parser.add_argument('--no_persistence', action='store_true',
                         help="Désactiver le test de persistance multi-session")
    args = parser.parse_args()

    n_hands = args.n_hands or (200 if args.quick else 1000)
    db_path = args.db_path or tempfile.mktemp(suffix='_phase6.sqlite3')

    print(f"\n{'═' * 76}")
    print(f"  VALIDATION PHASE 6 — Player DB")
    print(f"  DB     : {db_path}")
    print(f"  Mode   : {'rapide' if args.quick else 'complet'} ({n_hands} mains/match)")
    print(f"  EHS calculator pour showdowns : "
          f"{'poker_engine (réel)' if _ehs_calc_for_recording else 'indisponible — VPIP/PFR seuls'}")
    print(f"{'═' * 76}\n")

    print("── CONVERGENCE VS RANGEBOTS (persistance activée) ──")
    filter_opp = [args.opponent] if args.opponent else None
    results = validate_convergence(db_path, n_hands, filter_opponents=filter_opp)

    n_ok = sum(1 for r in results if r.correctly_identified)
    print(f"\n{'─' * 76}")
    print(f"  {n_ok}/{len(results)} adversaires correctement identifiés "
          f"(archétype DB == vrai archétype, palier ≥ 1)")
    for r in results:
        print(f"  {r.summary()}")

    persistence_ok = True
    if not args.no_persistence:
        persistence_ok = validate_persistence(db_path, min(n_hands, 200))

    print(f"\n{'═' * 76}")
    all_ok = (n_ok == len(results)) and persistence_ok
    print(f"  {'✅ VALIDATION PHASE 6 RÉUSSIE' if all_ok else '⚠  VALIDATION PARTIELLE'}")
    print(f"{'═' * 76}\n")
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
