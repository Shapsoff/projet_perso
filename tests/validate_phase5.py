"""
validate_phase5.py — Self-play + Validation Phase 5 (P4)
Bot Poker Académique

Valide le Best-Response Engine (Phase 5) contre :
  1. Les 9 RangeBots (vérité terrain phase 3/4)
  2. L'EHSBot v3 (baseline interne — use_best_response=False)
  3. Une instance miroir du Best-Response (mesure d'exploitabilité)

Interface simulateur utilisée :
    table = PokerTable(n_players=2, starting_stacks=1000, big_blind=10, seed=42)
    stats = table.run_session(
        bots={0: bot_v4.get_action, 1: range_bot.get_action},
        n_hands=2000,
    )
    # stats['bb_per_100'][0]  → BB/100 du joueur 0
    # stats['gains_total'][0] → gain net en chips du joueur 0
    # stats['hands_played']   → mains effectivement jouées

Métriques P4 :
  - BB/100 vs chacun des 9 RangeBots
  - BB/100 moyen par archétype (TAG / LAG / CALLING_STATION)
  - BB/100 v4 vs v3 (mesure du gain apporté par le Best-Response)
  - Exploitabilité ≈ |BB/100| en self-play (devrait être proche de 0)

Seuils de validation :
  - BB/100 moyen vs RangeBots ≥ 5.0 BB/100
  - Exploitabilité ≤ 10.0 BB/100

Usage :
    python tests/validate_phase5.py                  # validation complète
    python tests/validate_phase5.py --quick           # 500 mains/match
    python tests/validate_phase5.py --opponent TAG_0  # un seul opponent
    python tests/validate_phase5.py --self_play_only  # exploitabilité seule
    python tests/validate_phase5.py --no_self_play    # sans self-play
    python tests/validate_phase5.py --vs_v3           # ajouter v4 vs v3
"""

from __future__ import annotations

import sys
import os
import time
import argparse
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.simulator import PokerTable
from core.bots.ehs_bot import EHSBot, EHSBotConfig
from core.bots.range_bot import make_all_range_bots
from core.bots.range_definitions import ARCHETYPES, MUTATIONS
from core.game_state import Action as SimAction, ActionType, Street


# =============================================================================
# Adaptateur bot → BotFn compatible simulateur
# =============================================================================

def _make_bot_fn(bot, player_id: int) -> 'BotFn':
    """
    Crée une BotFn compatible simulateur depuis un bot quelconque.

    player_id : l'ID de ce bot dans la table (0 ou 1).
                Chaque bot doit connaître son propre player_id pour
                extraire correctement sa main et son stack depuis GameState.

    Le simulateur attend : GameState → Action (objet SimAction)
    Les bots existants ont : decide(dict) → Action interne
    """
    _MAP = {
        'fold':  ActionType.FOLD,
        'check': ActionType.CHECK,
        'call':  ActionType.CALL,
        'bet':   ActionType.RAISE,
        'raise': ActionType.RAISE,
        'allin': ActionType.ALLIN,
    }

    def bot_fn(state) -> SimAction:
        # Récupérer notre joueur selon notre player_id réel
        try:
            our = state.get_player(player_id)
        except KeyError:
            # Fallback sur our_player si player_id introuvable
            our = state.our_player

        stack     = float(our.stack)
        pos       = our.position
        street_raw = state.street

        # Notre main
        hand = [str(c) for c in our.hole_cards] if our.hole_cards else []

        # Adversaires = tous sauf nous
        players_dict = [
            {
                'id':        p.player_id,
                'player_id': p.player_id,
                'stack':     float(p.stack),
                'status':    p.status.value,
                'position':  p.position.value,
            }
            for p in state.players if p.player_id != player_id
        ]

        action_history = []
        for act in state.action_history:
            action_history.append({
                'street': act.street.value,
                'player': act.player_id,
                'action': act.action_type.value,
                'amount': float(act.amount),
            })

        gs_dict = {
            'hand':           hand,
            'board':          state.board_str(),
            'pot':            float(state.pot),
            'to_call':        float(state.to_call),
            'stack':          stack,
            'position':       pos.value,
            'street':         street_raw.value,
            'players':        players_dict,
            'player_id':      player_id,
            'action_history': action_history,
            'big_blind':      float(state.big_blind),
        }

        # Appeler decide()
        internal_action = bot.decide(gs_dict)

        # Convertir en SimAction
        at     = _MAP.get(internal_action.action_type, ActionType.CHECK)
        amount = int(getattr(internal_action, 'amount', 0))
        return SimAction(player_id, at, amount, street_raw)

    return bot_fn

logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s %(name)s %(message)s',
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ValidationConfig:
    """Paramètres de la session de validation."""
    n_hands_vs_rangebot:    int   = 2000
    n_hands_self_play:      int   = 5000
    n_hands_quick:          int   = 500
    big_blind:              int   = 10
    starting_stack:         int   = 1000
    seed:                   int   = 42
    # Seuils de validation
    min_bb100_vs_rangebot:  float = 5.0
    max_exploitability:     float = 50.0   # bot exploitatif → pas GTO


# =============================================================================
# Résultats
# =============================================================================

@dataclass
class MatchResult:
    """Résultat d'un match entre deux bots."""
    bot_name:     str
    opponent:     str
    n_hands:      int
    profit_chips: float
    bb_100:       float
    duration_s:   float

    def summary(self) -> str:
        return (
            f"{self.bot_name:20s} vs {self.opponent:22s} | "
            f"{self.n_hands:5d} mains | "
            f"BB/100={self.bb_100:+7.2f} | "
            f"profit={self.profit_chips:+9.1f} chips | "
            f"{self.duration_s:.1f}s"
        )


@dataclass
class ValidationReport:
    """Rapport complet de validation Phase 5."""
    matches:                List[MatchResult]          = field(default_factory=list)
    self_play_result:       Optional[MatchResult]      = None
    v3_baseline_result:     Optional[MatchResult]      = None
    config:                 ValidationConfig           = field(default_factory=ValidationConfig)
    avg_bb100_vs_rangebots: float                      = 0.0
    avg_bb100_by_archetype: Dict[str, float]           = field(default_factory=dict)
    exploitability:         float                      = 0.0
    total_hands:            int                        = 0
    total_duration_s:       float                      = 0.0

    def compute_aggregates(self) -> None:
        rangebot_matches = [m for m in self.matches
                            if m.opponent != "EHSBot_v3_baseline"]
        if rangebot_matches:
            self.total_hands     = sum(m.n_hands     for m in rangebot_matches)
            self.total_duration_s = sum(m.duration_s  for m in rangebot_matches)
            self.avg_bb100_vs_rangebots = (
                sum(m.bb_100 for m in rangebot_matches) / len(rangebot_matches)
            )
            for arch in ARCHETYPES:
                arch_m = [m for m in rangebot_matches if arch in m.opponent]
                if arch_m:
                    self.avg_bb100_by_archetype[arch] = (
                        sum(m.bb_100 for m in arch_m) / len(arch_m)
                    )
        if self.self_play_result:
            self.exploitability = abs(self.self_play_result.bb_100)

    def print_report(self) -> None:
        self.compute_aggregates()
        W = 76

        print(f"\n{'═' * W}")
        print(f"  RAPPORT DE VALIDATION — Phase 5 Best-Response Engine")
        print(f"{'═' * W}")

        # ── Résultats par opponent ─────────────────────────────────────────────
        rangebot_matches = [m for m in self.matches
                            if m.opponent != "EHSBot_v3_baseline"]
        if rangebot_matches:
            print(f"\n{'─' * W}")
            print(f"  VS RANGEBOTS")
            print(f"{'─' * W}")
            for m in rangebot_matches:
                flag = "✅" if m.bb_100 >= self.config.min_bb100_vs_rangebot else "⚠ "
                print(f"  {flag} {m.summary()}")

        # ── vs EHSBot v3 ──────────────────────────────────────────────────────
        if self.v3_baseline_result:
            print(f"\n{'─' * W}")
            print(f"  VS EHSBOT V3 (baseline)")
            print(f"{'─' * W}")
            m    = self.v3_baseline_result
            flag = "✅" if m.bb_100 > 0 else "⚠ "
            print(f"  {flag} {m.summary()}")

        # ── Self-play ──────────────────────────────────────────────────────────
        if self.self_play_result:
            print(f"\n{'─' * W}")
            print(f"  SELF-PLAY (exploitabilité)")
            print(f"{'─' * W}")
            flag = "✅" if self.exploitability <= self.config.max_exploitability else "⚠ "
            print(f"  {flag} {self.self_play_result.summary()}")
            print(f"       Exploitabilité ≈ {self.exploitability:.2f} BB/100 "
                  f"(seuil ≤ {self.config.max_exploitability:.1f})")

        # ── Agrégats ──────────────────────────────────────────────────────────
        if rangebot_matches:
            print(f"\n{'─' * W}")
            print(f"  MÉTRIQUES AGRÉGÉES")
            print(f"{'─' * W}")
            flag = "✅" if self.avg_bb100_vs_rangebots >= self.config.min_bb100_vs_rangebot else "⚠ "
            print(f"  {flag} BB/100 moyen (tous RangeBots) : "
                  f"{self.avg_bb100_vs_rangebots:+.2f}")
            for arch in ARCHETYPES:
                bb = self.avg_bb100_by_archetype.get(arch)
                if bb is not None:
                    flag = "✅" if bb >= self.config.min_bb100_vs_rangebot else "⚠ "
                    print(f"  {flag} BB/100 vs {arch:20s} : {bb:+.2f}")

            print(f"\n  Mains totales  : {self.total_hands:,}")
            print(f"  Durée totale   : {self.total_duration_s:.1f}s")
            if self.total_duration_s > 0:
                print(f"  Mains/seconde  : "
                      f"{self.total_hands / self.total_duration_s:.0f}")

        # ── Verdict ───────────────────────────────────────────────────────────
        print(f"\n{'═' * W}")
        ok_rb   = (not rangebot_matches
                   or self.avg_bb100_vs_rangebots >= self.config.min_bb100_vs_rangebot)
        ok_exp  = self.exploitability <= self.config.max_exploitability
        all_ok  = ok_rb and ok_exp

        if all_ok:
            print(f"  ✅ VALIDATION PHASE 5 RÉUSSIE")
        else:
            print(f"  ⚠  VALIDATION PARTIELLE")
            if not ok_rb:
                print(f"     BB/100={self.avg_bb100_vs_rangebots:+.2f} < "
                      f"{self.config.min_bb100_vs_rangebot:.1f} (sous-performance)")
            if not ok_exp:
                print(f"     Exploitabilité={self.exploitability:.2f} > "
                      f"{self.config.max_exploitability:.1f} (trop exploitable)")
        print(f"{'═' * W}\n")

    @property
    def success(self) -> bool:
        self.compute_aggregates()
        rangebot_matches = [m for m in self.matches
                            if m.opponent != "EHSBot_v3_baseline"]
        ok_rb  = (not rangebot_matches
                  or self.avg_bb100_vs_rangebots >= self.config.min_bb100_vs_rangebot)
        ok_exp = self.exploitability <= self.config.max_exploitability
        return ok_rb and ok_exp


# =============================================================================
# Runner de match
# =============================================================================

def run_match(
    bot,
    opp,
    bot_name:     str,
    opp_name:     str,
    n_hands:      int,
    config:       ValidationConfig,
) -> MatchResult:
    """
    Joue n_hands mains en simulant un cash game avec rebuy.

    Prend les bots directement (pas les callables) pour pouvoir
    appeler reset() entre les mains et réinitialiser entre les matches.

    Modèle cash game :
      - Les stacks persistent entre les mains (gain/perte cumulatifs)
      - Rebuy automatique à starting_stack quand un joueur descend
        sous REBUY_THRESHOLD × starting_stack (défaut : 70BB = 70%)
      - reset() appelé après chaque main pour incrémenter hand_count
        et préparer le Range Estimator pour la main suivante
    """
    REBUY_THRESHOLD = 0.70  # rebuy si stack < 70% du starting_stack

    # Créer les callables compatibles simulateur
    bot_fn = _make_bot_fn(bot, player_id=0)
    opp_fn = _make_bot_fn(opp, player_id=1)

    table = PokerTable(
        n_players=2,
        starting_stacks=config.starting_stack,
        big_blind=config.big_blind,
        seed=config.seed,
    )

    t0           = time.time()
    hands_played = 0
    profit_chips = 0.0  # accumulé main par main depuis result.gains

    for hand_idx in range(n_hands):
        # Rebuy si stack sous le seuil (cash game : on rebuye jusqu'au max)
        for pid in [0, 1]:
            if table.stacks[pid] < config.starting_stack * REBUY_THRESHOLD:
                table.stacks[pid] = config.starting_stack

        try:
            result = table.run_hand(bots={0: bot_fn, 1: opp_fn})
            # Accumuler le gain net du bot (player_id=0) depuis result.gains
            # result.gains[pid] = stack_fin - stack_debut de cette main
            profit_chips += float(result.gains.get(0, 0))
            hands_played += 1
        except Exception as e:
            logger.warning("Main %d échouée : %s", hand_idx, e)
            continue

        # reset() notifie chaque bot de la fin de main
        if hasattr(bot, 'reset'): bot.reset()
        if hasattr(opp, 'reset'): opp.reset()

    duration = time.time() - t0
    bb_100 = (profit_chips / config.big_blind) / max(hands_played, 1) * 100

    return MatchResult(
        bot_name=bot_name,
        opponent=opp_name,
        n_hands=hands_played,
        profit_chips=profit_chips,
        bb_100=bb_100,
        duration_s=duration,
    )


# =============================================================================
# Fonctions de validation
# =============================================================================

def validate_vs_rangebots(
    bot_v4:           EHSBot,
    config:           ValidationConfig,
    n_hands:          int,
    filter_opponents: Optional[List[str]] = None,
) -> List[MatchResult]:
    """
    Valide le bot contre les 9 RangeBots (ou un sous-ensemble si filter).

    Returns:
        List[MatchResult] — un par RangeBot testé.
    """
    all_bots  = make_all_range_bots(n_sims=500)
    opponents = (
        {k: v for k, v in all_bots.items() if k in filter_opponents}
        if filter_opponents else all_bots
    )

    results = []
    n_total = len(opponents)

    for i, (name, opp) in enumerate(opponents.items(), 1):
        print(f"  [{i:2d}/{n_total}] vs {name:<24s}", end=' ', flush=True)

        result = run_match(
            bot=bot_v4,
            opp=opp,
            bot_name=bot_v4.name,
            opp_name=name,
            n_hands=n_hands,
            config=config,
        )

        flag = "✅" if result.bb_100 >= config.min_bb100_vs_rangebot else "⚠ "
        print(f"BB/100={result.bb_100:+7.2f}  {flag}")
        results.append(result)

        # full_reset entre les matches : remet hand_count à 0
        # et réinitialise le Range Estimator pour le prochain adversaire
        if hasattr(bot_v4, 'full_reset'): bot_v4.full_reset()

    return results


def validate_vs_v3(
    bot_v4:  EHSBot,
    config:  ValidationConfig,
    n_hands: int,
) -> MatchResult:
    """
    Valide le bot v4 contre l'EHSBot v3 (baseline sans Best-Response).

    BB/100 positif = v4 > v3.
    """
    cfg_v3      = EHSBotConfig(use_best_response=False, use_range_estimator=True)
    bot_v3      = EHSBot(config=cfg_v3)
    bot_v3.name = "EHSBot_v3_baseline"

    print(f"  vs {bot_v3.name:<28s}", end=' ', flush=True)

    result = run_match(
        bot=bot_v4,
        opp=bot_v3,
        bot_name=bot_v4.name,
        opp_name=bot_v3.name,
        n_hands=n_hands,
        config=config,
    )

    flag = "✅" if result.bb_100 > 0 else "⚠ "
    print(f"BB/100={result.bb_100:+7.2f}  {flag}")
    if hasattr(bot_v4, 'full_reset'): bot_v4.full_reset()
    return result


def validate_self_play(
    config:  ValidationConfig,
    n_hands: int,
) -> MatchResult:
    """
    Self-play : deux instances EHSBot v4 s'affrontent.

    Mesure l'exploitabilité approximée :
      - Résultat proche de 0 BB/100 = les deux stratégies s'équilibrent
      - Résultat >> 0 = une instance exploite l'autre (bug ou asymétrie)

    Returns:
        MatchResult de l'instance A (player_id=0) vs l'instance B.
    """
    cfg_a   = EHSBotConfig(use_best_response=True, use_range_estimator=True)
    cfg_b   = EHSBotConfig(use_best_response=True, use_range_estimator=True)
    bot_a   = EHSBot(config=cfg_a)
    bot_b   = EHSBot(config=cfg_b)
    bot_a.name = "EHSBot_v4_A"
    bot_b.name = "EHSBot_v4_B"

    print(f"  {bot_a.name} vs {bot_b.name:<20s}", end=' ', flush=True)

    result = run_match(
        bot=bot_a,
        opp=bot_b,
        bot_name=bot_a.name,
        opp_name=bot_b.name,
        n_hands=n_hands,
        config=config,
    )

    exploit = abs(result.bb_100)
    flag    = "✅" if exploit <= config.max_exploitability else "⚠ "
    print(f"BB/100={result.bb_100:+7.2f}  exploitabilité≈{exploit:.2f}  {flag}")
    return result


# =============================================================================
# Point d'entrée
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation Phase 5 — Best-Response Engine"
    )
    parser.add_argument(
        '--quick', action='store_true',
        help=f"Mode rapide ({ValidationConfig().n_hands_quick} mains/match)",
    )
    parser.add_argument(
        '--opponent', type=str, default=None,
        help="Tester contre un seul opponent (ex: TAG_0, LAG_2, CALLING_STATION_1)",
    )
    parser.add_argument(
        '--n_hands', type=int, default=None,
        help="Nombre de mains par match (override --quick)",
    )
    parser.add_argument(
        '--self_play_only', action='store_true',
        help="Uniquement le self-play",
    )
    parser.add_argument(
        '--no_self_play', action='store_true',
        help="Désactiver le self-play",
    )
    parser.add_argument(
        '--vs_v3', action='store_true',
        help="Ajouter un match v4 vs v3",
    )
    args = parser.parse_args()

    config = ValidationConfig()

    # Nombre de mains
    if args.n_hands:
        n_hands     = args.n_hands
        n_self_play = args.n_hands
    elif args.quick:
        n_hands     = config.n_hands_quick
        n_self_play = config.n_hands_quick
    else:
        n_hands     = config.n_hands_vs_rangebot
        n_self_play = config.n_hands_self_play

    # Instancier le bot v4
    cfg_v4  = EHSBotConfig(use_best_response=True, use_range_estimator=True)
    bot_v4  = EHSBot(config=cfg_v4)

    print(f"\n{'═' * 76}")
    print(f"  VALIDATION PHASE 5 — Best-Response Engine")
    print(f"  Bot    : {bot_v4.name}")
    print(f"  Mode   : {'rapide' if args.quick else 'complet'} "
          f"({n_hands} mains/match)")
    print(f"  Seuils : BB/100 ≥ {config.min_bb100_vs_rangebot:.1f} vs RangeBots | "
          f"exploitabilité ≤ {config.max_exploitability:.1f}")
    print(f"{'═' * 76}\n")

    report = ValidationReport(config=config)

    # ── Validation vs RangeBots ────────────────────────────────────────────────
    if not args.self_play_only:
        filter_opp = [args.opponent] if args.opponent else None
        print("── VS RANGEBOTS ──")
        report.matches = validate_vs_rangebots(
            bot_v4=bot_v4,
            config=config,
            n_hands=n_hands,
            filter_opponents=filter_opp,
        )

        # ── vs EHSBot v3 ──────────────────────────────────────────────────────
        if args.vs_v3:
            print("\n── VS EHSBOT V3 ──")
            report.v3_baseline_result = validate_vs_v3(bot_v4, config, n_hands)

    # ── Self-play ──────────────────────────────────────────────────────────────
    if not args.no_self_play:
        print("\n── SELF-PLAY ──")
        report.self_play_result = validate_self_play(config, n_self_play)

    # ── Rapport final ──────────────────────────────────────────────────────────
    report.print_report()
    sys.exit(0 if report.success else 1)


if __name__ == '__main__':
    main()
