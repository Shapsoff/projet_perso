"""
metrics_range.py — Framework de métriques Phase 3
Bot Poker Académique

Fournit les métriques nécessaires pour valider la précision du Range Estimator
(phase 4) contre les RangeBots (vérité terrain connue).

Métriques définies (section 4 du document session 3) :
  1. Précision d'archétype  : % sessions où l'archétype est correctement identifié
  2. Erreur de range (L1)   : sum |P_estimée - P_réelle| sur tous les combos
  3. Convergence            : nombre de mains pour atteindre L1 < 0.10
  4. EV réalisée vs calculée: écart entre EV estimée et EV réelle (BB/100)
  5. BB/100 vs RangeBots    : performance de l'EHSBot contre les 9 RangeBots

Usage:
    from core.metrics_range import RangeMetrics, RangeEstimatorEvaluator

    # Évaluation d'une estimation de range
    metrics = RangeMetrics()
    l1 = metrics.l1_error(estimated_probs, true_combos)
    correct = metrics.archetype_correct('TAG', 'TAG')

    # Évaluation complète d'un estimateur
    evaluator = RangeEstimatorEvaluator()
    report = evaluator.evaluate(estimator, range_bot, n_hands=100)
    evaluator.print_report(report)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import time
import math

from core.bots.range_definitions import (
    get_range,
    get_all_configs,
    combo_in_range,
    ARCHETYPES,
    MUTATIONS,
    ArchetypeConfig,
    ALL_CARDS,
)
from core.bots.range_bot import RangeBot, make_range_bot, make_all_range_bots


# =============================================================================
# Structures de résultats
# =============================================================================

@dataclass
class HandEstimation:
    """Résultat d'une estimation de range sur une main observée."""
    hand_number:        int
    true_archetype:     str
    true_mutation:      int
    estimated_archetype: Optional[str]   # None si pas encore estimé
    estimated_mutation:  Optional[int]
    l1_error:           float            # erreur L1 sur la distribution
    archetype_correct:  bool
    mutation_correct:   bool


@dataclass
class SessionReport:
    """Rapport complet pour une session d'évaluation contre un RangeBot."""
    archetype:          str
    mutation:           int
    n_hands:            int
    bb_per_100:         float            # performance de l'EHSBot

    # Précision d'archétype
    archetype_accuracy: float            # fraction correctement identifiés
    mutation_accuracy:  float            # fraction mutations correctes

    # Erreur L1
    l1_final:           float            # L1 à la fin de la session
    l1_at_30_hands:     float            # L1 après 30 mains
    l1_at_50_hands:     float            # L1 après 50 mains

    # Convergence
    hands_to_converge:  Optional[int]    # mains pour L1 < 0.10, None si pas atteint

    # Historique L1
    l1_history:         List[Tuple[int, float]] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        return self.hands_to_converge is not None


@dataclass
class GlobalReport:
    """Rapport global sur les 9 RangeBots."""
    session_reports:    Dict[str, SessionReport]   # clé : "ARCHETYPE_mutation"
    elapsed_seconds:    float

    # Agrégats
    mean_archetype_accuracy: float
    mean_l1_final:           float
    mean_bb_per_100:         float
    n_converged:             int          # sur 9 RangeBots

    # Par archétype
    archetype_results:  Dict[str, Dict]


# =============================================================================
# Calcul des métriques
# =============================================================================

class RangeMetrics:
    """
    Calcule les métriques de précision du Range Estimator.

    Toutes les méthodes sont stateless — elles calculent depuis les données
    passées en argument sans stocker d'état.
    """

    # Nombre total de combos possibles (C(52,2) = 1326)
    N_COMBOS = 1326

    @staticmethod
    def l1_error(
        estimated_probs: Dict[str, float],
        true_combos:     set,
    ) -> float:
        """
        Calcule l'erreur L1 entre la distribution estimée et la vraie range.

        L1 = sum_{combo} |P_estimée(combo) - P_réelle(combo)|

        La distribution réelle est uniforme sur les combos dans la range :
          P_réelle(combo) = 1 / |range| si combo ∈ range, 0 sinon.

        Args:
            estimated_probs : {combo: probabilité} — distribution estimée
                              (ne doit pas nécessairement sommer à 1, normalisé en interne)
            true_combos     : set de combos dans la vraie range

        Returns:
            Erreur L1 ∈ [0, 2]. 0 = estimation parfaite, 2 = distribution inverse.
        """
        if not true_combos:
            return 2.0

        # Normaliser les probabilités estimées
        total = sum(estimated_probs.values())
        if total <= 0:
            # Distribution uniforme par défaut
            normalized = {c: 1.0 / RangeMetrics.N_COMBOS
                         for c in estimated_probs}
        else:
            normalized = {c: p / total for c, p in estimated_probs.items()}

        # Probabilité réelle : uniforme sur la vraie range
        p_true = 1.0 / len(true_combos)

        l1 = 0.0
        all_combos = set(normalized.keys()) | true_combos

        for combo in all_combos:
            p_est  = normalized.get(combo, 0.0)
            p_real = p_true if combo in true_combos else 0.0
            l1 += abs(p_est - p_real)

        return l1

    @staticmethod
    def archetype_correct(
        true_archetype:      str,
        estimated_archetype: Optional[str],
    ) -> bool:
        """True si l'archétype estimé correspond à la vérité terrain."""
        return estimated_archetype == true_archetype

    @staticmethod
    def mutation_correct(
        true_mutation:      int,
        estimated_mutation: Optional[int],
    ) -> bool:
        """True si la mutation estimée est exacte."""
        return estimated_mutation == true_mutation

    @staticmethod
    def hands_to_l1_threshold(
        l1_history:  List[Tuple[int, float]],
        threshold:   float = 0.10,
    ) -> Optional[int]:
        """
        Retourne le nombre de mains nécessaires pour que L1 < threshold.
        None si le seuil n'est jamais atteint.

        Args:
            l1_history : liste de (hand_number, l1_value) triée par hand_number
            threshold  : seuil cible (défaut: 0.10)
        """
        for hand_num, l1 in l1_history:
            if l1 < threshold:
                return hand_num
        return None

    @staticmethod
    def l1_at_hand(
        l1_history: List[Tuple[int, float]],
        target:     int,
    ) -> float:
        """
        Retourne la valeur L1 au moment le plus proche de target mains.
        Retourne 2.0 si l'historique est vide ou n'atteint pas target.
        """
        if not l1_history:
            return 2.0

        best_hand, best_l1 = None, 2.0
        for hand_num, l1 in l1_history:
            if hand_num <= target:
                best_hand = hand_num
                best_l1   = l1

        return best_l1


# =============================================================================
# Évaluateur complet
# =============================================================================

class RangeEstimatorEvaluator:
    """
    Évalue un Range Estimator contre les 9 RangeBots.

    Interface attendue pour le Range Estimator (phase 4) :
        estimator.update(action_history, game_state)
            → met à jour la distribution interne

        estimator.get_distribution()
            → Dict[str, float] : {combo: probabilité}

        estimator.get_best_archetype()
            → Tuple[str, int] : (archetype, mutation) les plus probables

        estimator.reset()
            → réinitialise pour une nouvelle main
    """

    def __init__(self):
        self.metrics = RangeMetrics()

    def evaluate_single_session(
        self,
        estimator,
        range_bot:   RangeBot,
        ehs_bot,                     # EHSBot instance
        simulator,                   # PokerTable instance (phase 1)
        n_hands:     int = 200,
        seed:        int = 42,
    ) -> SessionReport:
        """
        Évalue l'estimateur sur une session contre un RangeBot.

        À chaque main :
          1. Le simulateur joue EHSBot vs RangeBot
          2. L'estimateur observe les actions du RangeBot
          3. On mesure L1 entre l'estimation et la vraie range

        Note : cette méthode nécessite le simulateur de phase 1.
        Elle est prévue pour l'intégration complète en phase 4.
        Le squelette est défini ici pour guider l'implémentation.

        TODO (phase 4) : connecter au simulateur et à l'EHSBot
        """
        cfg         = range_bot.config
        l1_history  = []
        arch_correct = []
        mut_correct  = []
        bb_results   = []

        # TODO phase 4 : implémenter la boucle de simulation
        # for hand_idx in range(n_hands):
        #     result = simulator.play_hand(ehs_bot, range_bot, seed=seed+hand_idx)
        #     estimator.update(result.action_history, result.game_states)
        #     dist = estimator.get_distribution()
        #     l1   = self.metrics.l1_error(dist, cfg.preflop_combos)
        #     est_arch, est_mut = estimator.get_best_archetype()
        #     l1_history.append((hand_idx + 1, l1))
        #     arch_correct.append(est_arch == cfg.archetype)
        #     mut_correct.append(est_mut == cfg.mutation)
        #     bb_results.append(result.bb_won)

        # Placeholder — retourne un rapport vide en attendant phase 4
        return SessionReport(
            archetype=cfg.archetype,
            mutation=cfg.mutation,
            n_hands=n_hands,
            bb_per_100=0.0,
            archetype_accuracy=0.0,
            mutation_accuracy=0.0,
            l1_final=2.0,
            l1_at_30_hands=2.0,
            l1_at_50_hands=2.0,
            hands_to_converge=None,
            l1_history=l1_history,
        )

    def build_report_from_observations(
        self,
        estimator,
        true_config:    ArchetypeConfig,
        observations:   List[Dict],    # liste de game_states observés
    ) -> SessionReport:
        """
        Construit un SessionReport depuis des observations existantes.
        Utile pour tester l'estimateur hors simulateur.

        Args:
            estimator    : instance du Range Estimator (phase 4)
            true_config  : vraie config du RangeBot (vérité terrain)
            observations : liste de dicts {'action_history', 'game_state'}

        Returns:
            SessionReport avec l'historique L1 complet.
        """
        l1_history   = []
        arch_correct = []
        mut_correct  = []

        estimator.reset()

        for i, obs in enumerate(observations):
            estimator.update(
                obs.get('action_history', []),
                obs.get('game_state', {}),
            )

            dist             = estimator.get_distribution()
            l1               = self.metrics.l1_error(dist, true_config.preflop_combos)
            est_arch, est_mut = estimator.get_best_archetype()

            l1_history.append((i + 1, l1))
            arch_correct.append(est_arch == true_config.archetype)
            mut_correct.append(est_mut == true_config.mutation)

        n = len(observations)
        arch_acc = sum(arch_correct) / n if n > 0 else 0.0
        mut_acc  = sum(mut_correct)  / n if n > 0 else 0.0
        l1_final = l1_history[-1][1] if l1_history else 2.0

        return SessionReport(
            archetype=true_config.archetype,
            mutation=true_config.mutation,
            n_hands=n,
            bb_per_100=0.0,
            archetype_accuracy=arch_acc,
            mutation_accuracy=mut_acc,
            l1_final=l1_final,
            l1_at_30_hands=self.metrics.l1_at_hand(l1_history, 30),
            l1_at_50_hands=self.metrics.l1_at_hand(l1_history, 50),
            hands_to_converge=self.metrics.hands_to_l1_threshold(l1_history),
            l1_history=l1_history,
        )

    @staticmethod
    def print_report(report: SessionReport) -> None:
        """Affiche un SessionReport formaté."""
        key = f"{report.archetype}_mut{report.mutation}"
        print(f"\n  [{key}] {report.n_hands} mains observées")
        print(f"    Précision archétype : {report.archetype_accuracy:.1%}")
        print(f"    Précision mutation  : {report.mutation_accuracy:.1%}")
        print(f"    L1 final           : {report.l1_final:.4f}")
        print(f"    L1 après 30 mains  : {report.l1_at_30_hands:.4f}")
        print(f"    L1 après 50 mains  : {report.l1_at_50_hands:.4f}")
        if report.converged:
            print(f"    Convergence (L1<0.10) : {report.hands_to_converge} mains ✓")
        else:
            print(f"    Convergence (L1<0.10) : non atteinte")
        print(f"    BB/100             : {report.bb_per_100:+.2f}")

    @staticmethod
    def print_global_report(report: GlobalReport) -> None:
        """Affiche un GlobalReport formaté."""
        print("\n" + "=" * 60)
        print("RAPPORT GLOBAL — Range Estimator vs 9 RangeBots")
        print("=" * 60)
        print(f"\n  Temps total        : {report.elapsed_seconds:.1f}s")
        print(f"  Précision archétype: {report.mean_archetype_accuracy:.1%} (cible: > 80%)")
        print(f"  L1 final moyen     : {report.mean_l1_final:.4f} (cible: < 0.15)")
        print(f"  BB/100 moyen       : {report.mean_bb_per_100:+.2f}")
        print(f"  Convergences (L1<0.10) : {report.n_converged}/9")
        print()

        for archetype in ARCHETYPES:
            arch_data = report.archetype_results.get(archetype, {})
            print(f"  [{archetype}]")
            for mut in MUTATIONS:
                key = f"{archetype}_{mut}"
                sr  = report.session_reports.get(key)
                if sr:
                    conv = f"{sr.hands_to_converge}m" if sr.converged else "—"
                    print(f"    mut{mut} : arch={sr.archetype_accuracy:.0%} "
                          f"L1={sr.l1_final:.3f} conv={conv} "
                          f"BB={sr.bb_per_100:+.1f}")
        print()


# =============================================================================
# Simulation multi-sessions RangeBots (P2 du découpage phase 3)
# =============================================================================

def run_rangebot_simulation(
    ehs_bot_factory:  Callable,    # () → EHSBot instance
    simulator_factory: Callable,   # (seed) → PokerTable instance
    n_hands:          int = 500,
    n_sessions:       int = 10,
    verbose:          bool = True,
) -> GlobalReport:
    """
    Lance des simulations EHSBot vs chacun des 9 RangeBots.

    Mesure :
      - BB/100 de l'EHSBot contre chaque RangeBot
      - (Les métriques L1 et précision archétype seront disponibles
         quand le Range Estimator sera développé en phase 4)

    Args:
        ehs_bot_factory   : callable retournant un EHSBot frais
        simulator_factory : callable (seed) → PokerTable
        n_hands           : mains par session
        n_sessions        : sessions par RangeBot
        verbose           : afficher la progression

    Returns:
        GlobalReport avec les résultats agrégés.

    TODO (phase 4) : connecter le Range Estimator et activer les métriques L1
    """
    import numpy as np

    all_bots   = make_all_range_bots(n_sims=500)
    reports    = {}
    start      = time.time()

    for key, range_bot in all_bots.items():
        cfg    = range_bot.config
        bb_results = []

        if verbose:
            print(f"  {key} ({cfg.description[:50]}...)")

        for session_idx in range(n_sessions):
            seed      = session_idx * 100 + hash(key) % 1000
            ehs_bot   = ehs_bot_factory()
            simulator = simulator_factory(seed)

            # TODO (phase 4) : implémenter la boucle de simulation réelle
            # stats = simulator.run_session(
            #     {0: ehs_bot.get_action, 1: range_bot.get_action},
            #     n_hands=n_hands
            # )
            # bb_results.append(stats['bb_per_100'][0])

            # Placeholder
            bb_results.append(0.0)

        mean_bb = float(np.mean(bb_results)) if bb_results else 0.0

        reports[key] = SessionReport(
            archetype=cfg.archetype,
            mutation=cfg.mutation,
            n_hands=n_hands * n_sessions,
            bb_per_100=mean_bb,
            archetype_accuracy=0.0,
            mutation_accuracy=0.0,
            l1_final=2.0,
            l1_at_30_hands=2.0,
            l1_at_50_hands=2.0,
            hands_to_converge=None,
        )

    # Agréger
    elapsed          = time.time() - start
    all_bb           = [r.bb_per_100 for r in reports.values()]
    all_arch_acc     = [r.archetype_accuracy for r in reports.values()]
    all_l1           = [r.l1_final for r in reports.values()]
    n_converged      = sum(1 for r in reports.values() if r.converged)

    arch_results = {}
    for arch in ARCHETYPES:
        arch_reports = {k: v for k, v in reports.items()
                       if v.archetype == arch}
        arch_results[arch] = {
            'mean_bb':    sum(r.bb_per_100 for r in arch_reports.values()) / len(arch_reports),
            'mean_l1':    sum(r.l1_final   for r in arch_reports.values()) / len(arch_reports),
            'n_converged': sum(1 for r in arch_reports.values() if r.converged),
        }

    return GlobalReport(
        session_reports=reports,
        elapsed_seconds=elapsed,
        mean_archetype_accuracy=sum(all_arch_acc) / len(all_arch_acc) if all_arch_acc else 0,
        mean_l1_final=sum(all_l1) / len(all_l1) if all_l1 else 2.0,
        mean_bb_per_100=sum(all_bb) / len(all_bb) if all_bb else 0.0,
        n_converged=n_converged,
        archetype_results=arch_results,
    )
