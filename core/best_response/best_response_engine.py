"""
best_response_engine.py — Best-Response Engine
Phase 5 — Bot Poker Académique

Orchestre le calcul d'EV et sélectionne la meilleure action exploitative
contre la range estimée de l'adversaire. Remplace _make_decision() dans
l'EHSBot v3 tout en restant rétro-compatible.

Architecture :
    RangeEstimator (Dim 5)
        → distribution : Dict[combo, proba]
        → archetype estimé
              ↓
    FrequencyModel
        → P(fold|sizing), P(call|sizing), P(raise|sizing)
        → fold_range, call_range, raise_range
              ↓
    EVCalculator + MonteCarloEngine
        → EV(fold), EV(check), EV(call), EV(bet_33%), ..., EV(allin)
              ↓
    BestResponseEngine
        → meilleure action (max EV)
        → couche bluff minimale (pot odds)
        → Action retournée à l'EHSBot

Couche bluff minimale (P3 anticipé) :
    Si le bot ne bet que ses value hands, il devient lisible — l'adversaire
    peut folder face à toutes ses bets. On ajoute une fréquence de bluff
    minimale calibrée par pot odds :

        fréquence_bluff_min = pot_odds_offerts_à_l_adversaire
        ex : bet 75% pot → adversaire a 1.75:1 → doit gagner 36%
             → on bluff au minimum 36% de notre range de bet

    Ce n'est pas du CFR — c'est de l'arithmétique de base qui suffit à
    rendre le bot non-exploitable sur la fréquence de bluff.

    En pratique : si l'EV du meilleur bet est positif mais que notre
    range de value est trop petite, on ajoute des bluffs depuis les mains
    avec fort PPot (draws) qui ont une EV de bluff positive.

Fallback :
    Si le Range Estimator n'est pas disponible ou n'a pas assez de données
    (< min_hands_for_best_response), le moteur délègue à l'EHSBot v3
    classique (_make_decision_v3_fallback). Le comportement est identique
    à l'EHSBot v3 actuel — zéro régression.

Interface publique :
    engine = BestResponseEngine(mc_engine, config)
    action = engine.decide(
        game_state,       # dict format simulateur
        estimator,        # RangeEstimator (Dim 5)
        ehs_result,       # dict depuis EHSCalculator (Dim 1)
        spr_info,         # SPRInfo (Dim 4)
        texture_modifier, # float (Dim 3)
        bucket_modifier,  # float (Dim 2)
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.best_response.frequency_model import FrequencyModel
from core.best_response.ev_calculator import EVCalculator, EVResult, DEFAULT_SIZINGS
from core.best_response.sizing_optimizer import SizingOptimizer, OptimalSizingResult
from core.best_response.bluff_layer import BluffLayer, BluffLayerConfig, BluffEvalResult

logger = logging.getLogger(__name__)

# Nombre de mains minimum avant d'activer le Best-Response
# (en dessous, pas assez de données pour le Range Estimator)
_MIN_HANDS_DEFAULT = 30   # cohérent avec PlayerStats.is_reliable (30 mains)

# Seuil de confiance d'archétype minimum pour activer le Best-Response
# Historique : expérimentalement durci de 0.55 à 0.75 (14/07, phase 5) pour
# tenter de limiter les activations sur un mauvais modèle d'adversaire —
# get_archetype_probabilities() est un softmax sur seulement 3 archétypes
# (TAG/LAG/CALLING_STATION), un score élevé peut être atteint même quand
# aucun des 3 templates ne correspond vraiment (cf. self-play : l'EHSBot
# n'est ni TAG, ni LAG, ni CALLING_STATION, mais le softmax doit quand même
# désigner "le moins pire des 3"). CONSTAT (rapport phase 5, section 6.1) :
# aucun impact mesuré, la confiance plafonnait à ~0.35 en self-play, bien
# sous les DEUX valeurs — le seuil n'était pas le problème, le mécanisme de
# confiance lui-même l'était. Remis à 0.55 (session 6) : la Player DB
# (core/player_db/db_range_estimator.py) remplace ce softmax par une
# confiance calibrée sur le volume réel de mains observées par adversaire
# (0.45 à 15 mains → 0.95 à 150 mains, plafonnée à 0.40 tant que
# l'adversaire n'est pas identifié) — la protection que 0.75 cherchait à
# apporter existe désormais nativement, localisée là où le risque est réel.
_MIN_ARCHETYPE_CONFIDENCE = 0.55

# Fréquence de bluff minimale en dessous de laquelle on ignore la couche bluff
_MIN_BLUFF_FREQ = 0.05

# PPot minimum pour qu'une main soit considérée comme bluff candidat
_MIN_PPOT_BLUFF = 0.20


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BestResponseConfig:
    """
    Paramètres du Best-Response Engine.

    Tous les champs ont des valeurs par défaut raisonnables.
    """
    # Sizings évalués (fraction du pot)
    sizings: List[float] = field(
        default_factory=lambda: [0.33, 0.50, 0.75, 1.00, 2.00]
    )

    # Simulations Monte Carlo pour le calcul d'EV
    n_mc_sims_ev: int = 5000

    # Simulations pour le FrequencyModel (EHS des combos adverses)
    n_sims_frequency: int = 500

    # Mains minimum avant d'activer le Best-Response
    min_hands_for_best_response: int = _MIN_HANDS_DEFAULT

    # Confiance archétype minimum
    min_archetype_confidence: float = _MIN_ARCHETYPE_CONFIDENCE

    # Activer la couche bluff minimale (P3)
    use_bluff_layer: bool = True

    # Configuration fine de la couche bluff (None = defaults raisonnables)
    bluff_layer_config: object = None  # BluffLayerConfig optionnel

    # SPR en dessous duquel on commit (identique à l'EHSBot v3)
    spr_commit_threshold: float = 2.0

    # EV minimum en chips pour préférer une action vs fold/check
    # (évite les bets marginaux à EV légèrement positive)
    min_ev_threshold: float = 0.0

    # Activer l'interpolation quadratique pour sizing optimal continu (P2)
    use_sizing_optimizer: bool = True

    # ── Constantes fallback v3 ────────────────────────────────────────────────
    # Dupliquées ici pour éviter l'import circulaire avec ehs_bot.py.
    # Doivent rester synchronisées avec EHSBotConfig si ces valeurs changent.
    fallback_ehs_high:      float = 0.70
    fallback_ehs_low:       float = 0.45
    fallback_sizing_scale:  float = 1.4
    fallback_sizing_max:    float = 1.00
    fallback_sizing_prot:   float = 0.33


# =============================================================================
# Résultat de décision Best-Response
# =============================================================================

@dataclass
class BestResponseDecision:
    """
    Résultat complet d'une décision Best-Response.

    Contient l'action choisie + toutes les EVs calculées pour debug/logging.
    """
    best_action:    EVResult
    all_results:    List[EVResult]
    archetype:      str
    confidence:     float
    used_fallback:  bool       = False
    bluff_added:    bool       = False
    reason:         str        = ""

    def summary(self) -> str:
        tag = "[FALLBACK]" if self.used_fallback else "[BR]"
        bluff = " +bluff" if self.bluff_added else ""
        return (
            f"{tag} {self.best_action.action} "
            f"({self.best_action.sizing_pct:.0%} pot, "
            f"EV={self.best_action.ev:+.2f}) "
            f"arch={self.archetype}({self.confidence:.0%}){bluff} "
            f"— {self.reason}"
        )


# =============================================================================
# Best-Response Engine
# =============================================================================

class BestResponseEngine:
    """
    Sélectionne la meilleure action exploitative basée sur le calcul d'EV.

    Intégration dans l'EHSBot v4 :
        Le Best-Response Engine remplace _make_decision() dans decide().
        Tous les inputs (EHS, bucket, SPR, texture, Range Estimator) sont
        calculés en amont par le pipeline existant et passés ici.

    Rétro-compatibilité :
        Si le Range Estimator n'a pas assez de données ou si la confiance
        archétype est insuffisante, on délègue à _make_decision_v3_fallback()
        qui réplique exactement le comportement de l'EHSBot v3.
    """

    def __init__(
        self,
        mc_engine,
        config: Optional[BestResponseConfig] = None,
        player_db=None,
    ):
        """
        Args:
            mc_engine : instance de MonteCarloEngine (poker_engine C++)
            config    : BestResponseConfig (optionnel, defaults raisonnables)
            player_db : PlayerDB (Phase 6, optionnel). Si fourni, le
                        FrequencyModel utilisé devient un
                        DBAwareFrequencyModel (core/player_db/db_frequency_model.py)
                        qui exploite les fréquences fold/call/raise
                        RÉELLEMENT OBSERVÉES par adversaire, par bucket
                        d'EHS, dès que celui-ci atteint le Palier 2 (30+
                        mains cumulées en DB) — au lieu des seuils
                        déterministes des 9 archétypes RangeBot (limite
                        documentée phase 5, section 6.1/9.1). Sans
                        player_db (défaut, None) : comportement phase 5
                        strictement inchangé, aucune régression.
        """
        self._config    = config or BestResponseConfig()
        self._player_db = player_db

        # EHS calculator pour le FrequencyModel (partagé avec l'EHSBot)
        self._ehs_calc = None  # sera injecté via set_ehs_calculator()

        if player_db is not None:
            from core.player_db.db_frequency_model import DBAwareFrequencyModel
            self._freq_model = DBAwareFrequencyModel(
                ehs_calculator=None,  # injecté après
                n_sims=self._config.n_sims_frequency,
                player_db=player_db,
            )
        else:
            self._freq_model = FrequencyModel(
                ehs_calculator=None,  # injecté après
                n_sims=self._config.n_sims_frequency,
            )

        self._ev_calc = EVCalculator(
            mc_engine=mc_engine,
            frequency_model=self._freq_model,
            n_mc_sims=self._config.n_mc_sims_ev,
        )

        # Sizing Optimizer (P2) — interpolation quadratique sur la courbe EV
        self._sizing_optimizer = (
            SizingOptimizer(ev_calculator=self._ev_calc)
            if self._config.use_sizing_optimizer
            else None
        )

        # Bluff Layer (P3) — fréquence de bluff minimale calibrée par pot odds
        bl_cfg = (self._config.bluff_layer_config
                  if self._config.bluff_layer_config is not None
                  else BluffLayerConfig())
        self._bluff_layer = (
            BluffLayer(config=bl_cfg)
            if self._config.use_bluff_layer
            else None
        )

        logger.info(
            "BestResponseEngine initialisé (sizings=%s, mc_sims=%d, freq_sims=%d, optimizer=%s)",
            self._config.sizings,
            self._config.n_mc_sims_ev,
            self._config.n_sims_frequency,
            "on" if self._config.use_sizing_optimizer else "off",
        )

    def set_ehs_calculator(self, ehs_calc) -> None:
        """
        Injecte le calculateur EHS partagé avec l'EHSBot.
        Appelé par l'EHSBot lors de l'initialisation.
        """
        self._ehs_calc = ehs_calc
        self._freq_model._ehs_calc = ehs_calc

    # =========================================================================
    # Point d'entrée principal
    # =========================================================================

    def decide(
        self,
        game_state:       dict,
        estimator,
        ehs_result:       dict,
        spr_info,
        texture_modifier: float,
        bucket_modifier:  float,
        hand_count:       int,
    ) -> Tuple[BestResponseDecision, 'Action']:
        """
        Décide la meilleure action en calculant l'EV de chaque option.

        Args:
            game_state       : dict format simulateur
            estimator        : RangeEstimator (Dim 5), peut être None
            ehs_result       : dict {'EHS', 'PPot', 'NPot', 'HS'} (Dim 1)
            spr_info         : SPRInfo (Dim 4)
            texture_modifier : float (Dim 3)
            bucket_modifier  : float (Dim 2)
            hand_count       : nombre de mains jouées vs cet adversaire

        Returns:
            (BestResponseDecision, Action) — décision détaillée + action
            compatible avec le simulateur.
        """
        # Extraire les champs du game_state
        our_hand = game_state.get('hand', [])
        board    = game_state.get('board', [])
        pot      = float(game_state.get('pot', 1))
        to_call  = float(game_state.get('to_call', 0))
        stack    = float(game_state.get('stack', 0))
        street   = game_state.get('street', 'flop')
        players  = game_state.get('players', [])
        n_opp    = max(len(players), 1)

        ehs  = ehs_result.get('EHS', 0.5)
        ppot = ehs_result.get('PPot', 0.0)

        # ── SPR très bas → commitment forcé (identique v3) ────────────────────
        if spr_info.force_commit:
            amount  = min(stack, pot * 3)
            action  = _make_action('allin', amount, amount / pot if pot else 0)
            decision = BestResponseDecision(
                best_action=EVResult('allin', 0, amount, amount),
                all_results=[],
                archetype='unknown',
                confidence=0.0,
                used_fallback=True,
                reason="SPR force commit",
            )
            logger.debug("[BR] SPR force commit → allin %.0f", amount)
            return decision, action

        # ── Phase 6 : propager le profil DB courant au FrequencyModel ─────────
        # Si le FrequencyModel est DB-aware (player_db fourni au constructeur)
        # et que l'estimateur expose un profil (DBAwareRangeEstimator), on le
        # lui transmet pour activer les fréquences empiriques du Palier 2.
        # No-op silencieux dans tous les autres cas (comportement phase 5
        # inchangé — estimateur classique ou FrequencyModel non DB-aware).
        if hasattr(self._freq_model, 'set_active_profile'):
            self._freq_model.set_active_profile(getattr(estimator, 'last_profile', None))

        # ── Vérifier si le Best-Response peut s'activer ───────────────────────
        can_use_br, archetype, confidence = self._check_br_conditions(
            estimator, hand_count
        )

        if not can_use_br:
            # Fallback : comportement EHSBot v3
            action, reason = self._fallback_v3(
                ehs, ppot, ehs_result, bucket_modifier, spr_info,
                texture_modifier, pot, to_call, stack
            )
            decision = BestResponseDecision(
                best_action=EVResult(action.action_type, action.sizing_pct,
                                     action.amount, 0.0),
                all_results=[],
                archetype=archetype,
                confidence=confidence,
                used_fallback=True,
                reason=reason,
            )
            return decision, action

        # ── Best-Response actif ───────────────────────────────────────────────
        distribution = estimator.get_distribution()

        # Calculer l'EV de toutes les actions
        all_results = self._ev_calc.compute_all_actions(
            our_hand=our_hand,
            board=board,
            distribution=distribution,
            archetype=archetype,
            pot=pot,
            to_call=to_call,
            stack=stack,
            street=street,
            sizings=self._config.sizings,
            n_opponents=n_opp,
        )

        if not all_results:
            # Sécurité : fallback si calcul échoue
            action, reason = self._fallback_v3(
                ehs, ppot, ehs_result, bucket_modifier, spr_info,
                texture_modifier, pot, to_call, stack
            )
            decision = BestResponseDecision(
                best_action=EVResult(action.action_type, 0, action.amount, 0),
                all_results=[],
                archetype=archetype,
                confidence=confidence,
                used_fallback=True,
                reason=f"Calcul EV vide — {reason}",
            )
            return decision, action

        # Meilleure action = max EV (discret)
        best = all_results[0]

        # ── Sizing Optimizer P2 : affiner le sizing si bet/raise ──────────────
        # Si la meilleure action est un bet, on cherche le sizing optimal en
        # continu par interpolation quadratique sur la courbe EV(sizing).
        # L'optimizer retourne le même sizing discret si l'amélioration est
        # inférieure à _MIN_EV_IMPROVEMENT (pas de changement marginal).
        if (self._sizing_optimizer is not None
                and to_call == 0  # uniquement quand on a l'initiative
                and best.action in ('bet', 'allin')):
            opt = self._sizing_optimizer.find_optimal_sizing(
                our_hand=our_hand,
                board=board,
                distribution=distribution,
                archetype=archetype,
                pot=pot,
                to_call=to_call,
                stack=stack,
                street=street,
                n_opponents=n_opp,
            )
            if opt.improvement > 0:
                # Créer un EVResult synthétique pour le sizing optimal
                best = EVResult(
                    action='bet',
                    sizing_pct=opt.optimal_sizing,
                    amount=opt.optimal_amount,
                    ev=opt.optimal_ev,
                    ev_fold=opt.discrete_best.ev_fold,
                    ev_call=opt.discrete_best.ev_call,
                    ev_raise=opt.discrete_best.ev_raise,
                    equity_call=opt.discrete_best.equity_call,
                    p_fold=opt.discrete_best.p_fold,
                    p_call=opt.discrete_best.p_call,
                    p_raise=opt.discrete_best.p_raise,
                )
                logger.debug("[BR] SizingOptimizer: %s", opt.summary())

        # ── Couche bluff minimale P3 ─────────────────────────────────────────
        bluff_added = False
        if (self._bluff_layer is not None
                and best.action in ('bet', 'raise', 'allin')
                and best.p_fold > 0):
            bluff_eval = self._bluff_layer.evaluate(
                best_ev=best,
                ehs=ehs,
                ppot=ppot,
                pot=pot,
                street=street,
                archetype=archetype,
            )
            bluff_added = bluff_eval.should_bluff
            logger.debug("[BR] BluffLayer: %s", bluff_eval.summary())

        # Construire l'Action compatible simulateur
        action   = _make_action(best.action, best.amount, best.sizing_pct)
        decision = BestResponseDecision(
            best_action=best,
            all_results=all_results,
            archetype=archetype,
            confidence=confidence,
            bluff_added=bluff_added,
            reason=f"EV={best.ev:+.2f} vs "
                   f"{all_results[1].action if len(all_results) > 1 else '?'}"
                   f"(EV={all_results[1].ev if len(all_results) > 1 else 0.0:+.2f})",
        )

        logger.info("[BR] %s", decision.summary())
        return decision, action

    # =========================================================================
    # Conditions d'activation
    # =========================================================================

    def _check_br_conditions(
        self,
        estimator,
        hand_count: int,
    ) -> Tuple[bool, str, float]:
        """
        Vérifie si le Best-Response peut s'activer.

        Returns:
            (can_use, archetype, confidence)
        """
        if estimator is None:
            return False, 'unknown', 0.0

        if hand_count < self._config.min_hands_for_best_response:
            return False, 'unknown', 0.0

        arch_probs  = estimator.get_archetype_probabilities()
        best_arch   = max(arch_probs, key=arch_probs.__getitem__)
        confidence  = arch_probs[best_arch]

        if confidence < self._config.min_archetype_confidence:
            logger.debug(
                "[BR] Confiance insuffisante : %s=%.2f < %.2f",
                best_arch, confidence, self._config.min_archetype_confidence
            )
            return False, best_arch, confidence

        return True, best_arch, confidence

    # =========================================================================
    # Couche bluff minimale
    # =========================================================================

    def _apply_bluff_layer(
        self,
        best:        EVResult,
        all_results: List[EVResult],
        ppot:        float,
        pot:         float,
        to_call:     float,
    ) -> Tuple[EVResult, bool]:
        """Conservé pour rétro-compatibilité. Délègue à BluffLayer (P3)."""
        # Cette méthode n'est plus appelée directement depuis decide() —
        # le branchement se fait via self._bluff_layer.evaluate().
        # Conservée pour éviter les erreurs si appelée depuis un test existant.
        if self._bluff_layer is None or best.sizing_pct <= 0:
            return best, False
        from core.best_response.bluff_layer import BluffEvalResult
        eval_result = self._bluff_layer.evaluate(
            best_ev=best, ehs=0.5, ppot=ppot,
            pot=pot, street='flop', archetype='TAG',
        )
        return best, eval_result.should_bluff

    # =========================================================================
    # Fallback EHSBot v3
    # =========================================================================

    def _fallback_v3(
        self,
        ehs:              float,
        ppot:             float,
        ehs_result:       dict,
        bucket_modifier:  float,
        spr_info,
        texture_modifier: float,
        pot:              float,
        to_call:          float,
        stack:            float,
    ) -> Tuple['Action', str]:
        """
        Réplique exactement la logique _make_decision() de l'EHSBot v3.
        Utilisé quand le Best-Response ne peut pas s'activer.
        """
        # Constantes v3 depuis BestResponseConfig (sans import ehs_bot)
        high_threshold = self._config.fallback_ehs_high + bucket_modifier
        low_threshold  = self._config.fallback_ehs_low  + bucket_modifier

        # EHS élevé → value bet/raise
        if ehs > high_threshold:
            sizing = 0.33 + (ehs - high_threshold) * self._config.fallback_sizing_scale
            sizing = max(0.33, min(sizing * texture_modifier
                                   * spr_info.sizing_modifier(),
                                   self._config.fallback_sizing_max))
            bet_amount = min(sizing * pot, stack)
            if to_call > 0:
                raise_amount = min(to_call * 2.5 + pot * sizing, stack)
                return _make_action('raise', raise_amount, sizing), \
                       "fallback v3 — EHS élevé"
            return _make_action('bet', bet_amount, sizing), \
                   "fallback v3 — EHS élevé"

        # EHS médian → protection ou call
        if ehs >= low_threshold:
            if to_call == 0:
                sizing     = self._config.fallback_sizing_prot * texture_modifier \
                             * spr_info.sizing_modifier()
                sizing     = min(sizing, self._config.fallback_sizing_max)
                bet_amount = min(sizing * pot, stack)
                return _make_action('bet', bet_amount, sizing), \
                       "fallback v3 — protection"
            if to_call > 0:
                pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 1
                if ehs > pot_odds:
                    return _make_action('call', min(to_call, stack), 0), \
                           "fallback v3 — call rentable"
                return _make_action('fold', 0, 0), "fallback v3 — fold (EHS < pot_odds)"
            return _make_action('check', 0, 0), "fallback v3 — check"

        # EHS faible → check/fold, sauf draw fort
        if ppot >= 0.25 and to_call > 0:
            pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 1
            if (ehs + ppot * 0.5) > pot_odds:
                return _make_action('call', min(to_call, stack), 0), \
                       "fallback v3 — call draw fort"

        if to_call > 0:
            return _make_action('fold', 0, 0), "fallback v3 — fold (EHS faible)"
        return _make_action('check', 0, 0), "fallback v3 — check (EHS faible)"

    def __repr__(self) -> str:
        return (
            f"BestResponseEngine("
            f"sizings={self._config.sizings}, "
            f"mc_sims={self._config.n_mc_sims_ev}, "
            f"min_hands={self._config.min_hands_for_best_response})"
        )


# =============================================================================
# Helper — construction d'une Action compatible simulateur
# =============================================================================

def _make_action(action_type: str, amount: float, sizing_pct: float) -> 'Action':
    """
    Construit une Action compatible avec le simulateur.
    Import local pour éviter la circularité avec ehs_bot.py.
    """
    try:
        from core.bots.ehs_bot import Action
        return Action(action_type=action_type, amount=amount, sizing_pct=sizing_pct)
    except ImportError:
        # Fallback minimal si ehs_bot non disponible
        class _Action:
            def __init__(self, action_type, amount, sizing_pct):
                self.action_type = action_type
                self.amount      = amount
                self.sizing_pct  = sizing_pct
        return _Action(action_type, amount, sizing_pct)
