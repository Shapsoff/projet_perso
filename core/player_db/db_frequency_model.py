"""
db_frequency_model.py — FrequencyModel alimenté par les fréquences empiriques
Phase 6 — Bot Poker Académique

C'est la pièce qui adresse directement la limite documentée en phase 5
(frequency_model.py, section "Calibration") :

    "Les seuils EHS (seuil_bet, seuil_call) viennent directement de
    range_definitions.ArchetypeConfig — ils sont la vérité terrain des
    RangeBots [...]. Pour des joueurs réels (phase future), ces seuils
    seront remplacés par des valeurs calibrées depuis une Player DB."

DBAwareFrequencyModel hérite de FrequencyModel (phase 5) SANS EN MODIFIER
UNE SEULE LIGNE, pour la même raison que DBAwareRangeEstimator : le
pipeline EVCalculator → SizingOptimizer → BluffLayer construit sur
FrequencyModel.compute() est déjà validé (15/15 tests P1, cf. rapport
phase 5) et son contrat d'appel (signature de compute()) est documenté
comme interface publique stable — on ne le touche pas.

Principe :
    Pour un adversaire au Palier 2 (30+ mains DB, cf. profile_builder.py),
    on dispose de fréquences fold/call/raise RÉELLEMENT OBSERVÉES par
    bucket d'EHS (déciles) — pas déduites d'un seuil déterministe de
    RangeBot. classify_combo_response() (phase 5) répond toujours
    "fold OU call OU raise" (déterministe) ; ici, chaque combo peut
    répartir sa masse entre les trois actions selon les fréquences
    observées, ce qui est le comportement attendu d'un joueur réel — cf.
    point 3 de la stratégie validée en session 6 ("pour une même main il
    ne ferait pas toujours la même chose").

Repli sur le modèle archétype (comportement phase 5 inchangé) quand :
    - le profil est au Palier 0 ou 1 (pas encore assez de données),
    - OU le bucket (street, décile EHS, facing_bet) précis n'a pas assez
      d'observations (cf. MIN_OBS_PER_BUCKET dans profile_builder.py) —
      repli PAR BUCKET, pas un tout-ou-rien global : un adversaire peut
      avoir un profil fiable en bet 33% flop mais aucune donnée en raise
      turn, auquel cas seul ce dernier retombe sur l'archétype.

Compatible "drop-in" avec FrequencyModel : même signature de compute(),
utilisable partout où BestResponseEngine attend un FrequencyModel (cf.
l'intégration additive dans best_response_engine.py).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from core.best_response.frequency_model import (
    FrequencyModel,
    FoldCallRaiseResult,
    _MIN_COMBO_PROB,
)

from .player_db import PlayerDB, ehs_to_bucket
from .profile_builder import PlayerProfile

logger = logging.getLogger(__name__)


class DBAwareFrequencyModel(FrequencyModel):
    """
    FrequencyModel qui consulte les fréquences empiriques d'un
    PlayerProfile (Palier 2) plutôt que les seuils déterministes d'un
    archétype RangeBot, quand assez de données existent.

    Le profil actif est injecté via set_active_profile() avant chaque
    appel à compute() — voir l'intégration dans best_response_engine.py
    (DBAwareRangeEstimator.last_profile est réutilisé directement, pas de
    requête DB supplémentaire).
    """

    def __init__(
        self,
        ehs_calculator=None,
        n_sims: int = 500,
        player_db: Optional[PlayerDB] = None,
    ):
        super().__init__(ehs_calculator=ehs_calculator, n_sims=n_sims)
        self._player_db = player_db
        self._active_profile: Optional[PlayerProfile] = None
        # Compteurs de diagnostic (utiles en tests / logs de validation)
        self.n_empirical_combos = 0
        self.n_fallback_combos  = 0

    def set_active_profile(self, profile: Optional[PlayerProfile]) -> None:
        """À appeler avant compute() pour indiquer le profil de l'adversaire courant."""
        self._active_profile = profile

    # =========================================================================
    # Override du point d'entrée principal
    # =========================================================================

    def compute(
        self,
        distribution: Dict[str, float],
        archetype:    str,
        board:        List[str],
        bet_sizing:   float,
        pot:          float,
        n_opponents:  int = 1,
        street:       str = 'flop',
    ) -> FoldCallRaiseResult:
        profile = self._active_profile

        # Pas de profil Palier 2 exploitable → comportement phase 5 exact,
        # aucune régression (Palier 0/1, ou DBAwareFrequencyModel utilisé
        # sans Player DB).
        if profile is None or profile.tier < 2 or not profile.empirical_action_freq:
            return super().compute(
                distribution=distribution, archetype=archetype, board=board,
                bet_sizing=bet_sizing, pot=pot, n_opponents=n_opponents,
                street=street,
            )

        return self._compute_empirical(
            distribution=distribution, archetype=archetype, board=board,
            bet_sizing=bet_sizing, pot=pot, n_opponents=n_opponents,
            street=street, profile=profile,
        )

    # =========================================================================
    # Calcul empirique (Palier 2)
    # =========================================================================

    def _compute_empirical(
        self,
        distribution: Dict[str, float],
        archetype:    str,
        board:        List[str],
        bet_sizing:   float,
        pot:          float,
        n_opponents:  int,
        street:       str,
        profile:      PlayerProfile,
    ) -> FoldCallRaiseResult:
        board_key = "_".join(sorted(board))
        if board_key != self._board_key:
            self._ehs_cache.clear()
            self._board_key = board_key

        facing_bet = bet_sizing > 0.0
        board_set  = set(board)

        # Réplique EXACTEMENT la logique in_range de FrequencyModel.compute()
        # (range figée de l'archétype, PAS la masse de la distribution
        # courante) — nécessaire pour que classify_combo_response() reçoive
        # un signal in_range cohérent avec le comportement phase 5 dans la
        # branche de repli. Calculé une fois pour tout l'appel, comme dans
        # la classe de base.
        preflop_range = self._get_preflop_range(archetype)

        fold_weights:  Dict[str, float] = {}
        call_weights:  Dict[str, float] = {}
        raise_weights: Dict[str, float] = {}
        total_weight = 0.0
        n_ehs_computed = 0
        self.n_empirical_combos = 0
        self.n_fallback_combos  = 0

        for combo, prob in distribution.items():
            if prob < _MIN_COMBO_PROB:
                continue
            c1, c2 = combo[:2], combo[2:]
            if c1 in board_set or c2 in board_set:
                continue

            total_weight += prob

            cache_key = (combo, board_key)
            if cache_key in self._ehs_cache:
                ehs = self._ehs_cache[cache_key]
            else:
                ehs = self._compute_ehs(c1, c2, board, n_opponents)
                self._ehs_cache[cache_key] = ehs
                n_ehs_computed += 1

            bucket = ehs_to_bucket(ehs)
            empirical = profile.get_empirical_bucket(street, bucket, facing_bet)

            if empirical is not None:
                # Donnée réelle disponible pour ce bucket précis : on
                # répartit la masse du combo sur les trois actions selon
                # les fréquences OBSERVÉES pour ce joueur — plus de
                # frontière déterministe fold/call/raise sur un seuil EHS.
                self.n_empirical_combos += 1
                fold_weights[combo]  = prob * empirical['fold']
                call_weights[combo]  = prob * empirical['call']
                raise_weights[combo] = prob * empirical['raise']
            else:
                # Bucket insuffisamment observé pour ce joueur → repli
                # ponctuel sur la logique archétype de phase 5 (pas de
                # régression, juste absence de donnée à cet endroit précis).
                self.n_fallback_combos += 1
                from core.best_response.frequency_model import classify_combo_response
                in_range   = (combo in preflop_range or (c2 + c1) in preflop_range)
                aggression = self._get_aggression(archetype)
                to_call    = bet_sizing * pot
                response = classify_combo_response(
                    ehs=ehs, archetype=archetype, bet_sizing=bet_sizing,
                    pot=pot, to_call=to_call, in_range=in_range,
                    aggression=aggression,
                )
                if response == 'fold':
                    fold_weights[combo] = prob
                elif response == 'call':
                    call_weights[combo] = prob
                else:
                    raise_weights[combo] = prob

        if total_weight <= 0:
            logger.warning(
                "DBAwareFrequencyModel : distribution vide (board=%s, player=%s)",
                board, profile.player_id,
            )
            return FoldCallRaiseResult(
                fold_range={}, call_range={}, raise_range={},
                p_fold=1.0, p_call=0.0, p_raise=0.0,
                archetype=archetype, bet_sizing=bet_sizing,
            )

        p_fold  = sum(fold_weights.values())  / total_weight
        p_call  = sum(call_weights.values())  / total_weight
        p_raise = sum(raise_weights.values()) / total_weight
        total_p = p_fold + p_call + p_raise
        if total_p > 0:
            p_fold  /= total_p
            p_call  /= total_p
            p_raise /= total_p

        logger.debug(
            "DBAwareFrequencyModel[%s]: sizing=%.0f%% empirique=%d/%d "
            "(repli archétype=%d) | fold=%.1f%% call=%.1f%% raise=%.1f%%",
            profile.player_id, bet_sizing * 100,
            self.n_empirical_combos,
            self.n_empirical_combos + self.n_fallback_combos,
            self.n_fallback_combos, p_fold * 100, p_call * 100, p_raise * 100,
        )

        return FoldCallRaiseResult(
            fold_range=fold_weights,
            call_range=call_weights,
            raise_range=raise_weights,
            p_fold=p_fold,
            p_call=p_call,
            p_raise=p_raise,
            archetype=archetype,
            bet_sizing=bet_sizing,
            n_combos_ehs=n_ehs_computed,
        )

    def __repr__(self) -> str:
        player = self._active_profile.player_id if self._active_profile else None
        return (
            f"DBAwareFrequencyModel(active_player={player}, "
            f"n_sims={self._n_sims}, cache_size={len(self._ehs_cache)})"
        )
