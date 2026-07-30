"""
db_range_estimator.py — RangeEstimator alimenté par la Player DB
Phase 6 — Bot Poker Académique

DBAwareRangeEstimator hérite de RangeEstimator (phase 4) SANS EN MODIFIER
UNE SEULE LIGNE. C'est un choix de conception délibéré (décision session 6) :
range_estimator.py est validé par 27+ tests existants (phases 4 et 5) — le
risque de régression en le modifiant directement dépasse le bénéfice. On
extrait donc les deux points d'extension déjà privés-mais-overridables de
RangeEstimator :

    _build_current_prior()     — construit le prior preflop (appelé par reset())
    _update_archetype_estimate() — met à jour l'archétype estimé (appelé par new_hand())

et on les redirige vers le profil construit depuis la Player DB
(profile_builder.build_profile) au lieu de la logique "mix des 3
archétypes / 1 archétype / mix des mutations" de phase 4, qui ne
connaissait que les 9 templates RangeBot (limite documentée phase 5,
section 6.1/9.1).

Interface publique inchangée :
    Toutes les méthodes de RangeEstimator (get_distribution,
    get_archetype_probabilities, observe_action, update_from_game_state,
    reset, new_hand, full_reset...) restent utilisables telles quelles.
    DBAwareRangeEstimator est un remplacement direct ("drop-in") partout
    où RangeEstimator est attendu — y compris dans BestResponseEngine.decide(),
    qui consomme `estimator` via duck typing sans connaître sa classe
    concrète.

Ce que ça change concrètement pour le Best-Response Engine (section 9.1
du rapport phase 5) :
    - Palier 0 (< 15 mains DB) : prior de population par position au lieu
      du mix uniforme des 3 archétypes RangeBot.
    - Palier 1 (15-29 mains DB) : classification archétype comme avant,
      mais calculée sur les compteurs CUMULÉS multi-sessions plutôt que
      sur les seules stats du match en cours.
    - Palier 2 (30+ mains DB) : le prior se mélange progressivement avec
      la distribution réelle des combos vus au showdown pour CE joueur
      précis — ce n'est plus un des 9 templates, même approximativement.
    - get_archetype_probabilities() : la confiance retournée dérive du
      volume de données réellement accumulé sur ce joueur (calibrage
      profile_builder._archetype_confidence), pas d'un softmax sur 3
      classes qui peut atteindre un score élevé par pur artefact relatif
      même quand aucun des 3 templates ne correspond (limite exacte
      identifiée section 6.1 du rapport phase 5).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from core.range_estimator import RangeEstimator
from core.bots.range_definitions import ARCHETYPES

from .player_db import PlayerDB
from .profile_builder import build_profile, get_preflop_prior, PlayerProfile

logger = logging.getLogger(__name__)

# Confiance maximale accordée à un adversaire encore au Palier 0 (aucune
# donnée fiable) — empêche le Best-Response de s'activer sur un pur
# artefact du softmax bayésien hérité de la classe de base, cf. section
# 6.1 du rapport phase 5.
_TIER0_CONFIDENCE_CAP = 0.40


class DBAwareRangeEstimator(RangeEstimator):
    """
    RangeEstimator dont le prior et l'archétype estimé proviennent de la
    Player DB (persistance multi-session) au lieu d'être recalculés depuis
    zéro à chaque nouveau match contre un adversaire.

    Usage :
        db  = PlayerDB("data/player_db.sqlite3")
        est = DBAwareRangeEstimator(player_db=db, player_id="villain_42")
        est.set_villain_position("BTN")   # optionnel, affine le Palier 0
        # ... utilisation identique à RangeEstimator ...
    """

    def __init__(
        self,
        player_db:  PlayerDB,
        player_id:  str,
        smoothing:  float = 0.05,
        min_hands_for_archetype: int = 5,
    ):
        self._player_db  = player_db
        self._player_id  = player_id
        self._villain_position: Optional[str] = None
        self._last_profile: Optional[PlayerProfile] = None

        # __init__ de RangeEstimator appelle self.reset() en fin de
        # constructeur, qui appelle self._build_current_prior() — déjà
        # overridé ci-dessous. Les attributs _player_db/_player_id doivent
        # donc être posés AVANT super().__init__().
        super().__init__(smoothing=smoothing,
                          min_hands_for_archetype=min_hands_for_archetype)

        # RangeEstimator.__init__ laisse _archetype/_mutation à leurs
        # défauts codés en dur ('LAG', 1) — reset() ne les touche pas, seul
        # new_hand() le fait (cf. _update_archetype_estimate). Sans cette
        # synchronisation immédiate, la PREMIÈRE main d'un estimateur
        # fraîchement construit utiliserait 'LAG' pour les mises à jour
        # bayésiennes in-main (_update_preflop/_update_postflop, hérités
        # tels quels), même face à un adversaire déjà bien connu en DB.
        # Le prior initial (get_distribution) était déjà correct — ceci
        # aligne aussi l'archétype utilisé pour le raisonnement in-main.
        self._update_archetype_estimate()

    def full_reset(self) -> None:
        """
        Comme RangeEstimator.full_reset(), mais resynchronise immédiatement
        _archetype/_mutation depuis la Player DB juste après : la classe de
        base les réinitialise à 'LAG'/1 en dur avant de reconstruire le
        prior — sans ce correctif, la première main de CHAQUE nouveau match
        (pas seulement à la construction) retomberait sur 'LAG' jusqu'au
        premier new_hand(), même pour un adversaire déjà au Palier 2.
        """
        super().full_reset()
        self._update_archetype_estimate()

    # =========================================================================
    # Identité de l'adversaire courant
    # =========================================================================

    def set_villain_position(self, position: Optional[str]) -> None:
        """Position adverse courante — affine le prior de population (Palier 0)."""
        self._villain_position = position

    def set_player(self, player_id: str) -> None:
        """
        Change l'adversaire suivi par cet estimateur (nouveau match).
        À appeler avant full_reset() lors d'un changement de table.
        """
        self._player_id = player_id
        self._last_profile = None
        # /!\ Ne PAS conserver la position du joueur précédent : sinon le
        # premier prior Palier 0 du nouvel adversaire hériterait à tort
        # de la position du dernier occupant de ce slot (bug de "fuite"
        # entre adversaires successifs, cf. relecture phase 6).
        self._villain_position = None

    @property
    def player_id(self) -> str:
        return self._player_id

    @property
    def last_profile(self) -> Optional[PlayerProfile]:
        """
        Dernier PlayerProfile calculé (mis à jour à chaque reset()/new_hand()).
        Exposé publiquement pour que BestResponseEngine / DBAwareFrequencyModel
        puissent réutiliser le même profil sans reconstruire une requête DB.
        """
        return self._last_profile

    # =========================================================================
    # Points d'extension de RangeEstimator
    # =========================================================================

    def _build_current_prior(self) -> Dict[str, float]:
        profile = build_profile(self._player_db, self._player_id,
                                 position=self._villain_position)
        self._last_profile = profile
        prior = get_preflop_prior(profile, self._player_db, smoothing=self.smoothing)
        logger.debug(
            "DBAwareRangeEstimator[%s] prior reconstruit — %s",
            self._player_id, profile.summary(),
        )
        return prior

    def _update_archetype_estimate(self) -> None:
        profile = build_profile(self._player_db, self._player_id,
                                 position=self._villain_position)
        self._last_profile = profile

        if profile.tier >= 1 and profile.archetype is not None:
            if profile.archetype != self._archetype:
                logger.debug(
                    "DBAwareRangeEstimator[%s] archétype : %s → %s "
                    "(DB: %d mains, vpip=%.2f pfr=%.2f af=%.1f)",
                    self._player_id, self._archetype, profile.archetype,
                    profile.hands_seen, profile.vpip, profile.pfr, profile.af,
                )
            self._archetype = profile.archetype
            self._mutation  = profile.mutation if profile.mutation is not None else 1
        # Palier 0 : on conserve l'archétype/mutation courants de la classe
        # de base (LAG/1 par défaut) — cohérent avec le prior de population
        # utilisé pour la distribution, qui ne dépend d'aucun archétype.

    def get_archetype_probabilities(self) -> Dict[str, float]:
        """
        Remplace le softmax bayésien de la classe de base par une confiance
        directement dérivée du volume de données DB pour ce joueur (cf.
        profile_builder._archetype_confidence) — voir section 6.1 du
        rapport phase 5 sur les limites du softmax à 3 classes.
        """
        profile = self._last_profile or build_profile(
            self._player_db, self._player_id, position=self._villain_position
        )

        if profile.tier == 0 or profile.archetype is None:
            # Pas assez de données DB pour désigner un archétype — on
            # retombe sur le comportement de la classe de base mais
            # plafonné, pour ne jamais franchir un seuil de confiance par
            # pur artefact du softmax sur seulement 3 classes.
            base = super().get_archetype_probabilities()
            return {k: min(v, _TIER0_CONFIDENCE_CAP) for k, v in base.items()}

        conf = profile.archetype_confidence
        remainder = (1.0 - conf) / max(len(ARCHETYPES) - 1, 1)
        probs = {a: remainder for a in ARCHETYPES}
        probs[profile.archetype] = conf
        return probs

    def __repr__(self) -> str:
        tier = self._last_profile.tier if self._last_profile else "?"
        return (
            f"DBAwareRangeEstimator(player_id={self._player_id}, "
            f"tier={tier}, archetype={self._archetype}, "
            f"mutation={self._mutation}, updates={self._n_updates})"
        )
