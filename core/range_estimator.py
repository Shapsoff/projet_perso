"""
range_estimator.py — Range Estimator bayésien
Phase 4 — Bot Poker Académique

Estime la range adverse (distribution de probabilité sur les 1 326 combos)
en combinant un modèle bayésien preflop et des mises à jour postflop par action.

Architecture :
  ┌─────────────────────────────────────────────────────────────┐
  │  P0  — Prior preflop pondéré par le bucket d'action        │
  │         (BUCKET_RANGE_WIDTH + heuristiques VPIP/PFR)       │
  │  P0.5 — Classifieur d'archétype léger (VPIP / PFR / AF)   │
  │  P1  — Mises à jour postflop par action (fold/call/raise)  │
  │  P2  — Interface de métriques (L1, calibration, Brier)     │
  └─────────────────────────────────────────────────────────────┘

Modèle bayésien :

  Preflop :
    Prior(combo) ∝ P(combo ∈ range | archétype) × n_combos(classe)
    Normalisé pour sommer à 1 sur les 1 326 combos.

  Postflop — mise à jour par action observable (fold / call / raise) :
    P(combo | actions) ∝ P(combo | actions précédentes)
                        × L(action | combo, street, pot_odds)

    Les likelihoods sont calculés depuis les seuils EHS des archétypes,
    ce qui les rend cohérents avec le comportement déterministe des RangeBots.

  Prior pondéré par nombre de combos (correction décision session 4) :
    Les 169 classes de mains n'ont pas le même nombre de combos.
    Un prior uniforme sur les 169 classes sur-représente les mains rares
    (paires suited = 6 combos) et sous-représente les mains courantes (AKo = 12).
    Le prior est donc pondéré par n_combos(classe) / total_combos.

Likelihoods :
  L'intuition est simple :
    - fold   → la main était probablement faible (hors range ou EHS bas)
    - call   → la main était dans la range mais pas assez forte pour raise
    - raise  → la main était forte (dans la range, EHS élevé)

  Pour les RangeBots déterministes, ces likelihoods sont proches de 0/1.
  Pour des joueurs réels (futur), on utilisera des valeurs intermédiaires.

Interface publique :
    estimator = RangeEstimator()
    estimator.observe_action(action_type, street, pot, to_call, villain_id)
    estimator.observe_preflop_action(action_type, villain_id, board_cards)
    dist = estimator.get_distribution()
    arch, mut = estimator.get_best_archetype()
    estimator.reset()

Usage dans le simulateur :
    estimator = RangeEstimator(villain_id=1)
    # À chaque action observée dans game_state.action_history :
    estimator.update_from_game_state(game_state, villain_id=1)
    dist = estimator.get_distribution()

Limitation connue et acceptée (section 3.4 du document) :
    Validé contre des bots déterministes. La robustesse contre des joueurs
    réels avec mixed strategies est une problématique de phase ultérieure.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.bots.range_definitions import (
    get_range,
    get_all_configs,
    ArchetypeConfig,
    ARCHETYPES,
    MUTATIONS,
    RANKS,
    SUITS,
    RANK_VAL,
    hand_to_class,
    expand_hand_class,
)
from core.action_history import (
    ActionBucket,
    BUCKET_RANGE_WIDTH,
    classify_history,
    get_villain_bucket_from_state,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constantes
# =============================================================================

# Tous les combos possibles (C(52,2) = 1 326)
_ALL_CARDS  = [r + s for r in RANKS for s in SUITS]

def _build_all_combos() -> List[str]:
    combos = []
    cards  = _ALL_CARDS
    n      = len(cards)
    for i in range(n):
        for j in range(i + 1, n):
            combos.append(cards[i] + cards[j])
    return combos

ALL_COMBOS: List[str] = _build_all_combos()  # 1 326 éléments

# Index inversé combo → position dans ALL_COMBOS (accès O(1))
COMBO_INDEX: Dict[str, int] = {c: i for i, c in enumerate(ALL_COMBOS)}

# Nombre de combos par classe de main (pour la pondération du prior)
# paire = 6, suited = 4, offsuit = 12
def _combo_weight(card1: str, card2: str) -> float:
    """Poids d'un combo dans le prior (inverse du nb de combos de sa classe)."""
    r1, s1 = card1[0], card1[1]
    r2, s2 = card2[0], card2[1]
    if r1 == r2:
        return 1.0 / 6.0
    elif s1 == s2:
        return 1.0 / 4.0
    else:
        return 1.0 / 12.0

# Likelihoods par action (déterministe → proche de 0/1, léger lissage)
# Format : (L_in_range, L_out_of_range)
# Ces valeurs sont volontairement asymétriques pour les bots déterministes.
_EPSILON = 0.02  # lissage minimal pour éviter les 0 stricts (problème numérique)

# Likelihoods preflop
_L_PF_RAISE_IN      = 1.0 - _EPSILON   # raise si dans range
_L_PF_RAISE_OUT     = _EPSILON          # raise si hors range
_L_PF_CALL_IN       = 0.15             # call d'un 3bet si dans range (rare pour TAG)
_L_PF_CALL_OUT      = _EPSILON          # call si hors range → impossible pour RangeBot
_L_PF_FOLD_IN       = _EPSILON          # fold si dans range → impossible pour RangeBot
_L_PF_FOLD_OUT      = 1.0 - _EPSILON   # fold si hors range

# Likelihoods postflop (continue de la section P1)
# Dépendent du profil archétype → voir _postflop_likelihoods()


# =============================================================================
# Classifieur d'archétype léger (P0.5)
# =============================================================================

@dataclass
class VillainStats:
    """
    Statistiques légères d'un adversaire, calculées à la volée.
    VPIP / PFR / AF sont les trois statistiques fondamentales du profiling poker.
    """
    hands_seen:       int   = 0
    hands_vpip:       int   = 0   # a misé/callé volontairement preflop
    hands_pfr:        int   = 0   # a open-raise ou 3bet preflop
    aggressive_acts:  int   = 0   # bet / raise postflop
    passive_acts:     int   = 0   # call / check postflop

    @property
    def vpip(self) -> float:
        """Voluntarily Put In Pot — % de mains jouées."""
        return self.hands_vpip / self.hands_seen if self.hands_seen > 0 else 0.5

    @property
    def pfr(self) -> float:
        """Pre-Flop Raise — % de mains avec raise preflop."""
        return self.hands_pfr / self.hands_seen if self.hands_seen > 0 else 0.3

    @property
    def af(self) -> float:
        """Aggression Factor = (bet + raise) / call."""
        return (self.aggressive_acts / self.passive_acts
                if self.passive_acts > 0 else 3.0)

    @property
    def is_reliable(self) -> bool:
        """Stats fiables à partir de 20 mains (adapté au contexte simulation)."""
        return self.hands_seen >= 20

    def infer_archetype(self) -> str:
        """
        Infère l'archétype à partir des stats VPIP/PFR/AF.

        Règles empiriques calibrées sur les seuils des RangeBots :
          TAG  : VPIP < 0.20, PFR > 0.10, AF élevé
          LAG  : VPIP > 0.28, PFR > 0.20, AF élevé
          CALL : VPIP > 0.28, PFR < 0.15, AF faible

        Ces seuils ont été choisis pour séparer proprement les 9 archétypes
        même avec seulement 20-30 mains observées.

        Deux correctifs phase 6 (session 6, relecture) :

        1. Les 3 mutations TAG réellement configurées dans
           range_definitions.py (range_pct ≈ 3-7%) échouaient
           systématiquement la condition "PFR > 0.10" ci-dessous et
           retombaient sur LAG — un RangeBot ne limpe jamais (il relance ou
           fold), donc PFR == VPIP exactement, et à 3-7% ça reste sous la
           barre des 10%. La branche ajoutée avant la logique historique
           capture ce cas (range très serrée, quasiment aucun limp, AF
           élevé) sans toucher au calibrage existant pour les VPIP plus
           élevés.

        2. La branche TAG historique (v < 0.22 et p > 0.10) ne vérifiait
           JAMAIS l'AF — un adversaire CALLING_STATION (passif par
           construction) dont le VPIP échantillonné tombe sous 22% par
           pur bruit statistique (mesuré : ~47% des cas à 60 mains pour
           CALLING_STATION mutation 1, configuré à 29.4%) se faisait donc
           classer TAG à tort, malgré une agression manifestement basse.
           Ajout de "and a >= 1.5" à cette branche : les deux archétypes
           agressifs (TAG, LAG) restent capturés, CALLING_STATION en est
           exclu quel que soit le bruit d'échantillonnage sur le VPIP.

        Les 3 tests unitaires historiques (test_archetype_classifier) et
        le reste du calibrage existant restent inchangés à l'identique.

        Limite connue, non corrigée ici (délibérément, périmètre plus
        large et plus risqué — TAG et LAG sont tous deux "aggressive" en
        configuration, l'AF ne peut donc pas les distinguer) : LAG
        mutation 0 (VPIP=PFR≈20.7%) tombe toujours dans la branche TAG
        historique. Toute retouche du seuil v<0.22 lui-même doit composer
        avec le cas de test historique (VPIP=0.20 attendu TAG), à
        seulement 0.7 point de VPIP de LAG mutation 0 — à traiter
        séparément si besoin.
        """
        v, p, a = self.vpip, self.pfr, self.af

        if 0.0 < v <= 0.10 and p >= v * 0.90 and a >= 1.5:
            return 'TAG'
        if v < 0.22 and p > 0.10 and a >= 1.5:
            return 'TAG'
        elif v > 0.25 and p > 0.18 and a >= 1.5:
            return 'LAG'
        elif v > 0.25 and (p < 0.18 or a < 1.5):
            return 'CALLING_STATION'
        else:
            # Zone ambiguë : pencher vers LAG si agressif, CALL sinon
            return 'LAG' if a >= 2.0 else 'CALLING_STATION'


# =============================================================================
# Priors par archétype
# =============================================================================

class PriorBuilder:
    """
    Construit les priors preflop bayésiens pour chaque archétype.

    Le prior d'un archétype est une distribution de probabilité sur les
    1 326 combos, pondérée par la taille de chaque classe de main.

    Deux niveaux de prior :
      - Prior d'archétype   : mélange pondéré des 3 mutations de l'archétype
      - Prior de mutation   : distribution exacte de cette mutation
    """

    # Cache des priors (calculé une seule fois au démarrage)
    _cache: Dict[Tuple[str, Optional[int]], Dict[str, float]] = {}

    @classmethod
    def build_prior(
        cls,
        archetype: str,
        mutation:  Optional[int] = None,
        smoothing: float = 0.05,
    ) -> Dict[str, float]:
        """
        Construit le prior pour un archétype (toutes mutations) ou une mutation.

        Le lissage (smoothing) assigne une probabilité non nulle à tous les
        combos, même hors range. Cela évite la dégénérescence du modèle bayésien
        quand on voit une action inattendue (ex : TAG qui limp).

        Args:
            archetype : 'TAG', 'LAG', ou 'CALLING_STATION'
            mutation  : None → prior moyen sur les 3 mutations,
                        0/1/2 → prior exact de cette mutation
            smoothing : fraction de masse assignée aux combos hors range.
                        0.05 = 5% de masse uniforme sur les 1326 combos.

        Returns:
            Dict[combo, probabilité] normalisé (somme = 1).
        """
        key = (archetype, mutation)
        if key in cls._cache:
            return dict(cls._cache[key])

        # Collecter les combos dans la range et leur poids
        combo_weights: Dict[str, float] = {}

        if mutation is not None:
            configs = [get_range(archetype, mutation)]
        else:
            configs = [get_range(archetype, m) for m in MUTATIONS]

        for cfg in configs:
            for combo in cfg.preflop_combos:
                # Pondération par la taille de la classe (correction P0)
                c1, c2 = combo[:2], combo[2:]
                w = _combo_weight(c1, c2)
                combo_weights[combo] = combo_weights.get(combo, 0.0) + w

        # Normaliser la masse in-range
        total_in = sum(combo_weights.values())
        if total_in <= 0:
            # Fallback : uniforme
            result = {c: 1.0 / len(ALL_COMBOS) for c in ALL_COMBOS}
            cls._cache[key] = result
            return dict(result)

        # Appliquer le lissage
        # masse_in = (1 - smoothing), masse_out = smoothing / N_combos
        n_combos        = len(ALL_COMBOS)
        base_out        = smoothing / n_combos
        scale_in        = (1.0 - smoothing) / total_in

        result: Dict[str, float] = {}
        for combo in ALL_COMBOS:
            p_in = combo_weights.get(combo, 0.0) * scale_in
            result[combo] = p_in + base_out

        # Normaliser (somme exactement = 1 malgré les arrondis)
        total = sum(result.values())
        result = {c: p / total for c, p in result.items()}

        cls._cache[key] = result
        logger.debug(
            "Prior construit : archetype=%s mutation=%s — "
            "%d combos in range (poids total=%.3f)",
            archetype, mutation,
            len([c for c in result if combo_weights.get(c, 0) > 0]),
            total_in,
        )
        return dict(result)

    @classmethod
    def uniform_prior(cls) -> Dict[str, float]:
        """Prior uniforme sur les 1 326 combos (aucune info)."""
        p = 1.0 / len(ALL_COMBOS)
        return {c: p for c in ALL_COMBOS}


# =============================================================================
# Likelihoods postflop
# =============================================================================

def _postflop_likelihoods(
    action:       str,         # 'fold', 'call', 'raise', 'check', 'bet', 'allin'
    archetype:    str,         # pour ajuster les seuils EHS
    ehs_estimate: float,       # EHS estimé pour ce combo (0-1)
    pot_odds:     float,       # to_call / (pot + to_call), 0 si pas de mise
    in_range:     bool,        # ce combo est-il dans le prior range ?
) -> float:
    """
    Calcule la vraisemblance P(action | combo, contexte) pour la mise à jour
    bayésienne postflop.

    Les likelihoods reflètent le comportement déterministe des RangeBots :
      - Un RangeBot hors range fold toujours → L(fold | hors range) ≈ 1
      - Un RangeBot in range bet si EHS > seuil → L(bet | EHS > seuil) ≈ 1

    Args:
        action       : action observée du villain
        archetype    : archétype estimé (pour les seuils EHS)
        ehs_estimate : EHS estimé pour ce combo (neutre = 0.5 si inconnu)
        pot_odds     : rentabilité du call (utilisée pour le call)
        in_range     : ce combo est dans la range preflop

    Returns:
        Vraisemblance ∈ [0, 1].
    """
    action = action.lower()

    # Récupérer les seuils EHS de l'archétype
    cfg = get_range(archetype, 0)  # on utilise les seuils base (même pour mutations)
    seuil_bet  = cfg.postflop_ehs_threshold
    seuil_call = cfg.postflop_ehs_call

    # Hors range preflop : un RangeBot fold toujours postflop
    if not in_range:
        if action in ('fold',):
            return 1.0 - _EPSILON
        elif action in ('check',):
            return 0.10  # possible en cas d'edge case mais peu probable
        else:
            return _EPSILON

    # Dans la range → comportement dicté par l'EHS vs seuils
    if action in ('bet', 'raise', 'allin'):
        if ehs_estimate > seuil_bet:
            return 0.90   # très probable si EHS élevé
        elif ehs_estimate > seuil_call:
            return 0.25   # semi-bluff ou protection possible
        else:
            return _EPSILON

    elif action == 'call':
        if pot_odds <= 0:
            # Pas de mise à suivre → call impossible
            return _EPSILON
        if ehs_estimate > pot_odds and ehs_estimate > seuil_call:
            return 0.80
        elif ehs_estimate > pot_odds:
            return 0.40
        else:
            return 0.10   # mauvais call possible pour Calling Station

    elif action == 'check':
        if ehs_estimate > seuil_bet:
            return 0.10   # slow-play possible mais rare pour RangeBot
        elif ehs_estimate > seuil_call:
            return 0.50   # neutre
        else:
            return 0.80   # check attendu avec main faible

    elif action == 'fold':
        if ehs_estimate < seuil_call:
            return 0.70   # fold attendu avec main faible
        elif ehs_estimate < seuil_bet and pot_odds > ehs_estimate:
            return 0.50
        else:
            return _EPSILON  # fold inattendu avec main forte

    return 0.5  # action inconnue → neutre


# =============================================================================
# Range Estimator principal
# =============================================================================

class RangeEstimator:
    """
    Estimateur de range bayésien pour un adversaire donné.

    Usage typique :
        estimator = RangeEstimator()
        estimator.reset()

        # À chaque main :
        for action in game_state.action_history:
            if action.player_id == villain_id:
                estimator.observe_action(action, game_state)

        dist = estimator.get_distribution()
        arch, mut = estimator.get_best_archetype()

    État interne :
        _log_probs : Dict[combo, log(probabilité)] — espace log pour stabilité numérique
        _stats     : VillainStats — stats VPIP/PFR/AF accumulées sur plusieurs mains
        _n_updates : compteur de mises à jour (pour diagnostics)
        _archetype : archétype courant estimé (mis à jour après chaque main)
    """

    def __init__(
        self,
        smoothing:     float = 0.05,
        min_hands_for_archetype: int = 5,
    ):
        """
        Args:
            smoothing              : lissage du prior (voir PriorBuilder)
            min_hands_for_archetype: mains minimum avant de changer d'archétype estimé
        """
        self.smoothing               = smoothing
        self.min_hands_for_archetype = min_hands_for_archetype

        self._stats     = VillainStats()
        self._archetype = 'LAG'     # archétype par défaut (le plus large)
        self._mutation  = 1          # mutation base

        self._log_probs: Dict[str, float] = {}
        self._n_updates = 0
        self._hand_actions: List[dict] = []  # actions de la main en cours

        self.reset()

    # =========================================================================
    # Interface principale
    # =========================================================================

    def reset(self) -> None:
        """
        Réinitialise l'estimateur pour une nouvelle main.
        Les stats multi-mains (VillainStats) sont conservées entre les mains.
        """
        # Reconstruire le prior à partir de l'archétype courant et des stats
        prior = self._build_current_prior()
        self._log_probs = {c: math.log(p) for c, p in prior.items()}
        self._hand_actions = []
        self._n_updates    = 0
        logger.debug(
            "RangeEstimator reset — archetype=%s mutation=%s",
            self._archetype, self._mutation
        )

    def full_reset(self) -> None:
        """
        Réinitialise COMPLÈTEMENT l'estimateur, y compris les VillainStats
        multi-mains (VPIP/PFR/AF, hands_seen).

        À utiliser quand on change d'adversaire (nouveau match), contrairement
        à reset() qui est appelé entre deux mains du MÊME match et conserve
        volontairement les stats accumulées pour affiner la lecture au fil
        du match.

        Sans cet appel, les stats d'un adversaire précédent contaminent le
        prior/archétype estimé du nouvel adversaire (ex: après plusieurs
        matchs contre des TAG/LAG, l'estimateur garde une lecture "moyenne"
        qui ne correspond plus à rien face à un CALLING_STATION suivant).
        """
        self._stats     = VillainStats()
        self._archetype = 'LAG'
        self._mutation  = 1
        self.reset()
        logger.debug("RangeEstimator full_reset — stats et archétype remis à zéro")

    def new_hand(self) -> None:
        """
        Signale le début d'une nouvelle main.
        Met à jour les stats, réévalue l'archétype, puis réinitialise le prior.
        Appelé APRÈS avoir observé toutes les actions de la main précédente.
        """
        self._update_archetype_estimate()
        self.reset()
        self._stats.hands_seen += 1

    def observe_action(
        self,
        action_type: str,
        street:      str,
        pot:         float       = 0.0,
        to_call:     float       = 0.0,
        board:       List[str]   = None,
    ) -> None:
        """
        Observe une action du villain et met à jour la distribution.

        Args:
            action_type : 'fold', 'call', 'raise', 'check', 'bet', 'allin'
            street      : 'preflop', 'flop', 'turn', 'river'
            pot         : taille du pot avant l'action
            to_call     : montant à suivre (0 si check possible)
            board       : cartes du board (pour filtrer les combos impossibles)
        """
        board = board or []
        action_type = action_type.lower()
        street      = street.lower()

        # Enregistrer l'action pour les stats
        self._hand_actions.append({
            'action': action_type,
            'street': street,
            'pot':    pot,
            'to_call': to_call,
        })

        # Mettre à jour les stats VPIP/PFR/AF
        self._update_stats_from_action(action_type, street)

        # Filtrer les combos impossibles (cartes déjà sur le board)
        board_cards = set(board)

        # Pot odds
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0

        # === Mise à jour bayésienne preflop ===
        if street == 'preflop':
            self._update_preflop(action_type)
            return

        # === Mise à jour bayésienne postflop ===
        self._update_postflop(action_type, pot_odds, board_cards)
        self._n_updates += 1

        # Renormaliser périodiquement pour éviter le débordement numérique
        if self._n_updates % 10 == 0:
            self._renormalize()

    def update_from_game_state(
        self,
        game_state,        # GameState dataclass ou dict
        villain_id: int,
    ) -> None:
        """
        Met à jour l'estimateur depuis un GameState complet.

        Rejoue toutes les actions du villain depuis l'historique.
        Idempotent si appelé plusieurs fois sur le même état.

        Compatible avec GameState (dataclass) et dict (format simulateur).
        """
        # Extraire les données depuis GameState ou dict
        if hasattr(game_state, 'action_history'):
            # GameState dataclass
            history  = game_state.action_history
            board    = [str(c) for c in game_state.board]
            pot      = game_state.pot
            to_call  = game_state.to_call
            street   = game_state.street.value if hasattr(game_state.street, 'value') else game_state.street

            # Rejouer les actions
            for action in history:
                pid = action.player_id if hasattr(action, 'player_id') else action.get('player', -1)
                if pid != villain_id:
                    continue
                atype  = action.action_type.value if hasattr(action.action_type, 'value') else str(action.action_type)
                astreet = action.street.value if hasattr(action.street, 'value') else str(action.street)
                self.observe_action(
                    action_type=atype,
                    street=astreet,
                    pot=float(pot),
                    to_call=float(to_call),
                    board=board,
                )

        else:
            # dict (format simulateur)
            history = game_state.get('action_history', [])
            board   = game_state.get('board', [])
            pot     = float(game_state.get('pot', 0))
            to_call = float(game_state.get('to_call', 0))

            for event in history:
                pid = event.get('player', -1)
                if pid != villain_id:
                    continue
                self.observe_action(
                    action_type=event.get('action', ''),
                    street=event.get('street', 'preflop'),
                    pot=pot,
                    to_call=to_call,
                    board=board,
                )

    def update_from_bucket(self, bucket: ActionBucket) -> None:
        """
        Met à jour directement depuis un ActionBucket (Dimension 2).
        Utilisé quand on dispose déjà du bucket calculé par action_history.py.

        Remplace le prior actuel par le prior correspondant au bucket.
        Plus brutal qu'une mise à jour bayésienne, mais utile en début de main
        quand peu d'actions ont été observées.
        """
        range_width = BUCKET_RANGE_WIDTH[bucket]

        # Inférer un archétype depuis le bucket
        bucket_archetype = self._archetype_from_bucket(bucket)

        # Construire un prior mixte : archétype inféré pondéré par range_width
        prior_arch = PriorBuilder.build_prior(bucket_archetype, smoothing=self.smoothing)
        prior_unif = PriorBuilder.uniform_prior()

        # Mélange : range_width fraction de la range archétype, le reste uniforme
        w_arch = range_width
        w_unif = 1.0 - range_width

        mixed = {
            c: w_arch * prior_arch[c] + w_unif * prior_unif[c]
            for c in ALL_COMBOS
        }

        # Normaliser et passer en log
        total = sum(mixed.values())
        self._log_probs = {c: math.log(p / total) for c, p in mixed.items()}
        logger.debug(
            "update_from_bucket: bucket=%s range_width=%.2f archetype=%s",
            bucket.value, range_width, bucket_archetype
        )

    # =========================================================================
    # Résultats
    # =========================================================================

    def get_distribution(self) -> Dict[str, float]:
        """
        Retourne la distribution de probabilité actuelle sur les 1 326 combos.

        Returns:
            Dict[combo, probabilité], normalisé (somme = 1).
        """
        self._renormalize()
        log_max = max(self._log_probs.values())
        # Passage de log → probas avec soustraction du max (stabilité numérique)
        raw   = {c: math.exp(lp - log_max) for c, lp in self._log_probs.items()}
        total = sum(raw.values())
        return {c: p / total for c, p in raw.items()}

    def get_best_archetype(self) -> Tuple[str, int]:
        """
        Retourne l'archétype et la mutation les plus probables.

        Méthode hybride en deux étapes :

        Étape 1 — Score VPIP (si stats disponibles) :
          Compare le VPIP observé au VPIP théorique de chaque mutation.
          C'est le signal le plus discriminant : TAG fold 80%, LAG fold 65%,
          CS fold 60%. Avec les folds maintenant enregistrés, ce signal
          converge rapidement.

        Étape 2 — Score bayésien normalisé :
          Compare la distribution estimée à chaque prior, en normalisant
          par la taille de la range pour éviter le biais vers les ranges larges.
          Score = (masse in-range estimée) / (masse in-range théorique)
          → mesure si notre distribution est concentrée là où le prior l'attend.

        Returns:
            (archetype, mutation) les plus proches de la distribution actuelle.
        """
        dist = self.get_distribution()

        best_arch, best_mut, best_score = 'LAG', 1, -math.inf

        for archetype in ARCHETYPES:
            for mutation in MUTATIONS:
                cfg   = get_range(archetype, mutation)
                prior = PriorBuilder.build_prior(archetype, mutation,
                                                  smoothing=self.smoothing)

                # ── Score 1 : VPIP (si stats disponibles) ─────────────────
                # VPIP observé vs VPIP théorique de cette mutation
                # Poids fort : c'est le signal le plus direct sur la tightness
                vpip_score = 0.0
                if self._stats.hands_seen >= 5:
                    vpip_obs  = self._stats.vpip
                    vpip_theo = cfg.range_pct
                    # Score gaussien : exp(-((vpip_obs - vpip_theo)² / 2σ²))
                    sigma      = 0.08   # tolérance de 8%
                    vpip_score = math.exp(
                        -((vpip_obs - vpip_theo) ** 2) / (2 * sigma ** 2)
                    )

                # ── Score 2 : concentration bayésienne normalisée ──────────
                # Masse que notre distribution place dans la range théorique,
                # divisée par la masse que le prior uniforme y placerait.
                # → > 1 : notre distribution surpondère cette range (bon signe)
                # → < 1 : notre distribution sous-pondère cette range
                in_range_set    = cfg.preflop_combos
                mass_in_est     = sum(dist.get(c, 0) for c in in_range_set)
                mass_in_uniform = len(in_range_set) / len(ALL_COMBOS)

                if mass_in_uniform > 0:
                    concentration = mass_in_est / mass_in_uniform
                else:
                    concentration = 1.0

                # Score log pour stabilité numérique
                bayes_score = math.log(max(concentration, 1e-12))

                # ── Score combiné ──────────────────────────────────────────
                # VPIP dominant si stats disponibles, bayésien sinon
                w_vpip  = min(self._stats.hands_seen / 20.0, 1.0)  # monte à 1 en 20 mains
                w_bayes = 1.0 - w_vpip * 0.5   # toujours au moins 0.5

                score = w_vpip * vpip_score + w_bayes * bayes_score

                if score > best_score:
                    best_score = score
                    best_arch  = archetype
                    best_mut   = mutation

        return best_arch, best_mut

    def get_archetype_probabilities(self) -> Dict[str, float]:
        """
        Retourne la probabilité de chaque archétype (marginale sur les mutations).

        Utilise la même logique hybride VPIP + concentration bayésienne que
        get_best_archetype(), marginalisée sur les mutations.

        Returns:
            Dict['TAG'/'LAG'/'CALLING_STATION', probabilité], somme = 1.
        """
        dist   = self.get_distribution()
        scores: Dict[str, float] = {}

        for archetype in ARCHETYPES:
            arch_score = 0.0

            for mutation in MUTATIONS:
                cfg          = get_range(archetype, mutation)
                in_range_set = cfg.preflop_combos

                # Score VPIP
                vpip_score = 0.0
                if self._stats.hands_seen >= 5:
                    vpip_obs   = self._stats.vpip
                    vpip_theo  = cfg.range_pct
                    sigma      = 0.08
                    vpip_score = math.exp(
                        -((vpip_obs - vpip_theo) ** 2) / (2 * sigma ** 2)
                    )

                # Score concentration
                mass_in_est     = sum(dist.get(c, 0) for c in in_range_set)
                mass_in_uniform = len(in_range_set) / len(ALL_COMBOS)
                concentration   = mass_in_est / max(mass_in_uniform, 1e-12)
                bayes_score     = math.log(max(concentration, 1e-12))

                w_vpip  = min(self._stats.hands_seen / 20.0, 1.0)
                w_bayes = 1.0 - w_vpip * 0.5

                mut_score   = w_vpip * vpip_score + w_bayes * bayes_score
                arch_score += mut_score / len(MUTATIONS)   # moyenne sur mutations

            # Passage en espace positif pour softmax
            scores[archetype] = math.exp(max(arch_score, -500))

        total = sum(scores.values())
        if total > 0:
            return {k: v / total for k, v in scores.items()}
        return {a: 1.0 / 3 for a in ARCHETYPES}

    def get_combo_probability(self, card1: str, card2: str) -> float:
        """Probabilité estimée pour une main spécifique."""
        combo1 = card1 + card2
        combo2 = card2 + card1
        dist   = self.get_distribution()
        return dist.get(combo1, dist.get(combo2, 0.0))

    def get_stats(self) -> VillainStats:
        """Retourne les stats VPIP/PFR/AF accumulées."""
        return self._stats

    # =========================================================================
    # Mise à jour bayésienne preflop
    # =========================================================================

    def _update_preflop(self, action_type: str) -> None:
        """
        Mise à jour preflop : ajuste la distribution selon l'action.

        Pour les RangeBots :
          raise → dans range (quasi-certain)
          fold  → hors range (quasi-certain)
          call  → signal ambigu (Calling Station peut call avec range large)

        post_blind (mise forcée SB/BB, cf. game_state.py::ActionType) est
        explicitement exclu : ce n'est jamais une décision volontaire du
        joueur, elle ne doit donc jamais influencer la distribution
        estimée. Sortie anticipée plutôt que de compter sur le repli
        "action inconnue → neutre" plus bas (qui serait un no-op
        mathématique après renormalisation, mais itérerait les 1326
        combos pour rien, et laisserait ce cas non documenté — bug trouvé
        en session 6, où les blindes étaient encore taguées RAISE et
        faussaient tout le calcul).
        """
        if action_type == 'post_blind':
            return

        in_range_set = get_range(self._archetype, self._mutation).preflop_combos

        for combo in ALL_COMBOS:
            c1, c2 = combo[:2], combo[2:]
            c1_rev, c2_rev = combo[2:], combo[:2]
            in_range = (combo in in_range_set
                       or (c1_rev + c2_rev) in in_range_set)

            if action_type in ('raise', 'open', '3bet', '4bet', 'allin'):
                if in_range:
                    likelihood = _L_PF_RAISE_IN
                else:
                    likelihood = _L_PF_RAISE_OUT

            elif action_type == 'fold':
                if in_range:
                    likelihood = _L_PF_FOLD_IN
                else:
                    likelihood = _L_PF_FOLD_OUT

            elif action_type in ('call', 'limp'):
                if in_range:
                    likelihood = _L_PF_CALL_IN
                else:
                    likelihood = _L_PF_CALL_OUT

            else:
                likelihood = 0.5  # action inconnue → neutre

            if likelihood > 0:
                self._log_probs[combo] = self._log_probs.get(combo, 0.0) + math.log(likelihood)

    # =========================================================================
    # Mise à jour bayésienne postflop
    # =========================================================================

    def _update_postflop(
        self,
        action_type: str,
        pot_odds:    float,
        board_cards: set,
    ) -> None:
        """
        Mise à jour postflop : applique les likelihoods sur chaque combo.

        Pour chaque combo possible :
          1. Vérifier que les cartes ne sont pas sur le board
          2. Estimer l'EHS du combo (simplification : basé sur la classe de main)
          3. Calculer P(action | combo, archétype, EHS)
          4. Mettre à jour log_prob[combo] += log(likelihood)
        """
        in_range_set = get_range(self._archetype, self._mutation).preflop_combos

        for combo in ALL_COMBOS:
            c1, c2 = combo[:2], combo[2:]

            # Éliminer les combos impossibles (cartes sur le board)
            if c1 in board_cards or c2 in board_cards:
                self._log_probs[combo] = -math.inf
                continue

            # Vérifier si combo dans la range preflop estimée
            in_range = (combo in in_range_set
                       or (c2 + c1) in in_range_set)

            # Estimation EHS simplifiée basée sur la classe de main
            # (sans accès au poker_engine ici — l'EHS réel est calculé par l'EHSBot)
            ehs_est = self._estimate_ehs_from_class(c1, c2)

            likelihood = _postflop_likelihoods(
                action_type, self._archetype, ehs_est, pot_odds, in_range
            )

            if likelihood > 0:
                self._log_probs[combo] = self._log_probs.get(combo, 0.0) + math.log(likelihood)
            else:
                self._log_probs[combo] = -math.inf

    # =========================================================================
    # Helpers
    # =========================================================================

    def _build_current_prior(self) -> Dict[str, float]:
        """Construit le prior en fonction de l'archétype et des stats actuels."""
        if self._stats.is_reliable:
            # Stats suffisantes → prior précis sur l'archétype + mutation estimée
            return PriorBuilder.build_prior(
                self._archetype, self._mutation, self.smoothing
            )
        elif self._stats.hands_seen >= self.min_hands_for_archetype:
            # Quelques mains vues → prior sur l'archétype, mutations mélangées
            return PriorBuilder.build_prior(
                self._archetype, None, self.smoothing
            )
        else:
            # Trop peu d'info → mélange uniforme des 3 archétypes
            # (évite le biais vers LAG qui fausse l'identification TAG)
            priors = [PriorBuilder.build_prior(a, None, self.smoothing)
                      for a in ARCHETYPES]
            mixed  = {c: sum(p[c] for p in priors) / 3.0 for c in ALL_COMBOS}
            total  = sum(mixed.values())
            return {c: p / total for c, p in mixed.items()}

    def _update_archetype_estimate(self) -> None:
        """Met à jour l'archétype estimé depuis les stats et la distribution."""
        if self._stats.is_reliable:
            inferred = self._stats.infer_archetype()
            if inferred != self._archetype:
                logger.debug(
                    "Archétype mis à jour : %s → %s (VPIP=%.2f PFR=%.2f AF=%.1f)",
                    self._archetype, inferred,
                    self._stats.vpip, self._stats.pfr, self._stats.af,
                )
            self._archetype = inferred

        elif self._stats.hands_seen >= self.min_hands_for_archetype:
            # Estimation partielle depuis les actions observées
            self._archetype = self._stats.infer_archetype()

        # Estimer la mutation (tight vs loose au sein de l'archétype)
        self._mutation = self._estimate_mutation()

    def _estimate_mutation(self) -> int:
        """
        Estime la mutation (0/1/2) au sein de l'archétype.

        Méthode : comparer le VPIP observé aux VPIP théoriques des 3 mutations.
        La mutation avec le VPIP le plus proche de l'observé est sélectionnée.
        """
        if not self._stats.is_reliable:
            return 1  # mutation médiane par défaut

        vpip_obs = self._stats.vpip
        best_mut, best_diff = 0, math.inf

        for mut in MUTATIONS:
            cfg       = get_range(self._archetype, mut)
            vpip_theo = cfg.range_pct  # % de mains dans la range ≈ VPIP théorique
            diff      = abs(vpip_obs - vpip_theo)
            if diff < best_diff:
                best_diff = diff
                best_mut  = mut

        return best_mut

    def _update_stats_from_action(self, action_type: str, street: str) -> None:
        """Met à jour les stats VPIP/PFR/AF depuis une action observée."""
        if action_type == 'post_blind':
            # Mise forcée, jamais volontaire — ne doit jamais compter comme
            # VPIP/PFR/aggression. Exclusion explicite plutôt que de
            # compter sur le fait qu'aucune condition ci-dessous ne
            # matche 'post_blind' (bug trouvé en session 6 : les blindes
            # étaient taguées RAISE, comptant donc à tort comme VPIP=PFR=True
            # systématiquement).
            return
        if street == 'preflop':
            if action_type in ('call', 'limp'):
                self._stats.hands_vpip += 1
            elif action_type in ('raise', 'open', '3bet', '4bet'):
                self._stats.hands_vpip += 1
                self._stats.hands_pfr  += 1
        else:
            if action_type in ('bet', 'raise', 'allin'):
                self._stats.aggressive_acts += 1
            elif action_type in ('call', 'check'):
                self._stats.passive_acts += 1

    def _estimate_ehs_from_class(self, card1: str, card2: str) -> float:
        """
        Estimation heuristique de l'EHS basée sur la classe de main.
        Utilisée pour les likelihoods postflop quand le poker_engine n'est pas
        disponible dans le Range Estimator.

        Cette estimation est volontairement grossière — le Range Estimator
        n'a pas accès aux cartes du villain. Il distingue simplement :
          - mains premium (AA, KK, AK...) → EHS élevé
          - mains moyennes (paires, broadways) → EHS moyen
          - mains faibles (connecteurs, bas) → EHS bas

        L'EHS précis est calculé par l'EHSBot via poker_engine.
        """
        r1, s1 = card1[0], card1[1]
        r2, s2 = card2[0], card2[1]
        v1 = RANK_VAL[r1]
        v2 = RANK_VAL[r2]
        high = max(v1, v2)
        low  = min(v1, v2)
        is_pair   = (r1 == r2)
        is_suited = (s1 == s2)

        if is_pair:
            return 0.35 + high * 0.03   # paire de 2 → 0.35, paire d'As → 0.71
        elif high >= 12:  # As
            return 0.55 + (low / 12.0) * 0.15
        elif high >= 10:  # T, J, Q, K
            base = 0.45 + (high - 10) * 0.03
            if is_suited:
                base += 0.05
            return base
        else:
            base = 0.30 + high * 0.01
            if is_suited:
                base += 0.03
            return base

    def _archetype_from_bucket(self, bucket: ActionBucket) -> str:
        """Infère un archétype depuis un bucket d'action."""
        aggressive_buckets = {
            ActionBucket.THREBET_POT_BET,
            ActionBucket.THREBET_POT_CHECK_RAISE,
            ActionBucket.SQUEEZE_4BET,
            ActionBucket.CHECK_RAISE_FLOP,
            ActionBucket.CHECK_RAISE_TURN,
        }
        passive_buckets = {
            ActionBucket.LIMP_PASSIVE,
            ActionBucket.LIMP_CALL,
        }
        tight_buckets = {
            ActionBucket.THREBET_POT_BET,
            ActionBucket.THREBET_POT_PASSIVE,
            ActionBucket.SQUEEZE_4BET,
        }

        if bucket in tight_buckets:
            return 'TAG'
        elif bucket in passive_buckets:
            return 'CALLING_STATION'
        elif bucket in aggressive_buckets:
            return 'LAG'
        return 'LAG'

    def _renormalize(self) -> None:
        """Renormalise les log-probs pour éviter le débordement numérique."""
        finite = [lp for lp in self._log_probs.values() if lp > -math.inf]
        if not finite:
            # Tous à -inf : reset au prior
            prior = self._build_current_prior()
            self._log_probs = {c: math.log(p) for c, p in prior.items()}
            return
        log_max = max(finite)
        # Soustraire le max (décale la distribution sans changer les ratios)
        self._log_probs = {
            c: (lp - log_max if lp > -math.inf else -math.inf)
            for c, lp in self._log_probs.items()
        }

    def __repr__(self) -> str:
        return (
            f"RangeEstimator("
            f"archetype={self._archetype}, "
            f"mutation={self._mutation}, "
            f"hands_seen={self._stats.hands_seen}, "
            f"updates={self._n_updates})"
        )


# =============================================================================
# Métriques de calibration (placées ici pour import uniforme)
# =============================================================================

def brier_score(
    estimated_probs: Dict[str, float],
    true_combos: set,
) -> float:
    """
    Calcule le Brier Score : mesure la qualité des probabilités estimées.

    Brier = (1/N) × sum_{combo} (P_estimée(combo) - P_réelle(combo))²

    ∈ [0, 1]. 0 = parfait. Référence : prior uniforme ≈ 2 × r × (1-r)
    """
    if not true_combos:
        return 1.0

    n      = len(ALL_COMBOS)
    p_true = 1.0 / len(true_combos)
    total  = sum(estimated_probs.values())

    if total <= 0:
        normalized = {c: 1.0 / n for c in estimated_probs}
    else:
        normalized = {c: p / total for c, p in estimated_probs.items()}

    brier = 0.0
    for combo in ALL_COMBOS:
        p_est  = normalized.get(combo, 0.0)
        p_real = p_true if combo in true_combos else 0.0
        brier += (p_est - p_real) ** 2

    return brier / n
