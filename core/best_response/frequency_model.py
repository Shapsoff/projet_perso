"""
frequency_model.py — Fold/Call/Raise Frequency Model
Phase 5 — Bot Poker Académique

Calcule analytiquement les fréquences de réponse adverse (fold / call / raise)
pour chaque combo de la range estimée, en fonction du sizing du bet et de
l'archétype estimé par le Range Estimator.

Principe :
    Le comportement postflop des RangeBots est entièrement déterminé par leurs
    seuils EHS (définis dans range_definitions.py). Pour un bet donné, chaque
    combo de la range adverse fold / call / raise selon que son EHS est
    inférieur, intermédiaire ou supérieur aux seuils de l'archétype.

    L'EHS de chaque combo est calculé via poker_engine C++ sur le board courant
    (pas une heuristique). C'est la différence fondamentale avec le Range
    Estimator qui utilisait _estimate_ehs_from_class() — ici on veut la précision
    maximale pour le calcul d'EV.

Sorties :
    FoldCallRaiseResult — pour chaque action possible de l'adversaire :
        - fold_range   : PlayerRange (combo → proba) des combos qui foldent
        - call_range   : PlayerRange des combos qui callent
        - raise_range  : PlayerRange des combos qui raisent
        - p_fold       : probabilité globale que l'adversaire folde
        - p_call       : probabilité globale qu'il calle
        - p_raise      : probabilité globale qu'il raise

    Ces ranges sont passées directement au MonteCarloEngine pour le calcul d'EV.

Interface publique :
    model = FrequencyModel(ehs_calculator)
    result = model.compute(
        distribution,   # Dict[combo, proba] depuis RangeEstimator
        archetype,      # 'TAG' | 'LAG' | 'CALLING_STATION'
        board,          # cartes du board (pour EHS réel)
        bet_sizing,     # fraction du pot (ex: 0.75)
        pot,            # taille du pot
        to_call,        # montant déjà à suivre (0 si on ouvre)
        n_opponents,    # nombre d'adversaires actifs
        street,         # 'flop' | 'turn' | 'river'
    )

Calibration :
    Les seuils EHS (seuil_bet, seuil_call) viennent directement de
    range_definitions.ArchetypeConfig — ils sont la vérité terrain des RangeBots.
    Pour des joueurs réels (phase future), ces seuils seront remplacés par des
    valeurs calibrées depuis une Player DB.

Limite connue et acceptée :
    Les seuils EHS sont ceux de la mutation 0 (base) de l'archétype estimé.
    Les mutations ajustent la range preflop, pas les seuils postflop — cette
    approximation est acceptable et cohérente avec l'implémentation des RangeBots.

Performance :
    Le calcul EHS sur les 1 326 combos est la partie coûteuse. On utilise
    n_sims=500 (vs 10 000 pour la décision finale) pour rester sous ~50ms.
    Le cache board_ehs_cache évite de recalculer si le board n'a pas changé.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# PlayerRange : format attendu par MonteCarloEngine
PlayerRange = Dict[str, float]

# Seuil de probabilité minimal pour inclure un combo dans une range
# (évite de passer des poids négligeables au Monte Carlo)
_MIN_COMBO_PROB = 1e-6

# Simulations EHS pour le calcul des fréquences (compromis vitesse/précision)
_EHS_SIMS_FREQUENCY = 500


# =============================================================================
# Résultat du frequency model
# =============================================================================

@dataclass
class FoldCallRaiseResult:
    """
    Résultat du calcul de fréquences pour un sizing donné.

    Contient les trois ranges pondérées (pour Monte Carlo) et les probabilités
    globales (pour le calcul d'EV sans Monte Carlo, ex : EV(fold) = p_fold × pot).

    Attributes:
        fold_range   : combos qui foldent, pondérés par leur probabilité dans
                       la distribution estimée. Somme des poids = p_fold.
        call_range   : combos qui callent (idem).
        raise_range  : combos qui raisent (idem).
        p_fold       : probabilité pondérée que l'adversaire folde.
        p_call       : probabilité pondérée qu'il calle.
        p_raise      : probabilité pondérée qu'il raise.
        archetype    : archétype utilisé pour le calcul.
        bet_sizing   : sizing du bet (fraction du pot).
        n_combos_ehs : nombre de combos pour lesquels l'EHS a été calculé.
    """
    fold_range:    PlayerRange
    call_range:    PlayerRange
    raise_range:   PlayerRange
    p_fold:        float
    p_call:        float
    p_raise:       float
    archetype:     str
    bet_sizing:    float
    n_combos_ehs:  int = 0

    def __post_init__(self):
        # Vérification de cohérence (tolérance numérique)
        total = self.p_fold + self.p_call + self.p_raise
        if abs(total - 1.0) > 0.01:
            logger.warning(
                "FoldCallRaiseResult : p_fold+p_call+p_raise=%.4f (attendu 1.0)",
                total
            )

    def summary(self) -> str:
        return (
            f"FoldCallRaise(arch={self.archetype}, "
            f"sizing={self.bet_sizing:.0%}, "
            f"fold={self.p_fold:.1%}, "
            f"call={self.p_call:.1%}, "
            f"raise={self.p_raise:.1%})"
        )


# =============================================================================
# Seuils EHS par archétype
# =============================================================================

def _get_archetype_thresholds(archetype: str) -> Tuple[float, float]:
    """
    Retourne (seuil_bet, seuil_call) pour un archétype.

    Ces seuils déterminent le comportement postflop :
      - EHS > seuil_bet  → raise (si face à un bet) ou bet
      - EHS > seuil_call → call (si face à un bet)
      - EHS < seuil_call → fold (si face à un bet) ou check

    Valeurs calibrées depuis range_definitions.py (mutation 0 = base).
    On importe à la demande pour éviter les imports circulaires.

    Returns:
        (seuil_bet, seuil_call)
    """
    try:
        from core.bots.range_definitions import get_range
        cfg = get_range(archetype, mutation=0)
        return cfg.postflop_ehs_threshold, cfg.postflop_ehs_call
    except Exception:
        # Fallback si range_definitions non disponible
        defaults = {
            'TAG':             (0.65, 0.55),
            'LAG':             (0.45, 0.35),
            'CALLING_STATION': (0.75, 0.30),
        }
        return defaults.get(archetype, (0.60, 0.40))


def _get_calling_station_bet_threshold() -> float:
    """Seuil de bet pour la Calling Station (passif — ne bet que très fort)."""
    try:
        from core.bots.range_definitions import get_range
        return get_range('CALLING_STATION', 0).postflop_ehs_threshold
    except Exception:
        return 0.75


# =============================================================================
# Classificateur d'action par combo
# =============================================================================

def classify_combo_response(
    ehs:          float,
    archetype:    str,
    bet_sizing:   float,
    pot:          float,
    to_call:      float,
    in_range:     bool,
    aggression:   str = 'aggressive',
) -> str:
    """
    Détermine si un combo fold / call / raise face à notre bet.

    Logique miroir du comportement RangeBot (_decide_postflop) :
      - Si hors range preflop → fold systématique
      - Si in_range + EHS > seuil_bet → raise
      - Si in_range + EHS > seuil_call + pot odds rentables → call
      - Sinon → fold

    Pour un check de notre part (bet_sizing=0), la logique est différente :
      - check ou bet selon EHS vs seuil_bet (pas de fold possible)

    Args:
        ehs        : EHS réel calculé par poker_engine pour ce combo
        archetype  : archétype estimé de l'adversaire
        bet_sizing : notre bet en fraction du pot (0 = check)
        pot        : taille du pot avant notre action
        to_call    : montant que l'adversaire doit suivre (= notre bet)
        in_range   : ce combo est-il dans la range preflop estimée
        aggression : 'aggressive' (TAG/LAG) ou 'passive' (CALLING_STATION)

    Returns:
        'fold' | 'call' | 'raise'
    """
    seuil_bet, seuil_call = _get_archetype_thresholds(archetype)

    # Hors range preflop → fold systématique (comportement RangeBot)
    if not in_range:
        return 'fold'

    # Pas de bet de notre part (check) → l'adversaire bet ou check
    # On modélise ça comme 'call' (neutre, pas de fold possible face à un check)
    if bet_sizing == 0.0 or to_call == 0.0:
        if ehs > seuil_bet:
            return 'raise'  # il bet dans notre check
        return 'call'       # il check (on traite comme call pour le calcul d'EV)

    # Face à notre bet : fold / call / raise
    # Pot odds que l'adversaire reçoit
    total_pot = pot + to_call
    pot_odds  = to_call / total_pot if total_pot > 0 else 0.0

    if aggression == 'aggressive':
        # TAG / LAG : raise si très fort, call si rentable, fold sinon
        if ehs > seuil_bet:
            return 'raise'
        if ehs > seuil_call and ehs > pot_odds:
            return 'call'
        return 'fold'

    else:
        # Calling Station : call très large, raise rare (seuil élevé)
        if ehs > seuil_bet:
            return 'raise'
        if ehs > seuil_call:
            # CS call même si légèrement non rentable (comportement passif)
            return 'call'
        return 'fold'


# =============================================================================
# Frequency Model principal
# =============================================================================

class FrequencyModel:
    """
    Calcule les fréquences de réponse adverse (fold/call/raise) pour chaque
    sizing de bet envisagé par le Best-Response Engine.

    Le calcul est analytique (pas de simulation) :
      1. Pour chaque combo de la distribution estimée avec proba > seuil :
         a. Calculer l'EHS réel via poker_engine C++ (ou heuristique si absent)
         b. Classifier : fold / call / raise selon seuils archétype + pot odds
         c. Accumuler le poids du combo dans la range correspondante
      2. Normaliser p_fold, p_call, p_raise (somme = 1)
      3. Les trois PlayerRange sont passées au MonteCarloEngine

    Cache :
        _ehs_cache : Dict[(combo, board_key), float]
        Évite de recalculer l'EHS du même combo sur le même board.
        Invalidé automatiquement si le board change.
    """

    def __init__(self, ehs_calculator=None, n_sims: int = _EHS_SIMS_FREQUENCY):
        """
        Args:
            ehs_calculator : instance de poker_engine.EHSCalculator (optionnel).
                             Si None, on utilise l'estimation heuristique.
            n_sims         : simulations Monte Carlo pour l'EHS des combos.
                             500 est un bon compromis vitesse/précision ici.
        """
        self._ehs_calc  = ehs_calculator
        self._n_sims    = n_sims
        self._ehs_cache: Dict[Tuple[str, str], float] = {}
        self._board_key = ""  # clé du board courant (pour invalidation cache)

        logger.debug(
            "FrequencyModel initialisé (ehs_calc=%s, n_sims=%d)",
            "C++" if ehs_calculator else "heuristique",
            n_sims,
        )

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
        """
        Calcule les fréquences fold/call/raise pour un sizing donné.

        Args:
            distribution : Dict[combo, proba] depuis RangeEstimator.get_distribution()
            archetype    : archétype estimé ('TAG', 'LAG', 'CALLING_STATION')
            board        : cartes du board (ex: ['Ah', 'Kd', '7c'])
            bet_sizing   : notre bet en fraction du pot (ex: 0.75).
                           0.0 = check (pas de fold possible).
            pot          : taille du pot avant notre action
            n_opponents  : nombre d'adversaires (pour le calcul EHS)
            street       : street courante

        Returns:
            FoldCallRaiseResult avec les trois ranges et probabilités.
        """
        # Invalider le cache si le board a changé
        board_key = "_".join(sorted(board))
        if board_key != self._board_key:
            self._ehs_cache.clear()
            self._board_key = board_key

        # Récupérer l'aggression de l'archétype
        aggression = self._get_aggression(archetype)

        # Récupérer la range preflop estimée (pour in_range check)
        preflop_range = self._get_preflop_range(archetype)

        # Montant à suivre pour l'adversaire = notre bet
        to_call = bet_sizing * pot

        # Accumulateurs
        fold_weights:  Dict[str, float] = {}
        call_weights:  Dict[str, float] = {}
        raise_weights: Dict[str, float] = {}
        total_weight  = 0.0
        n_ehs_computed = 0

        # Cartes déjà utilisées (board) — pour filtrer les combos impossibles
        board_set = set(board)

        for combo, prob in distribution.items():
            if prob < _MIN_COMBO_PROB:
                continue

            c1, c2 = combo[:2], combo[2:]

            # Filtrer les combos dont les cartes sont sur le board
            if c1 in board_set or c2 in board_set:
                continue

            total_weight += prob

            # EHS réel pour ce combo sur ce board
            cache_key = (combo, board_key)
            if cache_key in self._ehs_cache:
                ehs = self._ehs_cache[cache_key]
            else:
                ehs = self._compute_ehs(c1, c2, board, n_opponents)
                self._ehs_cache[cache_key] = ehs
                n_ehs_computed += 1

            # Vérifier si combo dans la range preflop
            in_range = (combo in preflop_range
                        or (c2 + c1) in preflop_range)

            # Classifier la réponse
            response = classify_combo_response(
                ehs=ehs,
                archetype=archetype,
                bet_sizing=bet_sizing,
                pot=pot,
                to_call=to_call,
                in_range=in_range,
                aggression=aggression,
            )

            if response == 'fold':
                fold_weights[combo]  = prob
            elif response == 'call':
                call_weights[combo]  = prob
            else:
                raise_weights[combo] = prob

        # Éviter la division par zéro si distribution vide
        if total_weight <= 0:
            logger.warning(
                "FrequencyModel : distribution vide ou entièrement filtrée "
                "(board=%s, archetype=%s)",
                board, archetype,
            )
            return FoldCallRaiseResult(
                fold_range={}, call_range={}, raise_range={},
                p_fold=1.0, p_call=0.0, p_raise=0.0,
                archetype=archetype, bet_sizing=bet_sizing,
            )

        # Probabilités globales
        p_fold  = sum(fold_weights.values())  / total_weight
        p_call  = sum(call_weights.values())  / total_weight
        p_raise = sum(raise_weights.values()) / total_weight

        # Normalisation numérique (somme exactement = 1)
        total_p = p_fold + p_call + p_raise
        if total_p > 0:
            p_fold  /= total_p
            p_call  /= total_p
            p_raise /= total_p

        logger.debug(
            "FrequencyModel: arch=%s sizing=%.0f%% | "
            "fold=%.1f%% call=%.1f%% raise=%.1f%% | "
            "combos=%d ehs_computed=%d",
            archetype, bet_sizing * 100,
            p_fold * 100, p_call * 100, p_raise * 100,
            len(fold_weights) + len(call_weights) + len(raise_weights),
            n_ehs_computed,
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

    def compute_multiple_sizings(
        self,
        distribution: Dict[str, float],
        archetype:    str,
        board:        List[str],
        sizings:      List[float],
        pot:          float,
        n_opponents:  int = 1,
        street:       str = 'flop',
    ) -> Dict[float, FoldCallRaiseResult]:
        """
        Calcule les fréquences pour plusieurs sizings d'un coup.

        Optimisation : l'EHS de chaque combo est calculé une seule fois
        (partagé entre tous les sizings via le cache), ce qui rend le coût
        O(n_combos × n_sims + n_sizings) au lieu de O(n_combos × n_sims × n_sizings).

        Args:
            sizings : liste de fractions du pot (ex: [0.33, 0.5, 0.75, 1.0])

        Returns:
            Dict[sizing, FoldCallRaiseResult]
        """
        # Précalculer les EHS pour tous les combos une seule fois
        board_key = "_".join(sorted(board))
        if board_key != self._board_key:
            self._ehs_cache.clear()
            self._board_key = board_key

        board_set = set(board)
        for combo, prob in distribution.items():
            if prob < _MIN_COMBO_PROB:
                continue
            c1, c2 = combo[:2], combo[2:]
            if c1 in board_set or c2 in board_set:
                continue
            cache_key = (combo, board_key)
            if cache_key not in self._ehs_cache:
                self._ehs_cache[cache_key] = self._compute_ehs(
                    c1, c2, board, n_opponents
                )

        # Calculer les fréquences pour chaque sizing (cache chaud → rapide)
        return {
            sizing: self.compute(
                distribution=distribution,
                archetype=archetype,
                board=board,
                bet_sizing=sizing,
                pot=pot,
                n_opponents=n_opponents,
                street=street,
            )
            for sizing in sizings
        }

    # =========================================================================
    # Helpers
    # =========================================================================

    def _compute_ehs(
        self,
        card1:       str,
        card2:       str,
        board:       List[str],
        n_opponents: int,
    ) -> float:
        """
        Calcule l'EHS d'un combo via poker_engine C++ ou heuristique.

        On utilise calculate_multiway avec notre propre main = [card1, card2]
        du point de vue de l'adversaire. Le résultat est l'EHS de ce combo
        sur le board courant.

        Note : ici on calcule l'EHS "de l'adversaire" — c'est l'EHS du combo
        adverse, pas le nôtre. On ne connaît pas nos propres cartes dans ce
        contexte (on calcule pour toute la range adverse).
        """
        if self._ehs_calc is not None and len(board) >= 3:
            try:
                result = self._ehs_calc.calculate_multiway(
                    [card1, card2], board, n_opponents, self._n_sims
                )
                return result.EHS
            except Exception as e:
                logger.debug("EHS calc error pour %s%s: %s", card1, card2, e)
                # Fallback sur l'heuristique
        return self._estimate_ehs_heuristic(card1, card2)

    def _estimate_ehs_heuristic(self, card1: str, card2: str) -> float:
        """
        Estimation heuristique de l'EHS (identique à range_estimator.py).
        Utilisée si poker_engine n'est pas disponible.
        """
        _RANK_VAL = {r: i for i, r in enumerate(
            ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        )}
        r1, s1 = card1[0], card1[1]
        r2, s2 = card2[0], card2[1]
        v1, v2 = _RANK_VAL.get(r1, 6), _RANK_VAL.get(r2, 6)
        high, low = max(v1, v2), min(v1, v2)
        is_pair   = (r1 == r2)
        is_suited = (s1 == s2)

        if is_pair:
            return 0.35 + high * 0.03
        elif high >= 12:
            return 0.55 + (low / 12.0) * 0.15
        elif high >= 10:
            base = 0.45 + (high - 10) * 0.03
            return base + (0.05 if is_suited else 0.0)
        else:
            base = 0.30 + high * 0.01
            return base + (0.03 if is_suited else 0.0)

    def _get_aggression(self, archetype: str) -> str:
        """Retourne le style postflop de l'archétype."""
        try:
            from core.bots.range_definitions import get_range
            return get_range(archetype, 0).aggression
        except Exception:
            return 'passive' if archetype == 'CALLING_STATION' else 'aggressive'

    def _get_preflop_range(self, archetype: str) -> set:
        """
        Retourne la range preflop de l'archétype estimé (mutation 0).
        Utilisée pour déterminer si un combo est in_range.

        On prend la mutation 0 car le frequency model ne connaît pas la mutation
        exacte — c'est une approximation acceptable (les seuils postflop ne
        changent pas entre mutations).
        """
        try:
            from core.bots.range_definitions import get_range, MUTATIONS
            # Union des 3 mutations pour éviter de fold des combos plausibles
            combined = set()
            for mut in MUTATIONS:
                combined |= get_range(archetype, mut).preflop_combos
            return combined
        except Exception:
            return set()

    def clear_cache(self) -> None:
        """Vide le cache EHS (utile entre les streets)."""
        self._ehs_cache.clear()
        self._board_key = ""

    def __repr__(self) -> str:
        return (
            f"FrequencyModel("
            f"ehs={'C++' if self._ehs_calc else 'heuristique'}, "
            f"n_sims={self._n_sims}, "
            f"cache_size={len(self._ehs_cache)})"
        )
