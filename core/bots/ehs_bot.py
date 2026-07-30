"""
ehs_bot.py — EHSBot v4 : EHSBot v3 + Best-Response Engine (Phase 5 — P1)
Bot Poker Académique

Greffe le Best-Response Engine (Phase 5) sur l'EHSBot v3.
Rétro-compatible à 100% : même interface, mêmes noms de classes, même dict
game_state que le simulateur phase 3/4.

Changements vs v3 :
  - EHSBotConfig : champ use_best_response + BestResponseConfig embarquée
  - EHSBot.__init__() : instancie le BestResponseEngine si disponible
  - EHSBot.decide() : branche sur _make_decision_v4() si Best-Response actif,
    sinon délègue à _make_decision() v3 (comportement inchangé)
  - EHSBot.name : "EHSBot_v4"

Rétro-compatibilité garantie :
  - Les noms EHSBot / EHSBotConfig sont conservés
  - Le dict game_state accepté est identique aux versions précédentes
  - use_best_response=False → comportement v3 exact, aucune régression
  - Si core.best_response est absent → comportement v3 exact, aucune erreur
  - Si Range Estimator a < min_hands données → fallback v3 automatique
    (géré dans BestResponseEngine, transparent ici)

Pipeline de décision v4 :
  game_state
    → [EHSCalculator C++]    → ehs_result       (Dim 1)
    → [ActionHistory]        → bucket + modifier (Dim 2)
    → [BoardTexture]         → texture_modifier  (Dim 3)
    → [SPR]                  → spr_info          (Dim 4)
    → [RangeEstimator]       → distribution      (Dim 5)
    → [BestResponseEngine]   → EV par action → meilleure action  (Phase 5)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import logging
import uuid

try:
    import poker_engine
    _POKER_ENGINE_AVAILABLE = True
except ImportError:
    poker_engine = None
    _POKER_ENGINE_AVAILABLE = False

from core.board_texture import classify_board, BoardTexture
from core.spr import get_spr_info_from_game_state, SPRCategory, SPRInfo
from core.action_history import (
    classify_history,
    get_villain_bucket_from_state,
    get_aggregate_bucket,
    get_bucket_info,
    ActionBucket,
    BUCKET_EHS_MODIFIER,
)

# Import Range Estimator (phase 4) — optionnel
try:
    from core.range_estimator import RangeEstimator
    _RANGE_ESTIMATOR_AVAILABLE = True
except ImportError:
    RangeEstimator = None
    _RANGE_ESTIMATOR_AVAILABLE = False

# Import Best-Response Engine (phase 5) — optionnel
try:
    from core.best_response.best_response_engine import (
        BestResponseEngine,
        BestResponseConfig,
    )
    from core.best_response.frequency_model import FrequencyModel
    _BEST_RESPONSE_AVAILABLE = True
except ImportError:
    BestResponseEngine  = None
    BestResponseConfig  = None
    FrequencyModel      = None
    _BEST_RESPONSE_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Types d'actions
# =============================================================================

@dataclass
class Action:
    """Action retournée par le bot."""
    action_type: str          # 'fold', 'check', 'call', 'bet', 'raise', 'allin'
    amount:      float = 0.0
    sizing_pct:  float = 0.0  # sizing en % du pot (pour les logs)

    def __str__(self):
        if self.action_type in ('fold', 'check'):
            return self.action_type
        if self.action_type == 'call':
            return f"call {self.amount:.0f}"
        if self.action_type == 'allin':
            return f"allin {self.amount:.0f}"
        return f"{self.action_type} {self.amount:.0f} ({self.sizing_pct*100:.0f}% pot)"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EHSBotConfig:
    """Paramètres configurables de l'EHSBot v4 (rétro-compatible v2/v3)."""

    # ── Dimensions 1-4 (identiques v2) ───────────────────────────────────────
    n_sims:                int   = 10000
    ehs_high_threshold:    float = 0.70
    ehs_low_threshold:     float = 0.45
    spr_commit_threshold:  float = 2.0
    sizing_protection:     float = 0.33
    sizing_max:            float = 1.00
    use_board_texture:     bool  = True
    use_spr_modifier:      bool  = True
    use_action_history:    bool  = True
    sizing_scale_factor:   float = 1.4
    default_n_opponents:   int   = 1
    bucket_modifier_clamp: float = 0.12

    # ── Dimension 5 — Range Estimator (phase 4) ───────────────────────────────
    use_range_estimator:   bool  = True

    sizing_modifier_by_archetype: Dict[str, float] = field(default_factory=lambda: {
        'TAG':             +0.10,
        'LAG':             -0.05,
        'CALLING_STATION': +0.15,
    })

    ehs_threshold_by_archetype: Dict[str, float] = field(default_factory=lambda: {
        'TAG':             +0.03,
        'LAG':             +0.05,
        'CALLING_STATION': -0.03,
    })

    archetype_confidence_threshold: float = 0.50
    min_hands_for_range_estimator:  int   = 10
    range_estimator_smoothing:      float = 0.05

    # ── Phase 5 — Best-Response Engine ────────────────────────────────────────
    # Active le Best-Response Engine a la place de _make_decision() v3.
    # Si False ou si core.best_response est absent -> comportement v3 exact.
    use_best_response: bool = True

    # Sizings evalues par le Best-Response Engine (fraction du pot).
    best_response_sizings: list = field(
        default_factory=lambda: [0.33, 0.50, 0.75, 1.00, 2.00]
    )

    # Simulations Monte Carlo pour le calcul d'EV (compromis vitesse/precision).
    best_response_mc_sims: int = 5000

    # Simulations pour le FrequencyModel (EHS des combos adverses).
    best_response_freq_sims: int = 500

    # Mains minimum avant d'activer le Best-Response
    # (en dessous -> fallback v3 automatique).
    best_response_min_hands: int = 30   # cohérent avec PlayerStats.is_reliable

    # Confiance archétype minimum pour activer le Best-Response — 0.55,
    # valeur retenue après l'expérience phase 5 (0.75 testé, aucun impact
    # mesuré) et la Player DB phase 6 (confiance désormais calibrée sur le
    # volume réel de mains/adversaire, cf. best_response_engine.py pour le
    # détail complet).
    best_response_min_confidence: float = 0.55

    # ── Reproductibilité ──────────────────────────────────────────────────────
    # Seed du poker_engine.EHSCalculator (Dimension 1 — HS/PPot/NPot).
    # /!\ NE JAMAIS mettre 0 : d'après bindings.cpp, seed=0 signifie
    #     "graine aléatoire" côté C++ (typiquement random_device / horloge),
    #     ce qui rend CHAQUE calcul d'EHS non-reproductible d'un run à l'autre,
    #     même avec une seed fixée sur le simulateur (PokerTable) ou le
    #     MonteCarloEngine du Best-Response. C'était la cause des résultats
    #     différents entre deux exécutions de validate_phase5.py.
    ehs_seed: int = 42

    # ── Phase 6 — Player DB ─────────────────────────────────────────────────
    # Active la persistance des statistiques adverses multi-session
    # (core/player_db/). Si False (défaut historique) ou si le module
    # est absent -> comportement phase 5 exact, aucune régression.
    # /!\ Activé par défaut ici car c'est l'objet de la phase 6 — repasser
    #     à False pour reproduire un run phase 5 à l'identique.
    use_player_db:  bool = True
    player_db_path: str  = "data/player_db.sqlite3"





# =============================================================================
# Normalisation GameState -> dict
# =============================================================================

def _normalize_game_state(game_state) -> dict:
    """
    Convertit un objet GameState (simulateur) ou un dict en dict standard.

    GameState expose :
        our_hand_str()  -> list[str]   ex: ['Ah', 'Kd']
        board_str()     -> list[str]   ex: ['7c', '2h', 'Js']
        our_player      -> Player      avec .stack, .position
        active_opponents -> list[Player]
        pot, to_call, street, big_blind, our_id, action_history
    """
    if isinstance(game_state, dict):
        return game_state

    # Cartes — méthodes natives de GameState
    hand  = game_state.our_hand_str()
    board = game_state.board_str()

    # Notre joueur
    our  = game_state.our_player
    stack = float(our.stack)

    # Position (enum → string)
    pos_raw  = our.position
    position = pos_raw.value if hasattr(pos_raw, 'value') else str(pos_raw)

    # Street (enum → string)
    street_raw = game_state.street
    street = street_raw.value if hasattr(street_raw, 'value') else str(street_raw).lower()

    # Adversaires actifs — 'id' requis par action_history.py
    players_dict = [
        {
            'id':        p.player_id,   # clé attendue par action_history.py
            'player_id': p.player_id,   # clé attendue par _get_villain_id
            'stack':     float(p.stack),
            'status':    p.status.value if hasattr(p.status, 'value') else str(p.status),
            'position':  p.position.value if hasattr(p.position, 'value') else str(p.position),
        }
        for p in game_state.active_opponents
    ]

    # Action history
    action_history = []
    for act in game_state.action_history:
        st  = act.street
        at  = act.action_type
        action_history.append({
            'street': st.value  if hasattr(st, 'value')  else str(st),
            'player': act.player_id,
            'action': at.value  if hasattr(at, 'value')  else str(at),
            'amount': float(act.amount),
        })

    return {
        'hand':           hand,
        'board':          board,
        'pot':            float(game_state.pot),
        'to_call':        float(game_state.to_call),
        'stack':          stack,
        'position':       position,
        'street':         street,
        'players':        players_dict,
        'player_id':      game_state.our_id,
        'action_history': action_history,
        'big_blind':      float(game_state.big_blind),
    }


# =============================================================================
# EHSBot v4
# =============================================================================

class EHSBot:
    """
    Bot décisionnel intégrant les 4 dimensions de bucketing (v2)
    + Range Estimator bayésien (Dimension 5, phase 4)
    + Best-Response Engine exploitatif (phase 5).

    Pipeline de décision :
      game_state
        → [EHSCalculator C++]    → ehs_result       (Dim 1)
        → [ActionHistory]        → bucket + modifier (Dim 2)
        → [BoardTexture]         → texture_modifier  (Dim 3)
        → [SPR]                  → spr_info          (Dim 4)
        → [RangeEstimator]       → arch + sizing_adj (Dim 5, optionnelle)
        → [decide()]             → Action
    """

    def __init__(self, config: Optional[EHSBotConfig] = None):
        self.config = config or EHSBotConfig()
        self.name   = "EHSBot_v4"

        if _POKER_ENGINE_AVAILABLE:
            # seed=0 == "aléatoire" côté C++ (voir bindings.cpp) -> NE JAMAIS
            # utiliser 0 ici si on veut des runs reproductibles.
            self._ehs_calc = poker_engine.EHSCalculator(
                seed=self.config.ehs_seed, use_openmp=True
            )
            logger.info(
                "EHSBot v4 initialisé avec poker_engine C++ (ehs_seed=%d)",
                self.config.ehs_seed,
            )
        else:
            self._ehs_calc = None
            logger.warning("EHSBot v4 : poker_engine non disponible, stub actif")

        # Player DB — persistance multi-session (phase 6)
        self._player_db = None
        if self.config.use_player_db:
            try:
                from core.player_db.player_db import PlayerDB
                self._player_db = PlayerDB(self.config.player_db_path)
                logger.info(
                    "EHSBot v4 : Player DB active (%s)", self.config.player_db_path
                )
            except Exception as e:
                logger.warning(
                    "EHSBot v4 : core.player_db indisponible — Player DB "
                    "désactivée, comportement phase 5. (%s)", e
                )
                self._player_db = None

        # Range Estimator — instancié seulement si disponible et activé
        self._estimator  = None
        self._hand_count = 0

        if self.config.use_range_estimator:
            if _RANGE_ESTIMATOR_AVAILABLE:
                if self._player_db is not None:
                    from core.player_db.db_range_estimator import DBAwareRangeEstimator
                    self._estimator = DBAwareRangeEstimator(
                        player_db=self._player_db,
                        # Placeholder jetable et unique — jamais la chaîne
                        # littérale "unknown" (cf. full_reset()) : deux
                        # instances d'EHSBot pointant sur la même Player DB
                        # ne doivent jamais partager silencieusement une
                        # même entrée avant le premier set_opponent()/
                        # full_reset(opponent_id=...) explicite.
                        player_id=f"__anon_{uuid.uuid4().hex[:10]}",
                        smoothing=self.config.range_estimator_smoothing,
                        min_hands_for_archetype=self.config.min_hands_for_range_estimator,
                    )
                else:
                    self._estimator = RangeEstimator(
                        smoothing=self.config.range_estimator_smoothing,
                        min_hands_for_archetype=self.config.min_hands_for_range_estimator,
                    )
            else:
                logger.warning(
                    "EHSBot v4 : core.range_estimator non trouvé — "
                    "Dimension 5 désactivée, comportement v2."
                )

        # Best-Response Engine — instancié seulement si disponible et activé
        self._br_engine = None

        if self.config.use_best_response:
            if _BEST_RESPONSE_AVAILABLE:
                br_config = BestResponseConfig(
                    sizings=self.config.best_response_sizings,
                    n_mc_sims_ev=self.config.best_response_mc_sims,
                    n_sims_frequency=self.config.best_response_freq_sims,
                    min_hands_for_best_response=self.config.best_response_min_hands,
                    min_archetype_confidence=self.config.best_response_min_confidence,
                )

                # Instancier le MonteCarloEngine C++ pour le Best-Response
                _mc_engine = None
                if _POKER_ENGINE_AVAILABLE:
                    try:
                        _mc_engine = poker_engine.MonteCarloEngine(seed=42)
                    except Exception as e:
                        logger.warning("EHSBot v4 : MonteCarloEngine indisponible — %s", e)

                self._br_engine = BestResponseEngine(
                    mc_engine=_mc_engine,
                    config=br_config,
                    player_db=self._player_db,
                )
                # Partager le calculateur EHS (évite une instance dupliquée)
                self._br_engine.set_ehs_calculator(self._ehs_calc)

                logger.info(
                    "EHSBot v4 : Best-Response Engine actif "
                    "(sizings=%s, mc_sims=%d, freq_sims=%d, min_hands=%d)",
                    self.config.best_response_sizings,
                    self.config.best_response_mc_sims,
                    self.config.best_response_freq_sims,
                    self.config.best_response_min_hands,
                )
            else:
                logger.warning(
                    "EHSBot v4 : core.best_response non trouvé — "
                    "Best-Response désactivé, comportement v3."
                )

    # =========================================================================
    # Point d'entrée principal
    # =========================================================================

    def _effective_hand_count(self) -> int:
        """
        Phase 6 — nombre de mains à utiliser pour les seuils d'activation
        (Dimension 5 et Best-Response).

        BUG évité ici : sans ce correctif, self._hand_count (compteur
        LOCAL au match, remis à 0 par full_reset()) resterait la seule
        valeur utilisée pour ces seuils — un adversaire recroisé après
        une pause, pourtant déjà au Palier 2 en DB (30+ mains cumulées
        sur plusieurs sessions), redémarrerait à tort en fallback v3
        jusqu'à rejouer min_hands_for_best_response mains dans CE match
        précis. Ça viderait la persistance multi-session de sa valeur.

        max(local, DB) garantit qu'on ne fait jamais PIRE que le
        comportement phase 5 (si la Player DB n'est pas encore alimentée
        par hand_recorder.record_hand() côté harness, DB reste à 0 et le
        compteur local prend le relais normalement) tout en profitant de
        l'historique dès qu'il existe.
        """
        if self._player_db is not None and self._estimator is not None:
            profile = getattr(self._estimator, 'last_profile', None)
            if profile is not None:
                return max(self._hand_count, profile.hands_seen)
        return self._hand_count

    def decide(self, game_state: dict) -> Action:
        """
        Prend une décision pour l'état de jeu courant.

        Args:
            game_state : dict au format normalisé (identique v2, section 4.4)
        Returns:
            Action
        """
        # ── Extraction des champs (identique v2) ──────────────────────────────
        game_state = _normalize_game_state(game_state)
        hand     = game_state.get('hand', [])
        board    = game_state.get('board', [])
        pot      = float(game_state.get('pot', 1))
        to_call  = float(game_state.get('to_call', 0))
        stack    = float(game_state.get('stack', 0))
        position = game_state.get('position', 'BTN')
        street   = game_state.get('street', 'unknown')
        players  = game_state.get('players', [])
        our_id   = game_state.get('player_id', 0)   # clé v2 conservée
        n_opponents = max(len(players), self.config.default_n_opponents)

        # ── Dimension 1 : EHS ─────────────────────────────────────────────────
        ehs_result = self._compute_ehs(hand, board, n_opponents)
        ehs        = ehs_result['EHS']

        # ── Dimension 2 : Action History Bucket ───────────────────────────────
        bucket_modifier  = 0.0
        aggregate_bucket = ActionBucket.NO_ACTION

        if self.config.use_action_history:
            history = game_state.get('action_history', [])
            villain_buckets = get_villain_bucket_from_state(
                {'action_history': history, 'players': players},
                our_id=our_id,
            )
            aggregate_bucket = get_aggregate_bucket(villain_buckets)
            raw_modifier     = BUCKET_EHS_MODIFIER[aggregate_bucket]
            clamp            = self.config.bucket_modifier_clamp
            bucket_modifier  = max(-clamp, min(clamp, raw_modifier))

            logger.debug(
                "[%s] Bucket=%s modifier=%+.2f",
                self.name, aggregate_bucket.value, bucket_modifier
            )

        # ── Dimension 3 : Board Texture ───────────────────────────────────────
        texture          = None
        texture_modifier = 1.0
        if board and self.config.use_board_texture:
            texture          = classify_board(board)
            texture_modifier = texture.sizing_modifier()

        # ── Dimension 4 : SPR ─────────────────────────────────────────────────
        spr_info = get_spr_info_from_game_state(game_state)

        logger.debug(
            "[%s] %s %s | EHS=%.3f bucket=%s SPR=%.1f[%s] board=%s",
            self.name, hand, board, ehs,
            aggregate_bucket.value, spr_info.spr, spr_info.category.value,
            texture.value if texture else "N/A"
        )

        # ── Dimension 5 : Range Estimator ─────────────────────────────────────
        range_arch_modifier = 0.0
        range_siz_modifier  = 1.0

        if self._estimator is not None:
            # Déduire le villain_id depuis our_id et la liste des joueurs
            villain_id = self._get_villain_id(players, our_id)
            if villain_id is not None:
                # Phase 6 : tient _villain_position à jour à chaque main —
                # affine le prior de population du Palier 0 (no-op
                # silencieux pour un RangeEstimator classique, hasattr).
                # Effet différé d'une main : la position lue ici sert au
                # prior de la PROCHAINE main (reset()/new_hand()), pas à
                # la main en cours dont le prior est déjà construit.
                if hasattr(self._estimator, 'set_villain_position'):
                    villain_position = next(
                        (p.get('position') for p in players
                         if p.get('id', p.get('player_id', -1)) == villain_id),
                        None,
                    )
                    if villain_position:
                        self._estimator.set_villain_position(villain_position)
                self._estimator.update_from_game_state(game_state, villain_id)

            if self._effective_hand_count() >= self.config.min_hands_for_range_estimator:
                arch_probs = self._estimator.get_archetype_probabilities()
                best_arch  = max(arch_probs, key=arch_probs.__getitem__)
                confidence = arch_probs[best_arch]

                if confidence >= self.config.archetype_confidence_threshold:
                    range_arch_modifier = self.config.ehs_threshold_by_archetype.get(
                        best_arch, 0.0
                    )
                    range_siz_modifier = 1.0 + self.config.sizing_modifier_by_archetype.get(
                        best_arch, 0.0
                    )
                    logger.debug(
                        "[%s] Dim5: arch=%s conf=%.2f ehs_adj=%+.2f siz_adj=%+.1f%%",
                        self.name, best_arch, confidence,
                        range_arch_modifier, (range_siz_modifier - 1) * 100
                    )

        # ── Best-Response Engine (phase 5) ────────────────────────────────────
        # Actif si : use_best_response=True, core.best_response disponible,
        # Range Estimator a assez de données, confiance archétype suffisante.
        # Sinon : fallback transparent sur _make_decision() v3 ci-dessous.
        if self._br_engine is not None and self._estimator is not None:
            decision, action = self._br_engine.decide(
                game_state=game_state,
                estimator=self._estimator,
                ehs_result=ehs_result,
                spr_info=spr_info,
                texture_modifier=texture_modifier * range_siz_modifier,
                bucket_modifier=bucket_modifier + range_arch_modifier,
                hand_count=self._effective_hand_count(),
            )
            logger.info(
                "[%s] %s %s | street=%s EHS=%.3f SPR=%.1f → %s",
                self.name, hand, board, street, ehs, spr_info.spr, action
            )
            logger.debug("[%s] BR: %s", self.name, decision.summary())
            return action

        # ── Décision v3 (fallback) ────────────────────────────────────────────
        action = self._make_decision(
            ehs=ehs,
            ehs_result=ehs_result,
            bucket_modifier=bucket_modifier + range_arch_modifier,
            spr_info=spr_info,
            texture_modifier=texture_modifier * range_siz_modifier,
            pot=pot,
            to_call=to_call,
            stack=stack,
            position=position,
            street=street,
        )

        logger.info(
            "[%s] %s %s | street=%s EHS=%.3f bucket=%s SPR=%.1f → %s [v3]",
            self.name, hand, board, street, ehs,
            aggregate_bucket.value, spr_info.spr, action
        )

        return action

    # =========================================================================
    # Logique de décision v3 (conservée — utilisée comme fallback)
    # =========================================================================

    def _make_decision(
        self,
        ehs:              float,
        ehs_result:       dict,
        bucket_modifier:  float,
        spr_info:         SPRInfo,
        texture_modifier: float,
        pot:              float,
        to_call:          float,
        stack:            float,
        position:         str,
        street:           str,
    ) -> Action:
        high_threshold = self.config.ehs_high_threshold + bucket_modifier
        low_threshold  = self.config.ehs_low_threshold  + bucket_modifier

        # SPR très bas → commitment forcé
        if spr_info.force_commit:
            amount = min(stack, pot * 3)
            return Action(action_type='allin', amount=amount,
                          sizing_pct=amount/pot if pot else 0)

        # EHS élevé → bet/raise value
        if ehs > high_threshold:
            sizing     = self._compute_value_sizing(ehs, high_threshold,
                                                    texture_modifier, spr_info)
            bet_amount = min(sizing * pot, stack)
            if to_call > 0:
                raise_amount = min(to_call * 2.5 + pot * sizing, stack)
                return Action(action_type='raise', amount=raise_amount,
                              sizing_pct=sizing)
            return Action(action_type='bet', amount=bet_amount, sizing_pct=sizing)

        # EHS médian → protection ou call
        if ehs >= low_threshold:
            en_position = self._is_in_position(position)
            if en_position and to_call == 0:
                sizing     = self.config.sizing_protection \
                             * texture_modifier \
                             * spr_info.sizing_modifier()
                sizing     = min(sizing, self.config.sizing_max)
                bet_amount = min(sizing * pot, stack)
                return Action(action_type='bet', amount=bet_amount,
                              sizing_pct=sizing)
            if to_call > 0:
                if self._is_call_profitable(to_call, pot, ehs):
                    return Action(action_type='call',
                                  amount=min(to_call, stack))
                return Action(action_type='fold')
            return Action(action_type='check')

        # EHS faible → check/fold, sauf draw fort
        ppot = ehs_result.get('PPot', 0.0)
        if ppot >= 0.25 and to_call > 0:
            if self._is_call_profitable(to_call, pot, ehs + ppot * 0.5):
                logger.debug("[%s] Call draw fort PPot=%.2f", self.name, ppot)
                return Action(action_type='call', amount=min(to_call, stack))

        if to_call > 0:
            return Action(action_type='fold')
        return Action(action_type='check')

    # =========================================================================
    # Helpers
    # =========================================================================

    def _compute_ehs(
        self,
        hand:        List[str],
        board:       List[str],
        n_opponents: int,
    ) -> dict:
        """Calcule l'EHS via poker_engine C++, ou stub si indisponible."""
        if self._ehs_calc is not None:
            result = self._ehs_calc.calculate_multiway(
                hand, board, n_opponents, self.config.n_sims
            )
            return {
                'EHS':  result.EHS,
                'HS':   result.HS,
                'PPot': result.PPot,
                'NPot': result.NPot,
                'elapsed_ms': result.elapsed_ms,
            }
        import random
        hs   = random.uniform(0.3, 0.8)
        ppot = random.uniform(0.0, 0.3)
        npot = random.uniform(0.0, 0.2)
        ehs  = hs * (1 - npot) + (1 - hs) * ppot
        return {'EHS': ehs, 'HS': hs, 'PPot': ppot, 'NPot': npot,
                'elapsed_ms': 0.0}

    def _compute_value_sizing(
        self,
        ehs:              float,
        high_threshold:   float,
        texture_modifier: float,
        spr_info:         SPRInfo,
    ) -> float:
        sizing = 0.33 + (ehs - high_threshold) * self.config.sizing_scale_factor
        if self.config.use_board_texture:
            sizing *= texture_modifier
        if self.config.use_spr_modifier:
            sizing *= spr_info.sizing_modifier()
        return max(0.33, min(sizing, self.config.sizing_max))

    def _is_in_position(self, position: str) -> bool:
        return position.upper() in {'BTN', 'CO', 'MP'}

    def _is_call_profitable(
        self,
        to_call: float,
        pot:     float,
        equity:  float,
    ) -> bool:
        if pot + to_call <= 0:
            return False
        return equity > to_call / (pot + to_call)

    def _get_villain_id(self, players: list, our_id: int) -> Optional[int]:
        """Retourne l'id du premier adversaire actif."""
        for p in players:
            pid = p.get('id', p.get('player_id', -1))
            if pid != our_id and p.get('status', 'active') == 'active':
                return pid
        return None

    # =========================================================================
    # Interface simulateur
    # =========================================================================

    def get_action(self, game_state) -> 'Action':
        """
        Interface compatible simulateur — accepte dict ou objet GameState.

        Retourne un objet Action du simulateur (ActionType enum) quand
        game_state est un GameState, sinon retourne l'Action interne.
        """
        our_action = self.decide(game_state)

        # Si game_state est un dict (tests), retourner tel quel
        if isinstance(game_state, dict):
            return our_action

        # Convertir vers Action du simulateur (ActionType enum)
        try:
            from core.game_state import Action as SimAction, ActionType, Street
            _MAP = {
                'fold':  ActionType.FOLD,
                'check': ActionType.CHECK,
                'call':  ActionType.CALL,
                'bet':   ActionType.RAISE,
                'raise': ActionType.RAISE,
                'allin': ActionType.ALLIN,
            }
            at        = _MAP.get(our_action.action_type, ActionType.CHECK)
            amount    = int(our_action.amount)
            street    = game_state.street   # déjà un enum Street
            return SimAction(game_state.our_id, at, amount, street)
        except Exception as e:
            logger.warning("get_action : conversion SimAction échouée (%s)", e)
            return our_action

    def set_opponent(self, player_id: str, position: Optional[str] = None) -> None:
        """
        Phase 6 — identifie l'adversaire courant pour la Player DB.

        Sans Player DB active (use_player_db=False ou module indisponible),
        cette méthode est un no-op : le RangeEstimator classique n'a pas de
        notion d'identité persistante, exactement comme en phase 5.

        À appeler avant full_reset() lors d'un changement de match/table —
        voir full_reset(opponent_id=...) qui l'appelle automatiquement.
        """
        if self._player_db is None or self._estimator is None:
            return
        if hasattr(self._estimator, 'set_player'):
            self._estimator.set_player(player_id)
        if position and hasattr(self._estimator, 'set_villain_position'):
            self._estimator.set_villain_position(position)

    def reset(self):
        """
        Réinitialise entre les mains.
        v2 : stateless — reset() ne faisait rien.
        v3 : incrémente le compteur et notifie le Range Estimator.
        """
        self._hand_count += 1
        if self._estimator is not None:
            self._estimator.new_hand()

    def full_reset(self, opponent_id: Optional[str] = None):
        """
        Reset complet entre deux matches différents.
        Remet le hand_count à 0 et réinitialise le Range Estimator.
        Différent de reset() qui incrémente le compteur entre les mains.

        Args:
            opponent_id : Phase 6, optionnel. Si fourni et qu'une Player DB
                          est active, bascule l'estimateur sur cet
                          adversaire (cf. set_opponent()) AVANT le reset —
                          le profil DB de ce joueur (paliers 0/1/2) est
                          alors utilisé dès la main suivante.

                          Si None (défaut) ET Player DB active : une
                          identité jetable et unique est générée pour ce
                          match (__anon_xxxxxxxxxx). C'est un choix de
                          sécurité DÉLIBÉRÉ, pas juste "aucune identité" :
                          un appelant qui ignore la notion d'opponent_id
                          (ex: validate_phase5.py non modifié, qui appelle
                          full_reset() sans argument entre CHAQUE match
                          contre un adversaire DIFFÉRENT) ne doit jamais
                          voir les stats de deux adversaires distincts se
                          mélanger sous une même entrée DB implicite, et ne
                          doit jamais non plus continuer à suivre
                          silencieusement l'adversaire du match précédent.
                          Chaque appel sans opponent_id repart donc au
                          Palier 0, exactement comme un full_reset() phase
                          5 (aucune mémoire conservée) — la persistance
                          n'entre en jeu QUE quand l'appelant fournit
                          explicitement un opponent_id stable.
        """
        self._hand_count = 0
        if self._player_db is not None:
            if opponent_id is not None:
                self.set_opponent(opponent_id)
            else:
                self.set_opponent(f"__anon_{uuid.uuid4().hex[:10]}")
        if self._estimator is not None:
            # /!\ estimator.reset() est APPELÉ ENTRE CHAQUE MAIN et conserve
            # volontairement les VillainStats (VPIP/PFR/AF) pour construire
            # un lecture de l'adversaire au fil du match. Ici on change
            # d'adversaire : il faut repartir de zéro, donc full_reset()
            # et non reset().
            self._estimator.full_reset()

    def close(self) -> None:
        """Phase 6 — ferme proprement la connexion Player DB si active."""
        if self._player_db is not None:
            self._player_db.close()

    def __repr__(self):
        re_status = "active" if self._estimator else "off"
        br_status = "active" if self._br_engine else "off"
        return (
            f"EHSBot_v4(ehs_high={self.config.ehs_high_threshold}, "
            f"ehs_low={self.config.ehs_low_threshold}, "
            f"n_sims={self.config.n_sims}, "
            f"range_estimator={re_status}, "
            f"best_response={br_status}, "
            f"hands={self._hand_count})"
        )
