"""
bluff_layer.py — Couche Bluff Minimale
Phase 5 — P3 — Bot Poker Académique

Calcule la fréquence de bluff minimale nécessaire pour que le bot ne soit
pas exploitable sur sa fréquence de bet, et sélectionne les mains candidates
au bluff (semi-bluffs et bluffs purs) depuis le contexte de jeu courant.

Principe fondamental :
    Si le bot ne bet que ses value hands, l'adversaire peut folder
    systématiquement face à tous nos bets et nous exploiter.
    Pour rendre notre range de bet non-exploitable sur ce point précis,
    on doit bluffer avec une fréquence minimale f* telle que l'adversaire
    soit indifférent entre call et fold face à notre bet.

    Calcul de f* (théorie des jeux, pot odds) :
        EV(call) = EV(fold) ⟹ équilibre
        equity_vs_notre_range × (pot + call) - call = 0
        equity_vs_notre_range = call / (pot + call) = pot_odds

        Si notre range de bet contient une fraction f de bluffs :
            equity_adverse_vs_notre_range = f × 1.0 + (1-f) × (1 - notre_equity)
            ≈ f + (1-f) × (1 - notre_equity)

        À l'équilibre : f* = pot_odds / (1 + pot_odds - notre_equity)
        Approximation courante : f* ≈ pot_odds (si notre_equity ≈ 1 vs bluffs)

    En pratique :
        bet_50%_pot → pot_odds = 0.50/(1+0.50) = 33% → bluff_min = 33%
        bet_75%_pot → pot_odds = 0.75/(1+0.75) = 43% → bluff_min = 43%
        bet_100%_pot → pot_odds = 1.0/(1+1.0)  = 50% → bluff_min = 50%

Candidats au bluff (par ordre de priorité) :
    1. Semi-bluffs (PPot élevé) : draws forts qui peuvent gagner si callés
       → EV(bluff) = p_fold×pot + p_call×(ppot×new_pot - bet)
       → Positif si PPot > bet/(pot+2×bet)
    2. Bluffs à high card : mains sans showdown value mais avec backdoor draws
    3. Bluffs purs : mains sans aucune valeur (si les deux catégories ci-dessus
       sont insuffisantes pour atteindre f*)

    En pratique avec notre architecture :
    - PPot vient directement de ehs_result['PPot'] (Dim 1)
    - Le semi-bluff est validé si PPot ≥ seuil et EV_bluff > 0
    - On n'ajoute pas de bluffs purs si PPot < seuil (trop risqué)

Relation avec P0 (BestResponseEngine._apply_bluff_layer) :
    L'ancien stub dans best_response_engine.py est remplacé par un appel
    à BluffLayer.evaluate(). La décision finale reste dans BestResponseEngine.

Interface publique :
    layer = BluffLayer(config)
    result = layer.evaluate(
        best_ev_result,   # EVResult de la meilleure action de bet
        ehs,              # notre EHS (Dim 1)
        ppot,             # notre PPot (Dim 1)
        pot,              # taille du pot
        street,           # street courante
        archetype,        # archétype estimé de l'adversaire
    )
    # result.should_bluff      : bool — faut-il bluffer ?
    # result.bluff_type        : 'semi_bluff' | 'pure_bluff' | 'none'
    # result.bluff_freq        : fréquence de bluff calculée
    # result.min_bluff_freq    : fréquence minimale théorique
    # result.ev_bluff          : EV estimée du bluff
    # result.action_modifier   : modificateur d'action (bet ou check)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from core.best_response.ev_calculator import EVResult

logger = logging.getLogger(__name__)

# Fréquence de bluff minimale en dessous de laquelle on ne bluff pas
# (évite les situations où le calcul donne une fréquence négligeable)
_MIN_ACTIONABLE_BLUFF_FREQ = 0.05

# PPot minimum pour un semi-bluff valide
_MIN_PPOT_SEMI_BLUFF = 0.18

# PPot minimum pour un bluff pur (plus strict — on veut au moins un backdoor)
_MIN_PPOT_PURE_BLUFF = 0.08

# EHS maximum pour qu'une main soit considérée comme bluff (pas de value bet)
# Au-dessus de ce seuil, la main est une value hand, pas un bluff
_MAX_EHS_FOR_BLUFF = 0.55

# Fraction du pot minimum pour que la fold equity justifie un bluff
_MIN_FOLD_EQUITY_FOR_BLUFF = 0.30


# =============================================================================
# Résultat de l'évaluation bluff
# =============================================================================

@dataclass
class BluffEvalResult:
    """
    Résultat de l'évaluation de la couche bluff.

    Attributes:
        should_bluff     : True si on doit ajouter un bluff à notre range de bet
        bluff_type       : 'semi_bluff' | 'pure_bluff' | 'none'
        bluff_freq       : fréquence de bluff estimée de notre range courante
        min_bluff_freq   : fréquence minimale théorique (pot odds)
        ev_bluff         : EV estimée du bluff (en chips)
        deficit          : écart entre fréquence courante et minimale
        reason           : explication courte de la décision
    """
    should_bluff:   bool
    bluff_type:     str    # 'semi_bluff' | 'pure_bluff' | 'none'
    bluff_freq:     float  # fréquence actuelle estimée
    min_bluff_freq: float  # fréquence minimale théorique
    ev_bluff:       float  # EV du bluff estimée
    deficit:        float  # min_bluff_freq - bluff_freq (> 0 si sous-bluff)
    reason:         str

    def summary(self) -> str:
        tag = f"[{self.bluff_type.upper()}]" if self.should_bluff else "[NO BLUFF]"
        return (
            f"{tag} freq={self.bluff_freq:.1%} min={self.min_bluff_freq:.1%} "
            f"deficit={self.deficit:+.1%} EV={self.ev_bluff:+.2f} — {self.reason}"
        )


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BluffLayerConfig:
    """Paramètres de la couche bluff."""

    # Seuils PPot pour les candidats bluff
    min_ppot_semi_bluff: float = _MIN_PPOT_SEMI_BLUFF
    min_ppot_pure_bluff: float = _MIN_PPOT_PURE_BLUFF

    # EHS maximum pour qu'une main soit un bluff (pas une value hand)
    max_ehs_for_bluff: float = _MAX_EHS_FOR_BLUFF

    # Fold equity minimum pour justifier un bluff
    min_fold_equity: float = _MIN_FOLD_EQUITY_FOR_BLUFF

    # Fréquence de bluff minimale actionnable
    min_actionable_freq: float = _MIN_ACTIONABLE_BLUFF_FREQ

    # Activer les bluffs purs (désactiver pour plus de prudence)
    allow_pure_bluff: bool = True

    # Sur la river, pas de semi-bluff possible (plus de cartes à venir)
    # → seulement bluffs purs si activés
    river_semi_bluff: bool = False


# =============================================================================
# BluffLayer
# =============================================================================

class BluffLayer:
    """
    Évalue si notre range de bet nécessite l'ajout d'un bluff et sélectionne
    le type de bluff approprié.

    Logique de décision :
        1. Calculer la fréquence de bluff minimale f* depuis le sizing du bet
        2. Estimer la fréquence de bluff courante de notre range
           (approximation : 0 si EHS > max_ehs_for_bluff, PPot/1.0 sinon)
        3. Si déficit > min_actionable_freq :
           a. Vérifier si semi-bluff possible (PPot ≥ seuil, EV > 0)
           b. Sinon vérifier si bluff pur possible (fold equity suffisante)
           c. Sinon : ne pas forcer le bluff (mieux vaut check)
        4. Retourner BluffEvalResult avec la décision et les métriques

    Note sur la fréquence courante :
        On n'a pas accès à toute notre range de main — on joue une main
        spécifique. L'approximation est : si EHS > max_ehs_for_bluff, on
        est une value hand (fréquence bluff = 0). Sinon, on est un candidat
        bluff potentiel (fréquence = PPot normalisée).
        Cette approximation est suffisante pour l'usage en temps réel.
    """

    def __init__(self, config: Optional[BluffLayerConfig] = None):
        self._config = config or BluffLayerConfig()

    def evaluate(
        self,
        best_ev:   EVResult,
        ehs:       float,
        ppot:      float,
        pot:       float,
        street:    str,
        archetype: str,
    ) -> BluffEvalResult:
        """
        Évalue si un bluff doit être ajouté à la range de bet courante.

        Args:
            best_ev   : EVResult de la meilleure action (bet/raise)
            ehs       : notre EHS (Dim 1, depuis ehs_result)
            ppot      : notre PPot — pot. potential (Dim 1)
            pot       : taille du pot avant notre action
            street    : street courante ('flop'|'turn'|'river')
            archetype : archétype estimé de l'adversaire

        Returns:
            BluffEvalResult avec la décision de bluff.
        """
        bet_amount = best_ev.amount
        p_fold     = best_ev.p_fold

        # ── 1. Fréquence de bluff minimale théorique ──────────────────────────
        min_bluff_freq = self._compute_min_bluff_freq(bet_amount, pot)

        if min_bluff_freq < self._config.min_actionable_freq:
            return BluffEvalResult(
                should_bluff=False, bluff_type='none',
                bluff_freq=0.0, min_bluff_freq=min_bluff_freq,
                ev_bluff=0.0, deficit=0.0,
                reason=f"Fréquence minimale {min_bluff_freq:.1%} < seuil actionnable",
            )

        # ── 2. Vérifier si on est une value hand ──────────────────────────────
        if ehs > self._config.max_ehs_for_bluff:
            # On est une value hand — pas besoin d'évaluer le bluff
            # (le bet est déjà justifié par la valeur)
            return BluffEvalResult(
                should_bluff=False, bluff_type='none',
                bluff_freq=1.0 - ehs,  # approximation : notre range contient (1-EHS) de bluffs
                min_bluff_freq=min_bluff_freq,
                ev_bluff=0.0,
                deficit=max(0.0, min_bluff_freq - (1.0 - ehs)),
                reason=f"Value hand (EHS={ehs:.2f} > {self._config.max_ehs_for_bluff:.2f})",
            )

        # ── 3. Estimation de la fréquence de bluff courante ───────────────────
        # Approximation : fraction de notre range qui est un bluff
        # On utilise PPot comme proxy (draw fort = semi-bluff naturel)
        current_bluff_freq = min(ppot, 0.80)  # cappé à 80%
        deficit = min_bluff_freq - current_bluff_freq

        # ── 4. Fold equity suffisante ? ───────────────────────────────────────
        if p_fold < self._config.min_fold_equity:
            return BluffEvalResult(
                should_bluff=False, bluff_type='none',
                bluff_freq=current_bluff_freq, min_bluff_freq=min_bluff_freq,
                ev_bluff=0.0, deficit=deficit,
                reason=f"Fold equity insuffisante ({p_fold:.1%} < {self._config.min_fold_equity:.0%})",
            )

        # ── 5. Semi-bluff ? ───────────────────────────────────────────────────
        if street != 'river' or self._config.river_semi_bluff:
            if ppot >= self._config.min_ppot_semi_bluff:
                ev_semi_bluff = self._compute_ev_bluff(
                    bet_amount=bet_amount,
                    pot=pot,
                    p_fold=p_fold,
                    equity_if_called=ppot,  # on réalise PPot si callé
                )
                if ev_semi_bluff > 0:
                    return BluffEvalResult(
                        should_bluff=True,
                        bluff_type='semi_bluff',
                        bluff_freq=current_bluff_freq,
                        min_bluff_freq=min_bluff_freq,
                        ev_bluff=ev_semi_bluff,
                        deficit=deficit,
                        reason=(
                            f"Semi-bluff valide : PPot={ppot:.2f} "
                            f"EV={ev_semi_bluff:+.2f} "
                            f"fold_eq={p_fold:.1%}"
                        ),
                    )

        # ── 6. Bluff pur ? ────────────────────────────────────────────────────
        if self._config.allow_pure_bluff:
            if ppot >= self._config.min_ppot_pure_bluff:
                ev_pure_bluff = self._compute_ev_bluff(
                    bet_amount=bet_amount,
                    pot=pot,
                    p_fold=p_fold,
                    equity_if_called=0.0,  # on perd si callé
                )
                if ev_pure_bluff > 0:
                    return BluffEvalResult(
                        should_bluff=True,
                        bluff_type='pure_bluff',
                        bluff_freq=current_bluff_freq,
                        min_bluff_freq=min_bluff_freq,
                        ev_bluff=ev_pure_bluff,
                        deficit=deficit,
                        reason=(
                            f"Bluff pur valide : fold_eq={p_fold:.1%} "
                            f"EV={ev_pure_bluff:+.2f} "
                            f"PPot={ppot:.2f}"
                        ),
                    )

        # ── 7. Pas de bluff justifié ──────────────────────────────────────────
        return BluffEvalResult(
            should_bluff=False,
            bluff_type='none',
            bluff_freq=current_bluff_freq,
            min_bluff_freq=min_bluff_freq,
            ev_bluff=0.0,
            deficit=deficit,
            reason=(
                f"Bluff non justifié : PPot={ppot:.2f} "
                f"fold_eq={p_fold:.1%} street={street}"
            ),
        )

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _compute_min_bluff_freq(bet_amount: float, pot: float) -> float:
        """
        Calcule la fréquence de bluff minimale depuis les pot odds offerts.

        f* = pot_odds = bet / (pot + bet)

        C'est la fréquence à laquelle l'adversaire doit être indifférent
        entre call et fold — si on bluff moins souvent, il peut folder
        systématiquement et exploiter notre range.
        """
        if pot <= 0 or bet_amount <= 0:
            return 0.0
        total = pot + bet_amount
        return bet_amount / total

    @staticmethod
    def _compute_ev_bluff(
        bet_amount:      float,
        pot:             float,
        p_fold:          float,
        equity_if_called: float,
    ) -> float:
        """
        Calcule l'EV d'un bluff (ou semi-bluff).

        EV(bluff) = p_fold × pot
                  + (1 - p_fold) × (equity_if_called × (pot + 2×bet) - bet)

        Pour un bluff pur : equity_if_called = 0
            EV(bluff_pur) = p_fold × pot - (1 - p_fold) × bet

        Pour un semi-bluff : equity_if_called = PPot
            EV(semi_bluff) = p_fold × pot
                           + (1-p_fold) × (PPot × (pot+2×bet) - bet)
        """
        p_call = 1.0 - p_fold
        ev = (p_fold * pot
              + p_call * (equity_if_called * (pot + 2 * bet_amount) - bet_amount))
        return ev

    def __repr__(self) -> str:
        return (
            f"BluffLayer("
            f"min_ppot_semi={self._config.min_ppot_semi_bluff:.2f}, "
            f"min_ppot_pure={self._config.min_ppot_pure_bluff:.2f}, "
            f"allow_pure={self._config.allow_pure_bluff})"
        )
