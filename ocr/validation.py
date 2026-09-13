"""
validation.py — Sanity checks et lissage temporel des lectures de table
Phase 7 — Bot Poker Académique

Rôle : dernier filtre avant que state_builder.py n'accepte une TableRead
et ne l'utilise pour mettre à jour le GameState. Mieux vaut une frame
ignorée qu'un GameState corrompu qui fait raisonner tout le pipeline
aval (EHS, Best-Response, Player DB) sur des données fausses.

Deux familles d'outils :
  1. validate_table_read() et consorts — vérifications ponctuelles sur
     UNE lecture (cohérence interne : cartes dupliquées, montants
     négatifs, confidence trop basse).
  2. FrameStabilizer — lissage sur PLUSIEURS lectures successives, pour
     éviter qu'une valeur captée en pleine animation (mise qui glisse
     vers le pot, stack qui décrémente visuellement) ne soit interprétée
     comme la valeur finale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Tuple

from .vision_types import CardRead, TableRead


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok

    @staticmethod
    def combine(*results: "ValidationResult") -> "ValidationResult":
        errors: List[str] = []
        for r in results:
            errors.extend(r.errors)
        return ValidationResult(ok=not errors, errors=errors)


# =============================================================================
# Vérifications ponctuelles
# =============================================================================

def validate_cards_unique(read: TableRead) -> ValidationResult:
    """
    Une même carte ne peut jamais apparaître deux fois sur la table à la
    fois (board + mains visibles). Un doublon signale presque toujours
    une erreur de reconnaissance (template matching qui a confondu deux
    cartes similaires) plutôt qu'une vraie incohérence de jeu.
    """
    seen: Dict[str, str] = {}
    errors: List[str] = []

    def _check(card: CardRead, origin: str) -> None:
        s = card.as_str()
        if s is None:
            return
        if s in seen:
            errors.append(f"Carte {s} vue à la fois en {seen[s]} et en {origin}")
        else:
            seen[s] = origin

    for card in read.board_cards:
        _check(card, "board")

    for seat in read.seats:
        for i, card in enumerate(seat.hole_cards):
            _check(card, f"siège {seat.seat_index} carte {i}")

    return ValidationResult(ok=not errors, errors=errors)


def validate_stacks(read: TableRead) -> ValidationResult:
    """Stacks, mises et pot doivent être positifs ou nuls quand ils sont lus."""
    errors: List[str] = []

    if read.pot is not None and read.pot < 0:
        errors.append(f"pot négatif : {read.pot}")

    for seat in read.seats:
        if not seat.is_occupied:
            continue
        if seat.stack is not None and seat.stack < 0:
            errors.append(f"siège {seat.seat_index} : stack négatif ({seat.stack})")
        if seat.bet_this_street < 0:
            errors.append(
                f"siège {seat.seat_index} : bet_this_street négatif "
                f"({seat.bet_this_street})"
            )

    return ValidationResult(ok=not errors, errors=errors)


def validate_confidence(read: TableRead, min_confidence: float = 0.6) -> ValidationResult:
    """
    Signale les lectures dont la confiance est sous le seuil. Ne bloque
    rien à elle seule : c'est à l'appelant de décider s'il ignore la
    frame entière ou seulement le champ concerné.
    """
    errors: List[str] = []

    for seat in read.seats:
        if not seat.is_occupied:
            continue
        if seat.confidence < min_confidence:
            errors.append(
                f"siège {seat.seat_index} : confidence trop basse "
                f"({seat.confidence:.2f} < {min_confidence})"
            )
        for i, card in enumerate(seat.hole_cards):
            if card.is_visible() and card.confidence < min_confidence:
                errors.append(
                    f"siège {seat.seat_index} carte {i} : confidence trop "
                    f"basse ({card.confidence:.2f} < {min_confidence})"
                )

    for i, card in enumerate(read.board_cards):
        if card.is_visible() and card.confidence < min_confidence:
            errors.append(
                f"board carte {i} : confidence trop basse "
                f"({card.confidence:.2f} < {min_confidence})"
            )

    return ValidationResult(ok=not errors, errors=errors)


def validate_board_length(read: TableRead) -> ValidationResult:
    """Le board ne peut avoir que 0, 3, 4 ou 5 cartes (preflop/flop/turn/river)."""
    n = len(read.board_cards)
    if n in (0, 3, 4, 5):
        return ValidationResult(ok=True)
    return ValidationResult(ok=False, errors=[f"board de longueur invalide : {n}"])


def validate_table_read(
    read: TableRead,
    min_confidence: float = 0.6,
    check_confidence: bool = True,
) -> ValidationResult:
    """Point d'entrée principal : agrège toutes les vérifications ponctuelles."""
    checks = [
        validate_cards_unique(read),
        validate_stacks(read),
        validate_board_length(read),
    ]
    if check_confidence:
        checks.append(validate_confidence(read, min_confidence))
    return ValidationResult.combine(*checks)


# =============================================================================
# Lissage temporel
# =============================================================================

class FrameStabilizer:
    """
    Évite qu'une lecture bruitée sur une seule frame (animation de mise
    en cours, glitch de rendu, reflet) ne soit interprétée comme une
    vraie valeur.

    Une nouvelle valeur pour une clé donnée n'est confirmée qu'après
    `required_repeats` lectures IDENTIQUES et consécutives différentes
    de la valeur déjà confirmée ; jusque-là, `update()` continue de
    retourner la dernière valeur confirmée.

    La toute première valeur jamais vue pour une clé est confirmée
    immédiatement (rien à départager).
    """

    def __init__(self, required_repeats: int = 2) -> None:
        if required_repeats < 1:
            raise ValueError("required_repeats doit être >= 1")
        self.required_repeats = required_repeats
        self._confirmed: Dict[Hashable, Any] = {}
        self._pending: Dict[Hashable, Tuple[Any, int]] = {}

    def update(self, key: Hashable, value: Optional[Any]) -> Tuple[Optional[Any], bool]:
        """
        Args:
            key   : identifiant du champ suivi (ex: ("stack", seat_index)).
            value : nouvelle lecture brute, ou None si non lisible ce tour.

        Returns:
            (valeur_confirmée, vient_de_changer_ce_tour)
        """
        if value is None:
            return self._confirmed.get(key), False

        if key not in self._confirmed:
            self._confirmed[key] = value
            self._pending.pop(key, None)
            return value, True

        if value == self._confirmed[key]:
            self._pending.pop(key, None)
            return value, False

        cand_value, cand_count = self._pending.get(key, (None, 0))
        cand_count = cand_count + 1 if cand_value == value else 1
        self._pending[key] = (value, cand_count)

        if cand_count >= self.required_repeats:
            self._confirmed[key] = value
            del self._pending[key]
            return value, True

        return self._confirmed[key], False

    def reset(self, key: Optional[Hashable] = None) -> None:
        """Oublie l'historique d'une clé (ou tout, si key=None)."""
        if key is None:
            self._confirmed.clear()
            self._pending.clear()
        else:
            self._confirmed.pop(key, None)
            self._pending.pop(key, None)
