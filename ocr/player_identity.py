"""
player_identity.py — Résolution d'identité stable par siège
Phase 7 — Bot Poker Académique

Problème à résoudre : un siège physique n'est PAS un joueur. Un joueur
se lève, un autre s'assoit au même siège la main suivante — fusionner
leurs statistiques serait exactement le genre de bug documenté en phase
6 (cf. PHASE6_NOTES.md, fusion d'identités distinctes sous un même id
anonyme). À l'inverse, un pseudo mal lu une seule fois sur une main
bruitée ne doit PAS créer une nouvelle identité à chaque main pour le
même joueur réel — sinon la Player DB (core/player_db) ne accumule
jamais assez d'échantillon par joueur pour devenir fiable
(PlayerStats.is_reliable exige 30 mains, cf. core/game_state.py).

Stratégie :
  - Un pseudo différent de l'identité confirmée doit être lu
    `confirm_after` fois d'affilée (sur des mains différentes — le
    pseudo ne change jamais en cours de main, donc voter frame par
    frame n'aurait pas de sens) pour être promu nouvelle identité.
  - Un siège vu vide réinitialise IMMÉDIATEMENT toute identité
    (confirmée ou en attente) : la prochaine occupation démarre un vote
    neuf, sans continuité implicite avec l'ancien occupant.
  - Un pseudo illisible (None) sur une main ne casse jamais une
    identité déjà confirmée : on suppose le même joueur tant que le
    siège n'a pas été vu vide entre-temps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

_PSEUDO_CLEAN_RE = re.compile(r"[^\w\- ]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SeatIdentity:
    seat_index: int
    db_pid: str                # clé stable pour core.player_db.PlayerDB
    pseudo: Optional[str]      # dernier pseudo lu associé à cette identité (peut être None)
    is_new_identity: bool      # True le tour où cette identité vient d'être (ré)confirmée


def canonicalize_pseudo(pseudo: str) -> str:
    """Normalise un pseudo lu en OCR pour en faire une clé stable et comparable."""
    slug = _WHITESPACE_RE.sub(" ", pseudo).strip().lower()
    slug = _PSEUDO_CLEAN_RE.sub("", slug)
    return slug


class PlayerIdentityResolver:
    """Maintient l'association siège -> identité stable au fil des mains."""

    def __init__(self, confirm_after: int = 2) -> None:
        if confirm_after < 1:
            raise ValueError("confirm_after doit être >= 1")
        self.confirm_after = confirm_after
        # seat_index -> (db_pid, dernier pseudo brut connu)
        self._confirmed: Dict[int, Tuple[str, Optional[str]]] = {}
        # seat_index -> (db_pid candidat, nombre de votes consécutifs)
        self._pending: Dict[int, Tuple[str, int]] = {}

    def mark_seat_empty(self, seat_index: int) -> None:
        """Le siège vient d'être vu vide : toute identité connue pour lui est oubliée."""
        self._confirmed.pop(seat_index, None)
        self._pending.pop(seat_index, None)

    def resolve_seat(self, seat_index: int, pseudo: Optional[str]) -> SeatIdentity:
        """
        À appeler une fois par main (pas une fois par frame) pour chaque
        siège occupé, avec le pseudo lu pour cette main.
        """
        confirmed = self._confirmed.get(seat_index)

        if pseudo is None:
            if confirmed is not None:
                db_pid, known_pseudo = confirmed
                return SeatIdentity(seat_index, db_pid, known_pseudo, is_new_identity=False)
            # Jamais vu et illisible : identité anonyme provisoire, pas votée
            # (on ne veut pas confirmer une identité qu'on n'a jamais pu lire).
            fallback_pid = f"ocr:seat{seat_index}:unknown"
            return SeatIdentity(seat_index, fallback_pid, None, is_new_identity=False)

        slug = canonicalize_pseudo(pseudo)
        db_pid = f"ocr:{slug}" if slug else f"ocr:seat{seat_index}:unknown"

        if confirmed is not None and confirmed[0] == db_pid:
            self._pending.pop(seat_index, None)
            return SeatIdentity(seat_index, db_pid, pseudo, is_new_identity=False)

        if confirmed is None:
            # Première lecture jamais vue pour ce siège : rien à départager,
            # on confirme directement.
            self._confirmed[seat_index] = (db_pid, pseudo)
            self._pending.pop(seat_index, None)
            return SeatIdentity(seat_index, db_pid, pseudo, is_new_identity=True)

        # Une identité DIFFÉRENTE de celle confirmée est lue -> vote.
        cand_pid, cand_count = self._pending.get(seat_index, (None, 0))
        cand_count = cand_count + 1 if cand_pid == db_pid else 1
        self._pending[seat_index] = (db_pid, cand_count)

        if cand_count >= self.confirm_after:
            self._confirmed[seat_index] = (db_pid, pseudo)
            self._pending.pop(seat_index, None)
            return SeatIdentity(seat_index, db_pid, pseudo, is_new_identity=True)

        old_pid, old_pseudo = confirmed
        return SeatIdentity(seat_index, old_pid, old_pseudo, is_new_identity=False)
