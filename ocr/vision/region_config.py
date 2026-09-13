"""
region_config.py — Zones calibrées en coordonnées RELATIVES (fractions de
la fenêtre cliente), pas en pixels absolus.
Phase 7 — Bot Poker Académique

Une FractionalRegion mémorise (x, y, w, h) comme des fractions entre 0 et
1 de la largeur/hauteur de la zone cliente au moment de la calibration —
exactement le principe du responsive design web (% plutôt que px). À
l'exécution, to_pixels() les reconvertit en pixels ÉCRAN ABSOLUS à partir
de la BoundingBox ACTUELLE de la fenêtre (cf. window_anchor.py), quelle
que soit sa taille/position au moment présent.

Limite assumée : ça suppose que le client redessine son contenu de façon
proportionnelle quand on le redimensionne (vrai pour la quasi-totalité
des clients modernes basés sur du rendu web/Electron). Si un élément
précis de l'UI reste à taille fixe en pixels quel que soit le
redimensionnement (rare, mais ça arrive pour des éléments de chrome comme
une barre de menu), sa région calibrée dérivera légèrement à des tailles
de fenêtre très différentes de la référence — recalibrer à une taille
proche de l'usage réel limite le risque plutôt que de chercher à couvrir
toutes les tailles possibles.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple, Union

from .window_anchor import BoundingBox

PixelRect = Tuple[int, int, int, int]  # (left, top, width, height), écran absolu


@dataclass(frozen=True)
class FractionalRegion:
    """Une zone, en fractions 0..1 de la largeur/hauteur de la fenêtre cliente."""
    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        for name, v in (("x", self.x), ("y", self.y), ("w", self.w), ("h", self.h)):
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name}={v} hors de l'intervalle [0, 1]")
        if self.x + self.w > 1.0 + 1e-9:
            raise ValueError(f"x + w = {self.x + self.w:.4f} dépasse la largeur de la fenêtre (>1.0)")
        if self.y + self.h > 1.0 + 1e-9:
            raise ValueError(f"y + h = {self.y + self.h:.4f} dépasse la hauteur de la fenêtre (>1.0)")

    def to_pixels(self, bbox: BoundingBox) -> PixelRect:
        return (
            bbox.left + round(self.x * bbox.width),
            bbox.top + round(self.y * bbox.height),
            max(1, round(self.w * bbox.width)),
            max(1, round(self.h * bbox.height)),
        )

    @staticmethod
    def from_pixels(rect: PixelRect, reference_bbox: BoundingBox) -> "FractionalRegion":
        """
        Convertit un rectangle dessiné en pixels sur une image de
        référence (dont les dimensions/position correspondent à
        `reference_bbox`) en région fractionnelle réutilisable à toute
        taille de fenêtre. C'est ce qu'utilise calibration_tool.py après
        chaque clic-glisser.
        """
        left, top, width, height = rect
        return FractionalRegion(
            x=(left - reference_bbox.left) / reference_bbox.width,
            y=(top - reference_bbox.top) / reference_bbox.height,
            w=width / reference_bbox.width,
            h=height / reference_bbox.height,
        )


class RegionConfig:
    """Ensemble nommé de FractionalRegion, chargeable/sauvegardable en JSON."""

    def __init__(self, regions: Dict[str, FractionalRegion]) -> None:
        self.regions = dict(regions)

    def __contains__(self, name: str) -> bool:
        return name in self.regions

    def __len__(self) -> int:
        return len(self.regions)

    def names(self) -> list:
        return list(self.regions.keys())

    def resolve(self, name: str, bbox: BoundingBox) -> PixelRect:
        if name not in self.regions:
            raise KeyError(f"Région inconnue : {name!r} (disponibles : {self.names()})")
        return self.regions[name].to_pixels(bbox)

    def resolve_all(self, bbox: BoundingBox) -> Dict[str, PixelRect]:
        return {name: region.to_pixels(bbox) for name, region in self.regions.items()}

    def to_json(self, path: Union[str, Path]) -> None:
        data = {name: asdict(region) for name, region in self.regions.items()}
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "RegionConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({name: FractionalRegion(**fields) for name, fields in data.items()})
