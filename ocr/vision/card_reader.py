"""
card_reader.py — Reconnaissance de cartes par template matching
Phase 7 — Bot Poker Académique

Sur ce client, une carte n'est PAS rendue de façon identique selon d'où
elle vient : les deux cartes hero (affichées "en éventail") sont
inclinées — la gauche vers la gauche, la droite vers la droite — alors
que les cartes du board sont bien droites, non chevauchées. cv2.match-
Template n'est PAS invariant à la rotation : un template capturé droit
(depuis le board) matche mal contre un crop incliné (depuis la main),
même après normalize_for_template_matching (qui ne fait que redimen-
sionner, aucune correction d'angle).

Solution retenue : 3 banques de templates SÉPARÉES plutôt qu'une
correction géométrique (dé-skew) au runtime — fiable (pas d'angle à
calibrer, pas de risque de rognage après rotation) au prix d'un tri
manuel ~3x plus long à la collecte. template_collector.py route déjà
chaque crop vers le bon sous-dossier selon son origine :

    templates/board/<rang><couleur>.png       — board_card_1..5
    templates/hero_left/<rang><couleur>.png   — hero_card_1
    templates/hero_right/<rang><couleur>.png  — hero_card_2

(hero_card_1 = carte affichée à gauche, hero_card_2 = carte affichée à
droite, cf. l'ordre dans calibration_tool.SUGGESTED_REGION_NAMES —
vérifie que ta calibration respecte bien ce sens si tu obtiens des
lectures inversées.)

Convention de nommage héritée de calibration_tool.py /
template_collector.py : <rang><couleur>.png, ex. "As.png" (As de pique),
"Td.png" (Dix de carreau) — rang et couleur concaténés sans séparateur.
Le rang fait TOUJOURS un seul caractère (2-9, T, J, Q, K, A), donc la
couleur est systématiquement le dernier caractère du nom de fichier.
Plusieurs fichiers peuvent exister pour une même carte ('As.png',
'As_2.png'...) : cf. next_variant_path.

Sur les confusions résiduelles (chiffres à courbes proches — 6/8/9/3) :
deux pistes algorithmiques (letterbox préservant le ratio, recadrage
automatique sur la zone la plus variable de la banque) ont été testées
et n'ont pas apporté d'amélioration mesurable en pratique — abandonnées
volontairement pour ne pas garder de complexité sans bénéfice avéré.
La parade retenue à la place : la correction manuelle au fil de l'eau
(cf. next_variant_path, test_card_reader.py) — accumuler des exemplaires
réels au lieu d'ajuster l'algorithme à l'aveugle. À rouvrir seulement si
ces confusions s'avèrent réellement handicapantes pour le bot une fois
branché en conditions réelles.

Usage typique (depuis state_builder.py / table_read_from_frame, une
fois construit) :

    reader = CardReader("templates/")  # charge les 3 banques une fois
    ...
    card = reader.read_card_region("hero_card_1", frame.crop("hero_card_1"))
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from core.deck import RANKS, SUITS

from .frame_capture import normalize_for_template_matching
from ..vision_types import CardRead

# Taille canonique de comparaison (ratio ~ carte à jouer standard,
# 63x88mm). Doit rester identique entre templates et crops lus, mais
# la valeur elle-même est arbitraire — ajuste-la si tes templates
# collectés ont un ratio très différent une fois recadrés.
CANONICAL_SIZE: Tuple[int, int] = (64, 89)

# Variance de pixel (niveaux de gris) sous laquelle un crop est jugé
# VIDE (pas de carte distribuée à cet emplacement) plutôt que "carte
# présente mais mal reconnue" : une face de carte contient toujours du
# texte/motif à fort contraste, un tapis de table est comparativement
# uniforme, quel que soit le thème. Valeur de départ à AJUSTER
# empiriquement sur de vraies captures — même logique que le seuil de
# 8.0 dans template_collector.images_differ, qui répond à une question
# voisine (deux images sont-elles "la même" ?).
EMPTY_SLOT_STD_THRESHOLD = 12.0

_BOARD_PREFIX = "board_card_"
_HERO_LEFT_REGION = "hero_card_1"
_HERO_RIGHT_REGION = "hero_card_2"
_BANK_NAMES = ("board", "hero_left", "hero_right")


def region_bank_name(region_name: str) -> str:
    """
    Détermine quelle banque de templates utiliser pour une région de
    carte calibrée. Volontairement strict (ValueError sur toute région
    non explicitement reconnue, même si elle commence par "hero_card_")
    plutôt que de deviner une banque par défaut : lire une carte hero
    avec la banque board (ou l'inverse) donnerait une erreur silencieuse
    difficile à détecter, pas un crash — si tu ajoutes une nouvelle
    région de carte (ex: showdown d'un autre siège), ajoute-la ici
    explicitement plutôt que de laisser passer un mauvais routing.
    """
    if region_name.startswith(_BOARD_PREFIX):
        return "board"
    if region_name == _HERO_LEFT_REGION:
        return "hero_left"
    if region_name == _HERO_RIGHT_REGION:
        return "hero_right"
    raise ValueError(
        f"Région inconnue pour la lecture de carte : {region_name!r} — "
        f"attendu '{_BOARD_PREFIX}*', {_HERO_LEFT_REGION!r} ou "
        f"{_HERO_RIGHT_REGION!r}"
    )


def _parse_card_filename(stem: str) -> Tuple[str, str]:
    """
    Découpe un nom de fichier '<rang><couleur>' (ex: 'As' -> ('A', 's'),
    'Td' -> ('T', 'd')), en ignorant un éventuel suffixe '_N' (ex:
    'As_2', 'As_3' -> même carte que 'As') : plusieurs fichiers peuvent
    exister pour une même carte, chacun un exemplaire différent (cf.
    CardTemplateBank, qui garde tous les exemplaires et prend le
    meilleur score au moment du match — utile pour accumuler des
    corrections au fil des sessions sans jamais perdre l'exemplaire
    d'origine).
    """
    base = stem.split("_")[0]
    if len(base) != 2:
        raise ValueError(
            f"Nom de template invalide : {stem!r} — attendu '<rang><couleur>' "
            "ou '<rang><couleur>_N' (ex: 'As', 'As_2')."
        )
    rank, suit = base[0], base[1]
    if rank not in RANKS:
        raise ValueError(f"Rang invalide dans le nom de template {stem!r} : {rank!r}")
    if suit not in SUITS:
        raise ValueError(f"Couleur invalide dans le nom de template {stem!r} : {suit!r}")
    return rank, suit


def next_variant_path(bank_dir: Path, card: str) -> Path:
    """
    Détermine le prochain nom de fichier disponible pour ajouter un
    nouvel exemplaire de `card` dans `bank_dir` (ex: si 'As.png' existe
    déjà, retourne '.../As_2.png' ; si 'As_2.png' existe aussi,
    '.../As_3.png'). Ne crée ni n'écrase rien — juste le chemin, à
    l'appelant de sauvegarder l'image dessus.
    """
    if not (bank_dir / f"{card}.png").exists():
        return bank_dir / f"{card}.png"
    n = 2
    while (bank_dir / f"{card}_{n}.png").exists():
        n += 1
    return bank_dir / f"{card}_{n}.png"


def _looks_empty(crop: Image.Image, threshold: float) -> bool:
    """
    Heuristique bon marché pour distinguer "pas encore de carte ici" de
    "carte présente". Ne compare à AUCUNE image de référence (donc
    indépendante du thème/de la couleur du tapis) : juste la variance
    intrinsèque du crop, en niveaux de gris.
    """
    arr = np.asarray(crop.convert("L"), dtype=np.float64)
    return float(arr.std()) < threshold


@dataclass(frozen=True)
class MatchResult:
    rank: str
    suit: str
    score: float  # cv2.TM_CCOEFF_NORMED, plus haut = plus proche (peut être négatif)


class CardTemplateBank:
    """
    Banque de templates pour UNE origine de crop (board / hero_left /
    hero_right) — cf. docstring de module pour pourquoi ces trois
    origines ne doivent jamais être mélangées dans une même banque.
    """

    def __init__(
        self,
        templates_dir: Union[str, Path],
        canonical_size: Tuple[int, int] = CANONICAL_SIZE,
    ) -> None:
        self.canonical_size = canonical_size
        self._templates: Dict[str, List[np.ndarray]] = {}
        self._load(Path(templates_dir))

    def _load(self, templates_dir: Path) -> None:
        if not templates_dir.is_dir():
            raise FileNotFoundError(
                f"Dossier de templates introuvable : {templates_dir} — "
                "as-tu trié les crops collectés par template_collector.py "
                "dans ce dossier (un rang+couleur par fichier, ex: As.png) ?"
            )
        for path in sorted(templates_dir.glob("*.png")):
            rank, suit = _parse_card_filename(path.stem)
            with Image.open(path) as image:
                normalized = normalize_for_template_matching(image, self.canonical_size)
            self._templates.setdefault(f"{rank}{suit}", []).append(normalized)

        if not self._templates:
            raise ValueError(f"Aucun template .png trouvé dans {templates_dir}")

    def __len__(self) -> int:
        """Nombre de CARTES DISTINCTES chargées (pas le nombre total d'exemplaires)."""
        return len(self._templates)

    def match(self, crop: Image.Image) -> MatchResult:
        """
        Compare `crop` (déjà découpé, PAS encore normalisé) à tous les
        exemplaires de tous les templates de cette banque, et retourne
        le meilleur — le score d'une carte qui a plusieurs exemplaires
        est le MEILLEUR des scores individuels (pas une moyenne) : un
        seul bon exemplaire suffit à identifier la carte, peu importe
        que les autres exemplaires soient plus ou moins représentatifs
        du crop actuel (angle de capture, luminosité...). C'est ce
        mécanisme qui permet d'accumuler des corrections au fil du temps
        (cf. next_variant_path / test_card_reader.py) : chaque nouvel
        exemplaire ajouté ne fait qu'augmenter les chances de bon match,
        jamais les diminuer.

        Ne décide JAMAIS "carte non lisible" à ce niveau : c'est le rôle
        de CardReader.read_card_region() (détection d'emplacement vide)
        en amont, et de validation.validate_confidence() en aval —
        cette méthode retourne toujours son meilleur candidat, aussi
        mauvais soit le score, à charge de l'appelant de décider quoi en
        faire (cf. docstring de module).
        """
        probe = normalize_for_template_matching(crop, self.canonical_size)
        best_key: Optional[str] = None
        best_score = -1.0
        for key, templates in self._templates.items():
            score = max(
                float(cv2.matchTemplate(probe, template, cv2.TM_CCOEFF_NORMED).max())
                for template in templates
            )
            if score > best_score:
                best_score = score
                best_key = key
        assert best_key is not None  # _load garantit >= 1 template chargé
        rank, suit = _parse_card_filename(best_key)
        return MatchResult(rank=rank, suit=suit, score=best_score)


class CardReader:
    """
    Point d'entrée principal du module : route chaque région vers la
    bonne banque de templates et applique la détection d'emplacement
    vide en amont du template matching. Une seule instance à construire
    au démarrage du bot (le chargement des templates est le seul coût
    non négligeable) puis réutilisée à chaque frame.
    """

    def __init__(
        self,
        templates_root: Union[str, Path],
        canonical_size: Tuple[int, int] = CANONICAL_SIZE,
        empty_slot_std_threshold: float = EMPTY_SLOT_STD_THRESHOLD,
    ) -> None:
        self.root = Path(templates_root)
        self.empty_slot_std_threshold = empty_slot_std_threshold
        self._banks: Dict[str, CardTemplateBank] = {
            name: CardTemplateBank(self.root / name, canonical_size) for name in _BANK_NAMES
        }

    def bank_dir(self, region_name: str) -> Path:
        """Dossier disque de la banque correspondant à `region_name` (cf. region_bank_name)."""
        return self.root / region_bank_name(region_name)

    def read_card_region(self, region_name: str, crop: Image.Image) -> CardRead:
        """
        Lit UNE région de carte déjà découpée (cf. TableFrame.crop()) et
        retourne le CardRead correspondant.

        - Emplacement jugé vide (cf. _looks_empty) -> CardRead() : rank
          et suit à None, la convention "carte non visible" de
          vision_types.py — couvre ici le cas normal des board_card_4/5
          avant le turn/river, ou des hero_card_* avant le début de la
          main.
        - Sinon -> meilleur match de la banque appropriée, AVEC son
          score de corrélation tel quel comme confidence, même bas : ce
          n'est pas ce module qui juge qu'une carte face visible est
          illisible, c'est validation.validate_confidence() en aval.
        """
        bank_name = region_bank_name(region_name)
        if _looks_empty(crop, self.empty_slot_std_threshold):
            return CardRead()

        match = self._banks[bank_name].match(crop)
        return CardRead(rank=match.rank, suit=match.suit, confidence=match.score)
