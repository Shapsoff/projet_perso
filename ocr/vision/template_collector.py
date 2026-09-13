"""
template_collector.py — Collecte semi-automatique des templates de cartes
Phase 7 — Bot Poker Académique

À exécuter EN LOCAL, en parallèle d'une session de jeu (idéalement en
argent fictif / table de démonstration si le client en propose une).
Ne prend AUCUNE décision et ne clique sur rien : lit seulement les
zones de cartes déjà calibrées (regions.json) et sauvegarde chaque
image DISTINCTE observée dans un dossier, à trier ensuite à l'œil.

Usage :
    python -m ocr.vision.template_collector "motif fenêtre" regions.json templates_raw/ [poll_seconds] [classe_fenêtre] [inclure_barre_titre:true|false]

Si le client ouvre plusieurs fenêtres partageant un titre proche (ex: le
lobby ET une table contenant toutes deux "CoinPoker"), le titre seul ne
suffit pas à les distinguer et Win32WindowLocator lève
AmbiguousWindowError — passe la classe exacte de la fenêtre de TABLE en
5e argument (repérable via list_windows_v2.py, colonne "Classe") pour
lever l'ambiguïté. Passe une chaîne vide "" pour ce champ si tu veux
seulement régler le 6e argument sans filtrer par classe.

inclure_barre_titre (6e argument, false par défaut) DOIT être 'true' si
et seulement si l'image de référence utilisée pour calibrer regions.json
a elle-même été capturée fenêtre ENTIÈRE (bandeau de titre inclus, ex:
via un script maison basé sur PrintWindow/GetWindowRect) plutôt que zone
cliente seule (comportement par défaut de capture_reference.py) — sinon
les régions calibrées ne correspondent plus à ce que ce script capture
réellement (décalage vertical + mauvaise échelle, cf. window_anchor.py).

Les deux réglages ci-dessous (DEFAULT_CLASS_NAME, DEFAULT_INCLUDE_TITLEBAR)
servent de valeurs par défaut — modifie-les une fois pour ta machine/ton
client. Les arguments CLI, s'ils sont fournis, restent prioritaires
dessus pour un usage ponctuel différent.

Capture via PrintWindowFrameGrabber (pas MssFrameGrabber) : capture le
CONTENU de la fenêtre elle-même, pas une zone d'écran — reste correct
même si une autre fenêtre passe devant la table pendant la collecte
(cf. frame_capture.py). DEFAULT_SCALE_FACTOR compense un scaling
Windows non 100% (125% ici) qui fait que GetWindowRect renvoie des
coordonnées plus petites que ce que PrintWindow rend réellement — ajuste
cette valeur si tes templates sortent flous/rognés.

Anti-animation (REQUIRED_STABLE_POLLS) : une carte qui vient d'apparaître
glisse/se retourne pendant quelques frames avant de se stabiliser —
sauvegarder dès la première différence détectée capture ces frames de
transition (images floues/partielles, superflues). Une valeur ne se
sauvegarde que si le MÊME crop (à threshold près) est revu sur
REQUIRED_STABLE_POLLS lectures consécutives, ce qui filtre les frames
d'animation sans dépendre d'un délai fixe (même principe que
FrameStabilizer dans validation.py, adapté à une comparaison d'image
plutôt qu'à une égalité exacte).

Les crops de board / hero_card_1 / hero_card_2 n'ont PAS le même rendu
sur ce client (cartes hero inclinées en éventail, board droit) — cf.
card_reader.py, qui exige 3 banques de templates séparées plutôt qu'une
seule. Ce script route donc chaque crop, dès la sauvegarde, dans le
sous-dossier correspondant (templates_raw/board/, .../hero_left/,
.../hero_right/) via card_reader.region_bank_name() — pas besoin de
re-trier par origine au moment du renommage manuel.

Ensuite, à la main, DANS CHAQUE sous-dossier séparément : identifier
chaque image à l'œil et la renommer <rang><couleur>.png (ex: As.png,
Td.png) — jeter les dos de carte, zones vides, doublons. Range le
résultat dans templates/board/, templates/hero_left/,
templates/hero_right/ : CardReader("templates/") charge automatiquement
les 3 banques depuis ces sous-dossiers.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .card_reader import region_bank_name
from .frame_capture import PrintWindowFrameGrabber, capture_table_frame
from .region_config import RegionConfig
from .window_anchor import Win32WindowLocator

CARD_REGION_PREFIXES = ("hero_card_", "board_card_")

# ============================================================
# CONFIGURATION — spécifique à ta machine / ton client, à régler une
# fois puis à ne plus toucher (même esprit que le bloc CONFIGURATION de
# capture_reference_v2.py).
# ============================================================
DEFAULT_CLASS_NAME: Optional[str] = "UnityWndClass"  # cf. list_windows_v2.py
DEFAULT_INCLUDE_TITLEBAR = True  # reference.png inclut le bandeau de titre
DEFAULT_SCALE_FACTOR = 1.25  # Windows à 125% — cf. capture_reference_v2.py
REQUIRED_STABLE_POLLS = 2  # lectures consécutives identiques avant sauvegarde
# ============================================================


def card_region_names(regions: RegionConfig) -> List[str]:
    """Sous-ensemble des zones calibrées qui contiennent des cartes (pas stacks/pot/pseudos)."""
    return [name for name in regions.names() if name.startswith(CARD_REGION_PREFIXES)]


def images_differ(a: Image.Image, b: Image.Image, threshold: float = 8.0) -> bool:
    """
    Compare deux crops pour décider si le nouveau vaut la peine d'être
    sauvegardé (carte différente / venant d'apparaître) plutôt que
    resauvegarder la même carte à chaque itération de la boucle de
    capture pendant qu'elle reste affichée.

    threshold en différence moyenne de pixel (échelle 0-255) : 8.0
    tolère le bruit de compression/anti-aliasing d'une frame à l'autre
    sans confondre deux cartes réellement différentes.
    """
    arr_a = np.asarray(a.convert("RGB"), dtype=np.float64)
    arr_b = np.asarray(b.convert("RGB"), dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        return True
    return float(np.abs(arr_a - arr_b).mean()) > threshold


def run(
    title_substring: str,
    regions_json_path: str,
    output_dir: str,
    poll_seconds: float = 1.0,
    class_name: Optional[str] = DEFAULT_CLASS_NAME,
    include_titlebar: bool = DEFAULT_INCLUDE_TITLEBAR,
    scale_factor: float = DEFAULT_SCALE_FACTOR,
    required_stable_polls: int = REQUIRED_STABLE_POLLS,
) -> None:
    locator = Win32WindowLocator(title_substring, class_name, include_titlebar)
    grabber = PrintWindowFrameGrabber(scale_factor)
    regions = RegionConfig.from_json(regions_json_path)

    names = card_region_names(regions)
    if not names:
        raise ValueError(
            "Aucune région 'hero_card_*' / 'board_card_*' trouvée dans "
            f"{regions_json_path} — vérifie les noms utilisés à la calibration."
        )

    out = Path(output_dir)
    # Un sous-dossier par banque de templates (cf. card_reader.py) : board
    # / hero_left / hero_right ont des rendus différents et ne doivent
    # jamais être mélangés dans une même banque — router dès la collecte
    # évite un tri manuel supplémentaire par origine.
    for bank_name in ("board", "hero_left", "hero_right"):
        (out / bank_name).mkdir(parents=True, exist_ok=True)
    last_saved: Dict[str, Image.Image] = {}
    # name -> (crop candidat, nombre de lectures consécutives stables)
    pending: Dict[str, Tuple[Image.Image, int]] = {}
    saved_count = 0

    print(f"Collecte démarrée sur {len(names)} zones : {', '.join(names)}")
    print("Joue normalement — CTRL+C pour arrêter.\n")

    try:
        while True:
            frame = capture_table_frame(locator, grabber, regions)
            if frame is None:
                print("Fenêtre introuvable, nouvelle tentative...")
                time.sleep(poll_seconds)
                continue

            for name in names:
                crop = frame.crop(name)

                # Un crop n'est retenu comme CANDIDAT que s'il ressemble
                # au précédent candidat (même image en train de se
                # stabiliser) ; sinon on repart de zéro avec ce nouveau
                # crop — ce qui inclut le tout premier crop transitoire
                # d'une animation, qui ne compte alors que pour 1.
                cand_crop, cand_count = pending.get(name, (None, 0))
                if cand_crop is not None and not images_differ(crop, cand_crop):
                    cand_count += 1
                else:
                    cand_crop, cand_count = crop, 1
                pending[name] = (cand_crop, cand_count)

                previous = last_saved.get(name)
                is_stable = cand_count >= required_stable_polls
                is_new = previous is None or images_differ(crop, previous)
                if is_stable and is_new:
                    timestamp = int(time.time() * 1000)
                    bank_name = region_bank_name(name)
                    path = out / bank_name / f"{name}_{timestamp}.png"
                    crop.save(path)
                    last_saved[name] = crop
                    pending.pop(name, None)
                    saved_count += 1
                    print(f"  + {bank_name}/{path.name}")

            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print(f"\nArrêté. {saved_count} images sauvegardées dans {output_dir}")
        print(
            "Étape suivante : DANS CHAQUE sous-dossier (board/, hero_left/, "
            "hero_right/) séparément, trie à l'œil et renomme les cartes "
            "faces visibles en <rang><couleur>.png (ex: As.png) — jette dos "
            "de carte, zones vides, doublons. Range le résultat dans "
            "templates/board/, templates/hero_left/, templates/hero_right/ : "
            "CardReader('templates/') les charge automatiquement."
        )


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5, 6, 7):
        print("Usage : python -m ocr.vision.template_collector "
              '"motif fenêtre" regions.json templates_raw/ [poll_seconds] '
              '[classe_fenêtre] [inclure_barre_titre:true|false]')
        sys.exit(1)

    kwargs = {}
    if len(sys.argv) >= 5:
        kwargs["poll_seconds"] = float(sys.argv[4])
    if len(sys.argv) >= 6 and sys.argv[5]:
        kwargs["class_name"] = sys.argv[5]
    if len(sys.argv) == 7:
        kwargs["include_titlebar"] = sys.argv[6].strip().lower() in ("1", "true", "yes", "oui")

    run(sys.argv[1], sys.argv[2], sys.argv[3], **kwargs)
