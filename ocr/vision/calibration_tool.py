"""
calibration_tool.py — Calibration interactive des zones (rectangles) à lire
Phase 7 — Bot Poker Académique

À exécuter EN LOCAL, avec un affichage réel (ne fonctionne pas dans un
environnement headless). Nécessite `opencv-python` — PAS
`opencv-python-headless` : cv2.selectROI a besoin du support GUI, absent
du paquet headless. Si `pip show opencv-python-headless` affiche quelque
chose, désinstalle-le et installe `opencv-python` à la place.

Usage :
    1. Générer d'abord une image de référence de la zone CLIENTE (pas un
       screenshot brut de l'écran entier) via capture_reference.py —
       important : les régions calibrées ici seront relatives à CETTE
       image, donc elle doit correspondre exactement à ce que
       window_anchor.py capturera ensuite en direct.

           python -m ocr.vision.capture_reference "Nom de la fenêtre" reference.png

    2. Lancer la calibration :

           python -m ocr.vision.calibration_tool reference.png regions.json

    3. Pour chaque zone : une fenêtre s'ouvre sur l'image de référence.
       Dessiner le rectangle à la souris (clic-glisser), valider avec
       ESPACE ou ENTRÉE, ESC pour annuler cette zone. Le terminal
       propose ensuite un nom de zone suivant (liste indicative,
       éditable) ; taper une chaîne vide pour terminer et sauvegarder.
"""

from __future__ import annotations

import sys

import cv2

from .region_config import FractionalRegion, RegionConfig
from .window_anchor import BoundingBox

# Liste indicative pour ne pas avoir à se souvenir de toutes les zones à
# calibrer — proposée une à une comme valeur par défaut, mais librement
# remplaçable ou complétable (taper un autre nom au prompt).
SUGGESTED_REGION_NAMES = (
    ["hero_card_1", "hero_card_2"]
    + [f"board_card_{i}" for i in range(1, 6)]
    + ["pot"]
    + [
        f"seat_{i}_{field}"
        for i in range(6)
        for field in ("stack", "bet", "pseudo", "dealer_button", "to_act_indicator")
    ]
)


def run(reference_image_path: str, output_json_path: str) -> None:
    image = cv2.imread(reference_image_path)
    if image is None:
        raise FileNotFoundError(f"Image introuvable ou illisible : {reference_image_path}")

    height, width = image.shape[:2]
    # L'image de référence EST la zone cliente (cf. capture_reference.py) :
    # sa propre taille sert de bbox de référence pour la conversion en
    # fractions, à l'origine (0, 0) puisqu'on travaille ici en
    # coordonnées image, pas en coordonnées écran.
    reference_bbox = BoundingBox(left=0, top=0, width=width, height=height)

    regions: dict[str, FractionalRegion] = {}
    suggestions = iter(SUGGESTED_REGION_NAMES)

    print(f"Image de référence : {width}x{height}px")
    print("Pour chaque zone : dessiner le rectangle (clic-glisser), "
          "ESPACE/ENTRÉE pour valider, ESC pour annuler cette zone.")
    print("Nom de zone vide -> termine la calibration et sauvegarde.\n")

    while True:
        suggestion = next(suggestions, None)
        prompt = f"Nom de la zone{f' [{suggestion}]' if suggestion else ''} : "
        raw_name = input(prompt).strip()
        name = raw_name or (suggestion or "")
        if not name:
            break

        window_name = f"Calibration : {name}  (ESPACE=valider, ESC=annuler)"
        x, y, w, h = cv2.selectROI(window_name, image, showCrosshair=True)
        cv2.destroyWindow(window_name)

        if w == 0 or h == 0:
            print(f"  -> zone '{name}' annulée (rectangle vide), ignorée.")
            continue

        regions[name] = FractionalRegion.from_pixels((x, y, w, h), reference_bbox)
        r = regions[name]
        print(f"  -> '{name}' enregistrée : x={r.x:.4f} y={r.y:.4f} w={r.w:.4f} h={r.h:.4f}")

    if not regions:
        print("Aucune zone calibrée, rien à sauvegarder.")
        return

    RegionConfig(regions).to_json(output_json_path)
    print(f"\n{len(regions)} zones sauvegardées dans {output_json_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage : python -m ocr.vision.calibration_tool <reference.png> <regions.json>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
