"""
capture_reference.py — Capture unique de la zone cliente du client de poker,
à utiliser comme image de référence pour la calibration (cf.
calibration_tool.py).
Phase 7 — Bot Poker Académique

À exécuter EN LOCAL, sur la machine où tourne le client de poker (a
besoin de pywin32 et d'un affichage réel — ne fonctionne pas dans un
conteneur/environnement headless).

Usage :
    python -m ocr.vision.capture_reference "Nom (partiel) de la fenêtre" reference.png

L'image sauvegardée correspond EXACTEMENT à la zone cliente actuelle
(pas la fenêtre entière, pas l'écran entier) — c'est important : c'est
cette même zone que calibration_tool.py utilisera comme référence pour
convertir les rectangles dessinés en régions fractionnelles.
"""

from __future__ import annotations

import sys

from .frame_capture import MssFrameGrabber
from .window_anchor import Win32WindowLocator


def run(title_substring: str, output_path: str) -> None:
    locator = Win32WindowLocator(title_substring)
    bbox = locator.find_client_bbox()
    if bbox is None:
        raise RuntimeError(
            f"Aucune fenêtre visible dont le titre contient {title_substring!r} "
            "n'a été trouvée. Vérifie que le client est ouvert et non minimisé."
        )
    print(f"Fenêtre trouvée : zone cliente {bbox.width}x{bbox.height} à ({bbox.left}, {bbox.top})")

    grabber = MssFrameGrabber()
    image = grabber.grab(bbox)
    image.save(output_path)
    print(f"Référence sauvegardée : {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage : python -m ocr.vision.capture_reference "Titre de fenêtre" reference.png')
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
