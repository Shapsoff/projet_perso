"""
test_card_reader.py — Test manuel de card_reader.CardReader sur des crops réels
Phase 7 — Bot Poker Académique

Étape 7 du plan : vérifie que CardReader reconnaît correctement les
cartes AVANT de le brancher dans le pipeline complet
(table_read_from_frame, pas encore construit). Charge la banque une
seule fois, prédit chaque image d'un dossier, l'affiche (même mécanisme
que template_sorter.py) à côté de sa prédiction, et calcule un taux de
reconnaissance à la fin.

Boucle de correction : quand une prédiction est fausse, propose
d'ajouter l'image comme nouvel exemplaire de la VRAIE carte dans la
banque (cf. card_reader.next_variant_path — plusieurs exemplaires par
carte sont supportés, le meilleur score l'emporte au moment du match).
Optionnel à chaque fois (jamais automatique) : toutes les erreurs ne
sont pas de bons exemples à mémoriser (image mi-animation malgré le
filtre de stabilisation, cas limite ambigu...) — à toi de juger.

Usage :
    python -m ocr.vision.test_card_reader templates/ board_card_1 templates/board
    python -m ocr.vision.test_card_reader templates/ hero_card_1 templates_raw/hero_left

Le 1er argument est TOUJOURS le dossier racine des 3 banques (celui que
CardReader(...) charge en entier, cf. card_reader.py) — le 3e argument
est le dossier des images à tester (peut être templates/board lui-même
pour un premier sanity-check, ou mieux : des crops non utilisés dans la
banque, pour un test plus représentatif d'une vraie capture).

Pour chaque image, une fenêtre s'ouvre et le terminal demande :
    ENTRÉE seule       -> la prédiction affichée était correcte
    '<rang><couleur>'  -> la prédiction était fausse, la vraie carte est
                          celle-ci (ex: 'As') — comptée comme erreur,
                          puis proposition de l'ajouter à la banque
    's'                -> ignorer cette image (ni bonne ni mauvaise)
    'q'                -> arrêter et afficher le rapport avec ce qui a
                          été vu jusque-là
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageTk

from core.deck import RANKS, SUITS

from .card_reader import CardReader, next_variant_path


def _parse_card(raw: str) -> Optional[str]:
    raw = raw.strip()
    if len(raw) != 2:
        return None
    rank, suit = raw[0].upper(), raw[1].lower()
    if rank not in RANKS or suit not in SUITS:
        return None
    return f"{rank}{suit}"


def _show(path: Path, zoom: int = 6) -> tk.Tk:
    root = tk.Tk()
    root.title(path.name)
    img = Image.open(path).convert("RGB")
    big = img.resize((img.width * zoom, img.height * zoom), Image.NEAREST)
    photo = ImageTk.PhotoImage(big, master=root)
    label = tk.Label(root, image=photo)
    label.image = photo  # référence gardée : sinon le GC libère l'image affichée
    label.pack()
    root.update_idletasks()
    root.update()
    return root


def run(templates_root: str, region_name: str, test_dir: str) -> None:
    print(f"Chargement de la banque depuis {templates_root}...")
    reader = CardReader(templates_root)

    files = sorted(Path(test_dir).glob("*.png"))
    if not files:
        print(f"Aucune image .png trouvée dans {test_dir}.")
        return

    print(f"{len(files)} image(s) à tester sur la région '{region_name}'.\n")
    print("ENTRÉE = prédiction correcte | '<rang><couleur>' = fausse (indique la vraie carte) "
          "| 's' = ignorer | 'q' = arrêter\n")

    correct = 0
    wrong: List[Tuple[str, str, str, float]] = []  # (fichier, prédit, vrai, confidence)
    skipped = 0

    for path in files:
        img = Image.open(path).convert("RGB")
        card = reader.read_card_region(region_name, img)
        predicted = card.as_str() if card.is_visible() else "(vide/non reconnue)"

        window = _show(path)
        answer = input(f"[{path.name}] prédit: {predicted}  (confidence={card.confidence:.3f}) -> ").strip()
        window.destroy()

        lowered = answer.lower()
        if lowered == "q":
            print("\nArrêté.")
            break
        if lowered == "s":
            skipped += 1
            continue
        if answer == "":
            correct += 1
            continue

        true_card = _parse_card(answer)
        if true_card is None:
            print(f"  ! '{answer}' non reconnu comme carte — traité comme 'ignorer'.")
            skipped += 1
            continue

        wrong.append((path.name, predicted, true_card, card.confidence))

        add = input(f"  Ajouter cette image comme nouvel exemplaire de '{true_card}' "
                    "dans la banque ? (o/N) : ").strip().lower()
        if add == "o":
            dest = next_variant_path(reader.bank_dir(region_name), true_card)
            img.save(dest)
            print(f"  -> ajoutée : {dest} (pris en compte au PROCHAIN chargement "
                  "de CardReader, pas dans cette session en cours)")

    total_judged = correct + len(wrong)
    print("\n--- Rapport ---")
    if total_judged:
        print(f"Correctes : {correct}/{total_judged} ({100 * correct / total_judged:.1f}%)")
    else:
        print("Correctes : 0/0 (aucune image jugée)")
    print(f"Ignorées : {skipped}")

    if wrong:
        print("\nErreurs :")
        for fname, predicted, true_card, conf in wrong:
            print(f"  {fname} : prédit {predicted} (confidence={conf:.3f}), en réalité {true_card}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage : python -m ocr.vision.test_card_reader "
              "<templates_root> <region_name> <dossier_images_a_tester>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2], sys.argv[3])
