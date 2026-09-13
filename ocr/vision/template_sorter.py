"""
template_sorter.py — Tri interactif des crops collectés par template_collector.py
Phase 7 — Bot Poker Académique

À exécuter EN LOCAL, avec un affichage réel.

Affiche un par un les crops accumulés dans templates_raw/<banque>/ (UNE
banque à la fois — board, hero_left ou hero_right, cf. card_reader.py
pour pourquoi elles restent séparées) et demande son rang+couleur DANS
LE TERMINAL.

Affichage via tkinter (bibliothèque standard Python), PAS via
cv2.imshow ni PIL Image.show() ni un programme externe :
  - cv2.imshow a montré un rendu noir peu fiable sur cette machine
    (backend, indépendant de la capture elle-même — les fichiers sont
    corrects, vérifié en les ouvrant manuellement).
  - PIL Image.show() affiche correctement, mais ne renvoie aucun moyen
    de fermer la fenêtre par code (lance le visualisateur par défaut de
    l'OS en fire-and-forget) — les fenêtres s'accumulent.
  - Lancer un programme externe (mspaint.exe) contournait ça, mais reste
    un processus externe plutôt qu'une solution 100% Python.
tkinter (+ PIL.ImageTk pour convertir l'image) reste dans le process
Python, donne un vrai handle (root.destroy()) pour fermer la fenêtre
juste après la réponse, et utilise un pipeline de rendu différent de
cv2 — qui devrait éviter le souci de rendu noir rencontré avec cv2.

Contrepartie mineure : Windows peut brièvement afficher la fenêtre comme
« Non répondant » pendant l'attente de ta réponse dans le terminal (elle
ne traite aucun message pendant ce temps) — l'image reste affichée
normalement, ce n'est qu'un indicateur cosmétique, pas un plantage.

Usage (une fois par banque) :
    python -m ocr.vision.template_sorter templates_raw/board templates/board
    python -m ocr.vision.template_sorter templates_raw/hero_left templates/hero_left
    python -m ocr.vision.template_sorter templates_raw/hero_right templates/hero_right

Pour chaque image, dans CE terminal :
    - '<rang><couleur>' (ex: As, Td, 9h ; casse ignorée) -> renomme et
      DÉPLACE le fichier vers templates/<banque>/<rang><couleur>.png. Si
      ce fichier existe déjà, demande confirmation avant d'écraser.
    - 'skip' -> passe au suivant sans y toucher.
    - 'del'  -> supprime le fichier.
    - 'q'    -> arrête le tri ici ; les fichiers non traités restent dans
      templates_raw/<banque>/ pour la prochaine fois.

À la fin (ou sur 'q'), affiche les cartes de cette banque ENCORE
manquantes dans templates/<banque>/ (parmi les 52 possibles).
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from typing import Optional

from PIL import Image, ImageTk

from core.deck import RANKS, SUITS

ALL_CARDS = [f"{rank}{suit}" for rank in RANKS for suit in SUITS]


def _parse_card(raw: str) -> Optional[str]:
    """
    Valide une saisie '<rang><couleur>' contre RANKS/SUITS ; None si
    invalide. Casse ignorée à la saisie (rang normalisé en majuscule,
    couleur en minuscule) pour matcher la convention <RANG><couleur>
    (ex: 'As', 'Td') quelle que soit la façon dont l'utilisateur tape.
    """
    raw = raw.strip()
    if len(raw) != 2:
        return None
    rank, suit = raw[0].upper(), raw[1].lower()
    if rank not in RANKS or suit not in SUITS:
        return None
    return f"{rank}{suit}"


def _show_preview(image_path: Path) -> tk.Tk:
    """
    Affiche `image_path` agrandi (sans lissage, pour rester net malgré
    la petite taille du crop) dans une fenêtre tkinter, et force son
    rendu immédiat via update(). Retourne la racine Tk — à fermer
    explicitement (root.destroy()) une fois la réponse reçue.
    """
    root = tk.Tk()
    root.title(image_path.name)

    with Image.open(image_path) as img:
        preview = img.resize((img.width * 4, img.height * 4), Image.NEAREST)
        photo = ImageTk.PhotoImage(preview, master=root)

    label = tk.Label(root, image=photo)
    label.image = photo  # référence gardée : sinon le GC libère l'image affichée
    label.pack()

    root.update_idletasks()
    root.update()
    return root


def run(raw_dir: str, templates_dir: str) -> None:
    raw_path = Path(raw_dir)
    out_path = Path(templates_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_path.glob("*.png"))
    if not files:
        print(f"Aucune image .png trouvée dans {raw_dir}.")
        return

    print(f"{len(files)} image(s) à trier dans {raw_dir}.")
    print("Une fenêtre s'ouvre pour chaque crop et se referme automatiquement "
          "après ta réponse — réponds dans CE terminal.")
    print("'<rang><couleur>' (ex: As, Td, 9h), 'skip', 'del' ou 'q'.\n")

    remaining = len(files)
    for path in files:
        try:
            window: Optional[tk.Tk] = _show_preview(path)
        except Exception as exc:
            print(f"  ! Impossible d'afficher {path.name} ({exc}) — ignorée.")
            window = None

        answer = input(f"[{remaining} restante(s)] {path.name} -> ").strip()
        if window is not None:
            window.destroy()
        remaining -= 1
        lowered = answer.lower()

        if lowered == "q":
            print(f"\nArrêté. Les images non traitées restent dans {raw_dir}.")
            break
        if lowered == "skip":
            continue
        if lowered == "del":
            path.unlink()
            print(f"  {path.name} -> supprimée.")
            continue

        card = _parse_card(answer)
        if card is None:
            print(f"  ! '{answer}' non reconnu (attendu ex: 'As', 'Td') — "
                  f"{path.name} laissée en place, retape-la plus tard.")
            continue

        dest = out_path / f"{card}.png"
        if dest.exists():
            confirm = input(f"  '{card}.png' existe déjà — remplacer ? (o/N) : ").strip().lower()
            if confirm != "o":
                print("  -> ancien conservé, image source laissée en place.")
                continue
            dest.unlink()

        path.rename(dest)
        print(f"  {path.name} -> {dest}")

    have = {p.stem for p in out_path.glob("*.png")}
    missing = [c for c in ALL_CARDS if c not in have]
    print(f"\n{len(have)}/52 cartes présentes dans {templates_dir}.")
    if missing:
        print(f"Manquantes ({len(missing)}) : {', '.join(missing)}")
    else:
        print("Banque complète !")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage : python -m ocr.vision.template_sorter "
              "templates_raw/<banque> templates/<banque>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
