"""
generate_hero_templates_manual.py — Ajustement manuel de la position et
de la rotation des cartes hero synthétisées depuis board/.
Phase 7 — Bot Poker Académique

Suite à generate_hero_templates.py (calibrage entièrement automatique) :
un calibrage basé sur cv2.matchTemplate peut laisser un résidu visible
de l'ancienne carte de référence sur les bords si l'alignement n'est pas
parfaitement pixel-perfect (score de calibrage élevé mais pas 1.0). Sans
accès à tes 40/37 vraies captures pour affiner l'automatisation à
l'aveugle, un ajustement à l'œil est plus fiable — et pour une vingtaine
de cartes par banque, tout à fait faisable à la main.

Le calibrage automatique (calibrate(), repris de generate_hero_templates.py)
sert de POINT DE DÉPART pour chaque carte : tu n'ajustes qu'à la marge,
pas de zéro. Une fenêtre s'ouvre par carte manquante, avec un aperçu en
direct qui se met à jour à chaque pression de touche.

Contrôles clavier (dans la fenêtre) :
    Flèches           -> déplacer de 1px
    Maj + Flèches     -> déplacer de 5px
    A / E             -> tourner de -1° / +1°
    Maj+A / Maj+E     -> tourner de -5° / +5°
    ENTRÉE            -> valider cette carte, passer à la suivante
    R                 -> réinitialiser au calibrage automatique
    S                 -> passer cette carte sans la sauvegarder
    Q                 -> arrêter (les cartes non traitées restent à faire)

Usage :
    python -m ocr.vision.generate_hero_templates_manual templates/board templates/hero_left templates_review/hero_left
    python -m ocr.vision.generate_hero_templates_manual templates/board templates/hero_right templates_review/hero_right

Comme pour la version automatique, le résultat va dans un dossier
SÉPARÉ (jamais directement dans templates/hero_left ou hero_right) — à
copier toi-même une fois satisfait (ex: via template_sorter.py, ou
juste en déplaçant les fichiers).
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

ANGLE_SEARCH_RANGE = range(-40, 41)
PAD_FACTOR = 1.0
MASK_DILATE_PX = 2  # marge de sécurité autour du masque, pour tolérer un petit résidu d'imprécision
ZOOM = 6


def _rotate_padded(img: Image.Image, angle: float, pad_w: int, pad_h: int, fill=(0, 0, 0)) -> Image.Image:
    w, h = img.size
    padded = Image.new("RGB", (w + 2 * pad_w, h + 2 * pad_h), fill)
    padded.paste(img, (pad_w, pad_h))
    return padded.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=fill)


def _rotate_mask(size: Tuple[int, int], angle: float, pad_w: int, pad_h: int, dilate_px: int = MASK_DILATE_PX) -> Image.Image:
    """
    Masque de la silhouette de carte tournée, avec une marge de sécurité
    (dilate_px) : mieux vaut montrer un ou deux pixels de trop de la
    NOUVELLE carte (ou de son remplissage transitoire) que de laisser
    filtrer l'ANCIENNE carte de référence par un masque trop juste.
    """
    w, h = size
    mask = Image.new("L", (w, h), 255)
    padded_mask = Image.new("L", (w + 2 * pad_w, h + 2 * pad_h), 0)
    padded_mask.paste(mask, (pad_w, pad_h))
    rotated = padded_mask.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=0)

    if dilate_px > 0:
        arr = np.asarray(rotated)
        kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
        arr = cv2.dilate(arr, kernel)
        rotated = Image.fromarray(arr)

    return rotated


def _best_offset(big: Image.Image, target: Image.Image) -> Optional[Tuple[int, int, float]]:
    big_arr = np.asarray(big.convert("RGB"))
    target_arr = np.asarray(target.convert("RGB"))
    if big_arr.shape[0] < target_arr.shape[0] or big_arr.shape[1] < target_arr.shape[1]:
        return None
    result = cv2.matchTemplate(big_arr, target_arr, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_loc[0], max_loc[1], max_val


def calibrate(board_dir: Path, hero_dir: Path, pad_factor: float = PAD_FACTOR) -> Tuple[int, int, int]:
    """Calibrage automatique (identique à generate_hero_templates.py) — sert de point de départ."""
    board_cards = {p.stem for p in board_dir.glob("*.png")}
    hero_cards = {p.stem for p in hero_dir.glob("*.png")}
    calibration_cards = sorted(board_cards & hero_cards)

    if len(calibration_cards) < 5:
        raise ValueError(
            f"Seulement {len(calibration_cards)} carte(s) en commun entre "
            f"{board_dir} et {hero_dir} — trop peu pour calibrer (5 minimum)."
        )

    pairs = [
        (Image.open(board_dir / f"{c}.png").convert("RGB"), Image.open(hero_dir / f"{c}.png").convert("RGB"))
        for c in calibration_cards
    ]
    board_w, board_h = pairs[0][0].size
    pad_w, pad_h = int(board_w * pad_factor), int(board_h * pad_factor)

    angle_scores: Dict[int, float] = {}
    for angle in ANGLE_SEARCH_RANGE:
        scores = []
        for board_img, hero_img in pairs:
            big = _rotate_padded(board_img, angle, pad_w, pad_h)
            r = _best_offset(big, hero_img)
            if r is not None:
                scores.append(r[2])
        if scores:
            angle_scores[angle] = sum(scores) / len(scores)

    best_angle = max(angle_scores, key=angle_scores.get)

    offsets = []
    for board_img, hero_img in pairs:
        big = _rotate_padded(board_img, best_angle, pad_w, pad_h)
        r = _best_offset(big, hero_img)
        if r is not None:
            offsets.append((r[0], r[1]))

    xs = sorted(o[0] for o in offsets)
    ys = sorted(o[1] for o in offsets)
    print(f"Calibrage automatique (point de départ) : angle={best_angle}°, "
          f"dx={xs[len(xs)//2]}, dy={ys[len(ys)//2]}, score moyen={angle_scores[best_angle]:.3f}")
    return best_angle, xs[len(xs) // 2], ys[len(ys) // 2]


def _compose(
    board_img: Image.Image,
    background_ref: Image.Image,
    angle: float,
    dx: int,
    dy: int,
    pad_w: int,
    pad_h: int,
) -> Tuple[Image.Image, int, int]:
    big = _rotate_padded(board_img, angle, pad_w, pad_h)
    big_mask = _rotate_mask(board_img.size, angle, pad_w, pad_h)

    hw, hh = background_ref.size
    dx = max(0, min(dx, big.width - hw))
    dy = max(0, min(dy, big.height - hh))

    aligned_card = big.crop((dx, dy, dx + hw, dy + hh))
    aligned_mask = big_mask.crop((dx, dy, dx + hw, dy + hh))

    result = background_ref.copy()
    result.paste(aligned_card, (0, 0), mask=aligned_mask)
    return result, dx, dy


def _adjust_card(
    card: str,
    board_img: Image.Image,
    background_ref: Image.Image,
    init_angle: int,
    init_dx: int,
    init_dy: int,
    pad_w: int,
    pad_h: int,
) -> Tuple[str, float, int, int]:
    """Ouvre la fenêtre d'ajustement pour UNE carte, bloque jusqu'à une action utilisateur."""
    state = {"angle": float(init_angle), "dx": init_dx, "dy": init_dy, "action": "skip"}

    root = tk.Tk()
    root.title(f"Ajuster : {card}")

    hw, hh = background_ref.size
    label = tk.Label(root)
    label.pack()
    info = tk.Label(root, text="", font=("Consolas", 10))
    info.pack()
    tk.Label(
        root,
        text="Flèches: déplacer (Maj=x5) | A/E: tourner (Maj=x5) | "
             "ENTRÉE: valider | R: reset | S: passer | Q: quitter",
        font=("Consolas", 9),
    ).pack()

    def render() -> None:
        img, dx, dy = _compose(board_img, background_ref, state["angle"], state["dx"], state["dy"], pad_w, pad_h)
        state["dx"], state["dy"] = dx, dy
        big_img = img.resize((hw * ZOOM, hh * ZOOM), Image.NEAREST)
        photo = ImageTk.PhotoImage(big_img, master=root)
        label.configure(image=photo)
        label.image = photo  # référence gardée : sinon le GC libère l'image
        info.configure(text=f"{card} — angle={state['angle']:.0f}°  dx={state['dx']}  dy={state['dy']}")

    def move(ddx: int, ddy: int):
        state["dx"] += ddx
        state["dy"] += ddy
        render()

    def rotate(delta: float):
        state["angle"] += delta
        render()

    def finish(action: str):
        def _handler(event=None):
            state["action"] = action
            root.quit()
        return _handler

    def reset(event=None):
        state["angle"], state["dx"], state["dy"] = float(init_angle), init_dx, init_dy
        render()

    root.bind("<Up>", lambda e: move(0, -1))
    root.bind("<Down>", lambda e: move(0, 1))
    root.bind("<Left>", lambda e: move(-1, 0))
    root.bind("<Right>", lambda e: move(1, 0))
    root.bind("<Shift-Up>", lambda e: move(0, -5))
    root.bind("<Shift-Down>", lambda e: move(0, 5))
    root.bind("<Shift-Left>", lambda e: move(-5, 0))
    root.bind("<Shift-Right>", lambda e: move(5, 0))
    root.bind("a", lambda e: rotate(-1))
    root.bind("e", lambda e: rotate(1))
    root.bind("A", lambda e: rotate(-5))
    root.bind("E", lambda e: rotate(5))
    root.bind("<Return>", finish("confirm"))
    root.bind("s", finish("skip"))
    root.bind("q", finish("quit"))
    root.bind("r", reset)

    render()
    root.mainloop()
    root.destroy()

    return state["action"], state["angle"], state["dx"], state["dy"]


def run(board_dir: str, hero_dir: str, output_dir: str) -> None:
    board_path, hero_path, out_path = Path(board_dir), Path(hero_dir), Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    angle0, dx0, dy0 = calibrate(board_path, hero_path)

    background_ref = Image.open(next(iter(hero_path.glob("*.png")))).convert("RGB")

    board_cards = {p.stem for p in board_path.glob("*.png")}
    hero_cards = {p.stem for p in hero_path.glob("*.png")}
    missing = sorted(board_cards - hero_cards)

    if not missing:
        print("Rien à générer : toutes les cartes de board sont déjà présentes.")
        return

    print(f"{len(missing)} carte(s) à ajuster.\n")

    saved = 0
    for card in missing:
        board_img = Image.open(board_path / f"{card}.png").convert("RGB")
        pad_w, pad_h = int(board_img.width * PAD_FACTOR), int(board_img.height * PAD_FACTOR)

        action, angle, dx, dy = _adjust_card(card, board_img, background_ref, angle0, dx0, dy0, pad_w, pad_h)

        if action == "quit":
            print(f"\nArrêté. {saved} carte(s) sauvegardée(s), le reste n'a pas été traité.")
            return
        if action == "skip":
            print(f"  {card} passée.")
            continue

        result, dx, dy = _compose(board_img, background_ref, angle, dx, dy, pad_w, pad_h)
        result.save(out_path / f"{card}.png")
        saved += 1
        print(f"  {card} -> sauvegardée (angle={angle:.0f}°, dx={dx}, dy={dy})")

    print(f"\n{saved}/{len(missing)} carte(s) sauvegardée(s) dans {output_dir}.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage : python -m ocr.vision.generate_hero_templates_manual "
              "<board_dir> <hero_dir_reel> <output_dir_a_verifier>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2], sys.argv[3])
