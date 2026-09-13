"""
generate_hero_templates.py — Génère les templates hero_left/hero_right
manquants par rotation calibrée des templates board correspondants.
Phase 7 — Bot Poker Académique

Motivation : board/ est plus rapide à compléter (visible à toute main
qui atteint le flop) que hero_left/hero_right (seulement tes propres
cartes, distribuées au hasard). Le dessin du rang/couleur est le même
asset graphique dans les deux cas, seule l'inclinaison diffère (cf.
card_reader.py) — on peut donc synthétiser les cartes hero encore
manquantes en tournant leur équivalent board, plutôt que d'attendre de
les voir toutes apparaître naturellement en jeu.

Deux choix de méthode, par rapport à une simple "rotation + remplissage
noir" :

1. Angle ESTIMÉ AUTOMATIQUEMENT, pas deviné à l'œil ni codé en dur :
   pour chaque carte présente à la fois dans board/ et dans le dossier
   hero réel (tes vrais exemples déjà collectés), on cherche l'angle de
   rotation qui, appliqué au template board, correspond le mieux à la
   vraie image hero (cv2.matchTemplate trouve en même temps le meilleur
   angle ET la meilleure position, sans supposer de signe ou de valeur a
   priori). Le résultat est agrégé sur TOUTES les cartes en commun (40
   pour hero_left, 37 pour hero_right dans ton cas), pas une seule paire
   — plus robuste qu'un calibrage sur un seul exemple.

2. Fond RÉEL plutôt que noir : les coins qu'une carte tournée ne
   recouvre pas dans son rectangle de capture montrent en réalité le
   feutre de la table, pas du noir. Remplir en noir fonctionnerait sans
   doute pour le template matching (cv2.matchTemplate reste surtout
   sensible aux STRUCTURES corrélées, pas à la couleur de fond exacte),
   mais coller la carte tournée sur un vrai fond pris d'un de tes
   exemples hero réels (via un masque de la forme de la carte tournée)
   colle plus fidèlement à ce qu'une vraie capture montre — sans coût
   supplémentaire notable.

Les résultats sont écrits dans un dossier SÉPARÉ (jamais directement
dans templates/hero_left/ ou hero_right/) : à vérifier visuellement
(ex: via template_sorter.py pointé sur ce dossier, ou juste à l'œil)
avant de les copier toi-même dans la banque définitive — un calibrage
qui aurait mal tourné (angle instable, cartes mal alignées) doit rester
visible avant d'entrer dans une banque dont dépend tout le reste du
pipeline (card_reader.py).

Usage :
    python -m ocr.vision.generate_hero_templates templates/board templates/hero_left templates_review/hero_left
    python -m ocr.vision.generate_hero_templates templates/board templates/hero_right templates_review/hero_right

Le script affiche l'angle retenu, le score moyen de calibrage (proche de
1.0 = très fiable, en dessous de ~0.85 = à vérifier avec plus
d'attention) et la dispersion des décalages trouvés par carte (une
grande dispersion suggère que l'hypothèse "rotation pure" ne suffit pas
complètement — perspective, échelle légèrement différente...).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import cv2

ANGLE_SEARCH_RANGE = range(-40, 41)  # degrés, recherche large : pas d'hypothèse de signe
PAD_FACTOR = 1.0  # marge ajoutée avant rotation, en fraction de la taille du template board


def _rotate_padded(img: Image.Image, angle: float, pad_w: int, pad_h: int) -> Image.Image:
    """Ajoute une marge noire (transitoire, jamais dans le résultat final) puis tourne autour du centre."""
    w, h = img.size
    padded = Image.new("RGB", (w + 2 * pad_w, h + 2 * pad_h), (0, 0, 0))
    padded.paste(img, (pad_w, pad_h))
    return padded.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0))


def _rotate_mask(size: Tuple[int, int], angle: float, pad_w: int, pad_h: int) -> Image.Image:
    """
    Même transformation que _rotate_padded, mais sur un masque plein
    blanc (= 'ici il y a de la carte') pour ensuite ne coller que ces
    pixels-là sur un fond réel. Interpolation BILINEAR (pas NEAREST)
    pour un bord anti-crénelé, plus fidèle à un vrai rendu.
    """
    w, h = size
    mask = Image.new("L", (w, h), 255)
    padded_mask = Image.new("L", (w + 2 * pad_w, h + 2 * pad_h), 0)
    padded_mask.paste(mask, (pad_w, pad_h))
    return padded_mask.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=0)


def _best_offset(big: Image.Image, target: Image.Image) -> Optional[Tuple[int, int, float]]:
    """
    Trouve où, dans `big` (plus grand), `target` se superpose le mieux —
    cv2.matchTemplate teste implicitement TOUTES les positions en un
    seul appel, donc la position (dx, dy) est retrouvée sans recherche
    séparée. None si `big` est trop petit pour contenir `target` (pad
    insuffisant pour cet angle).
    """
    big_arr = np.asarray(big.convert("RGB"))
    target_arr = np.asarray(target.convert("RGB"))
    if big_arr.shape[0] < target_arr.shape[0] or big_arr.shape[1] < target_arr.shape[1]:
        return None
    result = cv2.matchTemplate(big_arr, target_arr, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_loc[0], max_loc[1], max_val


def calibrate(
    board_dir: Path, hero_dir: Path, pad_factor: float = PAD_FACTOR
) -> Tuple[int, int, int, Dict[str, float]]:
    """
    Détermine (angle, dx, dy) à partir des cartes présentes à la fois
    dans board_dir et hero_dir (tes vrais exemples déjà collectés).

    Retourne (angle, dx, dy, scores_par_carte) — scores_par_carte sert
    de diagnostic (cf. docstring de module).
    """
    board_cards = {p.stem for p in board_dir.glob("*.png")}
    hero_cards = {p.stem for p in hero_dir.glob("*.png")}
    calibration_cards = sorted(board_cards & hero_cards)

    if len(calibration_cards) < 5:
        raise ValueError(
            f"Seulement {len(calibration_cards)} carte(s) en commun entre "
            f"{board_dir} et {hero_dir} — trop peu pour calibrer de façon "
            "fiable (5 minimum recommandé)."
        )

    pairs = [
        (card, Image.open(board_dir / f"{card}.png").convert("RGB"),
         Image.open(hero_dir / f"{card}.png").convert("RGB"))
        for card in calibration_cards
    ]

    # Phase 1 : meilleur angle GLOBAL (moyenne des scores sur toutes les
    # cartes de calibrage, pas juste la meilleure carte individuelle).
    board_w, board_h = pairs[0][1].size
    pad_w, pad_h = int(board_w * pad_factor), int(board_h * pad_factor)

    angle_scores: Dict[int, float] = {}
    for angle in ANGLE_SEARCH_RANGE:
        scores = []
        for _, board_img, hero_img in pairs:
            big = _rotate_padded(board_img, angle, pad_w, pad_h)
            r = _best_offset(big, hero_img)
            if r is not None:
                scores.append(r[2])
        if scores:
            angle_scores[angle] = sum(scores) / len(scores)

    if not angle_scores:
        raise ValueError("Aucun angle testé n'a produit de correspondance — pad_factor trop faible ?")

    best_angle = max(angle_scores, key=angle_scores.get)

    # Phase 2 : décalage (dx, dy) précis à cet angle, agrégé par médiane
    # (robuste à une carte de calibrage occasionnellement mauvaise).
    offsets: List[Tuple[int, int]] = []
    per_card_scores: Dict[str, float] = {}
    for card, board_img, hero_img in pairs:
        big = _rotate_padded(board_img, best_angle, pad_w, pad_h)
        r = _best_offset(big, hero_img)
        if r is None:
            continue
        x, y, score = r
        offsets.append((x, y))
        per_card_scores[card] = score

    xs = sorted(o[0] for o in offsets)
    ys = sorted(o[1] for o in offsets)
    median_x, median_y = xs[len(xs) // 2], ys[len(ys) // 2]

    print(f"Angle retenu : {best_angle}° (score moyen {angle_scores[best_angle]:.3f})")
    print(f"Décalage retenu : dx={median_x}, dy={median_y}")
    x_spread = xs[-1] - xs[0]
    y_spread = ys[-1] - ys[0]
    print(f"Dispersion des décalages trouvés par carte : x±{x_spread}px, y±{y_spread}px "
          f"({'faible, bon signe' if max(x_spread, y_spread) <= 4 else 'notable, à vérifier visuellement'})")
    worst_cards = sorted(per_card_scores, key=per_card_scores.get)[:3]
    print("Cartes de calibrage les moins bien alignées (à vérifier si le résultat final déçoit) :")
    for card in worst_cards:
        print(f"  {card} : score {per_card_scores[card]:.3f}")

    return best_angle, median_x, median_y, per_card_scores


def synthesize_missing(
    board_dir: Path,
    hero_dir: Path,
    output_dir: Path,
    angle: int,
    dx: int,
    dy: int,
    pad_factor: float = PAD_FACTOR,
) -> List[str]:
    """
    Génère, dans output_dir, une version synthétique de chaque carte
    présente dans board_dir mais absente de hero_dir. Le fond vient d'un
    exemple RÉEL de hero_dir (pas de couleur inventée) ; seule la carte
    elle-même vient de la rotation du template board.
    """
    board_cards = {p.stem for p in board_dir.glob("*.png")}
    hero_cards = {p.stem for p in hero_dir.glob("*.png")}
    missing = sorted(board_cards - hero_cards)

    if not missing:
        print("Rien à générer : toutes les cartes de board sont déjà présentes dans la banque hero.")
        return []

    background_ref_path = next(iter(hero_dir.glob("*.png")))
    background_ref = Image.open(background_ref_path).convert("RGB")
    hero_size = background_ref.size

    output_dir.mkdir(parents=True, exist_ok=True)

    for card in missing:
        board_img = Image.open(board_dir / f"{card}.png").convert("RGB")
        pad_w, pad_h = int(board_img.width * pad_factor), int(board_img.height * pad_factor)

        big = _rotate_padded(board_img, angle, pad_w, pad_h)
        big_mask = _rotate_mask(board_img.size, angle, pad_w, pad_h)

        hw, hh = hero_size
        if dx + hw > big.width or dy + hh > big.height:
            print(f"  ! {card} : décalage hors limites (pad_factor trop faible pour cet angle), ignorée.")
            continue

        aligned_card = big.crop((dx, dy, dx + hw, dy + hh))
        aligned_mask = big_mask.crop((dx, dy, dx + hw, dy + hh))

        result = background_ref.copy()
        result.paste(aligned_card, (0, 0), mask=aligned_mask)
        result.save(output_dir / f"{card}.png")

    print(f"{len(missing)} carte(s) générée(s) dans {output_dir}.")
    return missing


def run(board_dir: str, hero_dir: str, output_dir: str) -> None:
    board_path, hero_path, out_path = Path(board_dir), Path(hero_dir), Path(output_dir)

    print(f"Calibrage à partir des cartes communes à {board_dir} et {hero_dir}...")
    angle, dx, dy, _ = calibrate(board_path, hero_path)

    print()
    missing = synthesize_missing(board_path, hero_path, out_path, angle, dx, dy)

    if missing:
        print(f"\nVérifie les {len(missing)} images générées dans {output_dir} avant de les copier "
              f"dans {hero_dir} — ex: python -m ocr.vision.template_sorter {output_dir} {hero_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage : python -m ocr.vision.generate_hero_templates "
              "<board_dir> <hero_dir_reel> <output_dir_a_verifier>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2], sys.argv[3])
