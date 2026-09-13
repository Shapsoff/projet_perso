"""
frame_capture.py — Capture d'écran de la fenêtre cible + découpage en régions
Phase 7 — Bot Poker Académique

Combine window_anchor.py (où est la fenêtre, maintenant) et
region_config.py (quelles zones lire, en %) pour produire, à chaque
appel, un TableFrame : image brute + toutes les régions déjà recadrées,
prêtes à être passées aux lecteurs (card_reader.py par template matching,
text_reader.py par OCR — à construire une fois les assets du client
cible disponibles).

FrameGrabber est abstrait (Protocol) pour la même raison que
WindowLocator dans window_anchor.py : pouvoir tester tout le
découpage/normalisation sans dépendre d'un vrai écran (mss ne fonctionne
pas dans un conteneur sans affichage, ni pendant l'exécution des tests
automatisés).

Deux implémentations, deux garanties différentes :
  MssFrameGrabber        — capture une ZONE D'ÉCRAN (mss). Rapide, mais
                            capture "ce qu'il y a affiché à ces
                            coordonnées", peu importe QUELLE fenêtre s'y
                            trouve réellement — si une autre fenêtre
                            recouvre la table au moment de l'appel,
                            c'est ELLE qui est capturée, silencieusement.
  PrintWindowFrameGrabber — capture le CONTENU de la fenêtre elle-même
                            (API Windows PrintWindow), correct même si
                            elle est recouverte. Plus lent (rendu GDI à
                            chaque appel).

grab() reçoit maintenant un hwnd optionnel EN PLUS de bbox : bbox reste
nécessaire pour MssFrameGrabber (coordonnées écran) et pour connaître la
zone visée en général, hwnd est nécessaire pour PrintWindowFrameGrabber
(qui cible la fenêtre elle-même, pas une zone d'écran). Win32WindowLocator
expose le hwnd trouvé via l'attribut `last_hwnd` juste après
find_client_bbox() — capture_table_frame() le lit automatiquement.
"""

from __future__ import annotations

from typing import Dict, Optional, Protocol, Tuple

import numpy as np
from PIL import Image

from .region_config import PixelRect, RegionConfig
from .window_anchor import BoundingBox, WindowLocator


class FrameGrabber(Protocol):
    """Abstraction sur la source de capture (cf. MssFrameGrabber / PrintWindowFrameGrabber)."""
    def grab(self, bbox: BoundingBox, hwnd: Optional[int] = None) -> Image.Image: ...


class MssFrameGrabber:
    """
    Capture réelle via la librairie mss (rapide, adaptée au polling
    répété). Capture une ZONE D'ÉCRAN à des coordonnées fixes — si une
    autre fenêtre recouvre la table au moment de l'appel, c'est son
    contenu qui est capturé, pas celui de la table (cf.
    PrintWindowFrameGrabber si ça pose problème).

    hwnd est accepté pour respecter le Protocol FrameGrabber mais
    ignoré : mss n'a aucun moyen de cibler une fenêtre précise
    indépendamment de sa position à l'écran.
    """

    def grab(self, bbox: BoundingBox, hwnd: Optional[int] = None) -> Image.Image:
        import mss

        with mss.mss() as sct:
            shot = sct.grab(bbox.as_mss_dict())
            return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")


class PrintWindowFrameGrabber:
    """
    Capture directement le CONTENU de la fenêtre (API Windows
    PrintWindow) plutôt qu'une zone d'écran — demande à LA FENÊTRE
    ELLE-MÊME de se redessiner dans un buffer hors écran. Résultat
    correct même si une autre fenêtre recouvre partiellement ou
    entièrement la table au moment de la capture, contrairement à
    MssFrameGrabber.

    Contrepartie : nettement plus lent (rendu GDI à chaque appel,
    potentiellement dizaines à centaines de ms selon la complexité de la
    fenêtre) — un bon compromis pour un polling lent (ex:
    template_collector.py, ~1 capture/s), à reconsidérer pour la boucle
    de décision temps réel si la latence devient gênante.

    Nécessite hwnd — lève une erreur explicite si absent plutôt que de
    deviner (ex: avec StaticWindowLocator, utilisé pour les tests, qui
    n'a pas de vraie fenêtre à cibler).

    scale_factor compense un éventuel écart entre les coordonnées
    LOGIQUES retournées par GetWindowRect et la résolution PHYSIQUE
    réellement disponible quand le processus appelant n'est pas
    DPI-aware (ex: scaling Windows à 125% -> souvent besoin de 1.25).
    Pas de valeur universelle : dépend du scaling Windows de LA machine
    sur laquelle ça tourne. capture_table_frame() se base ensuite sur la
    taille RÉELLE de l'image renvoyée (pas sur bbox) pour résoudre les
    régions, donc une valeur de scale_factor pas encore parfaitement
    réglée ne fait "que" perdre en netteté/précision, pas décaler les
    régions (cf. capture_table_frame()).
    """

    def __init__(self, scale_factor: float = 1.0) -> None:
        self.scale_factor = scale_factor

    def grab(self, bbox: BoundingBox, hwnd: Optional[int] = None) -> Image.Image:
        if hwnd is None:
            raise ValueError(
                "PrintWindowFrameGrabber nécessite un hwnd — le WindowLocator "
                "utilisé doit l'exposer (cf. Win32WindowLocator.last_hwnd, "
                "rempli après find_client_bbox()). Impossible avec "
                "StaticWindowLocator, qui n'a pas de vraie fenêtre."
            )

        import ctypes
        from ctypes import wintypes

        import win32gui
        import win32ui

        user32 = ctypes.windll.user32
        user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
        user32.PrintWindow.restype = wintypes.BOOL
        pw_render_full_content = 2

        rect = win32gui.GetWindowRect(hwnd)
        logical_width = rect[2] - rect[0]
        logical_height = rect[3] - rect[1]
        width = max(1, round(logical_width * self.scale_factor))
        height = max(1, round(logical_height * self.scale_factor))

        hwnd_dc = win32gui.GetWindowDC(hwnd)
        dc = win32ui.CreateDCFromHandle(hwnd_dc)
        mem_dc = dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(dc, width, height)
        mem_dc.SelectObject(bitmap)

        result = user32.PrintWindow(hwnd, mem_dc.GetSafeHdc(), pw_render_full_content)
        if not result:
            user32.PrintWindow(hwnd, mem_dc.GetSafeHdc(), 0)  # repli sans le flag

        bits = bitmap.GetBitmapBits(True)
        if not bits:
            raise RuntimeError("Impossible de récupérer les bits du bitmap (capture vide).")

        image = Image.frombuffer("RGB", (width, height), bits, "raw", "BGRX", 0, 1)

        dc.DeleteDC()
        mem_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
        win32gui.DeleteObject(bitmap.GetHandle())

        return image


def crop_region(frame: Image.Image, pixel_rect: PixelRect, frame_origin: BoundingBox) -> Image.Image:
    """
    Découpe `pixel_rect` (coordonnées ÉCRAN absolues, produites par
    RegionConfig.resolve/resolve_all) dans `frame` (image capturée dont
    le coin haut-gauche correspond à `frame_origin`). Les deux doivent
    partager la même origine — c'est le cas normal quand `frame` vient de
    grabber.grab(frame_origin).
    """
    left, top, width, height = pixel_rect
    rel_left = left - frame_origin.left
    rel_top = top - frame_origin.top
    return frame.crop((rel_left, rel_top, rel_left + width, rel_top + height))


def normalize_for_template_matching(
    region_image: Image.Image, canonical_size: Tuple[int, int],
) -> np.ndarray:
    """
    Redimensionne une région recadrée vers une taille CANONIQUE fixe
    avant comparaison à des templates de cartes.

    Nécessaire car les régions fractionnelles (region_config.py) donnent
    des rectangles dont la taille en PIXELS varie avec la taille de la
    fenêtre — une carte capturée sur une table redimensionnée à 1600px de
    large n'a pas le même nombre de pixels que sur une table à 1280px de
    large. Les templates de cartes, eux, sont capturés une fois à une
    taille fixe : ramener chaque lecture à cette même taille canonique
    avant cv2.matchTemplate est ce qui rend le template matching robuste
    au redimensionnement — exactement comme les FractionalRegion rendent
    le DÉCOUPAGE robuste au déplacement/redimensionnement.

    (Une version letterbox préservant le ratio d'origine a été testée
    pour réduire les confusions entre chiffres à courbes similaires —
    6/8/9/3 — mais n'a pas apporté d'amélioration mesurable en pratique.
    Revenu volontairement à ce resize simple : pas de complexité
    supplémentaire pour un gain qui ne s'est pas confirmé. cf.
    card_reader.py pour la vraie parade retenue face à ces confusions —
    correction manuelle au fil de l'eau plutôt qu'un ajustement
    algorithmique de plus.)
    """
    resized = region_image.convert("RGB").resize(canonical_size, Image.LANCZOS)
    return np.array(resized)


class TableFrame:
    """
    Une capture complète à un instant t : image brute + toutes les
    régions déjà recadrées.
    """

    def __init__(self, bbox: BoundingBox, raw: Image.Image, crops: Dict[str, Image.Image]) -> None:
        self.bbox = bbox
        self.raw = raw
        self.crops = crops

    def crop(self, name: str) -> Image.Image:
        return self.crops[name]


def capture_table_frame(
    locator: WindowLocator,
    grabber: FrameGrabber,
    regions: RegionConfig,
) -> Optional[TableFrame]:
    """
    Point d'entrée principal de ce module : localise la fenêtre, capture,
    découpe toutes les régions calibrées.

    Renvoie None si la fenêtre cible n'est pas trouvée à cet instant
    (fermée, minimisée) — à l'appelant de décider quoi faire (retenter,
    alerter, mettre le bot en pause), jamais de lever une exception pour
    un état par ailleurs normal (le client a été réduit un instant).

    Le hwnd trouvé par le locator (s'il en expose un, ex:
    Win32WindowLocator.last_hwnd) est transmis au grabber — nécessaire
    pour PrintWindowFrameGrabber, ignoré par MssFrameGrabber.

    Les régions sont résolues sur la taille RÉELLE de l'image capturée
    (raw.size), pas sur bbox.width/height : pour MssFrameGrabber les deux
    sont déjà identiques (ce recalcul est alors un no-op), mais un
    grabber qui compense un scaling DPI (cf. PrintWindowFrameGrabber)
    peut renvoyer une image à une résolution différente de bbox — sans
    ce recalcul, les régions calibrées seraient mal alignées.
    """
    bbox = locator.find_client_bbox()
    if bbox is None:
        return None

    hwnd = getattr(locator, "last_hwnd", None)
    raw = grabber.grab(bbox, hwnd)

    capture_bbox = BoundingBox(bbox.left, bbox.top, raw.width, raw.height)
    pixel_rects = regions.resolve_all(capture_bbox)
    crops = {name: crop_region(raw, rect, capture_bbox) for name, rect in pixel_rects.items()}
    return TableFrame(bbox=capture_bbox, raw=raw, crops=crops)
