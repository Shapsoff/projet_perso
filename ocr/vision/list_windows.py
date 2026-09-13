"""
list_windows.py — Liste les titres de toutes les fenêtres visibles
Phase 7 — Bot Poker Académique 

À exécuter EN LOCAL (pywin32, Windows uniquement — ne fonctionne pas
dans un environnement headless) pour trouver le bon title_substring à
utiliser avec Win32WindowLocator / capture_reference.py.

Usage :
    1. Ouvrir une table sur le client de poker.
    2. python -m ocr.vision.list_windows
    3. Repérer, dans la liste imprimée, le titre exact de la fenêtre de
       table. Choisir comme title_substring un motif présent sur TOUTE
       table de ce client (ex: "CoinPoker"), pas un nom de table précis
       (ex: PAS "CoinPoker - Table Aurora") — sinon il faudrait retrouver
       un nouveau motif à chaque nouvelle table ouverte.
"""

from __future__ import annotations

from typing import List


def list_visible_window_titles() -> List[str]:
    """
    Isolé de run() pour rester la seule partie qui dépend réellement de
    pywin32 — run() n'ajoute que la mise en forme de la sortie.
    """
    import win32gui

    titles: List[str] = []

    def _callback(hwnd: int, _extra: object) -> None:
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title.strip():
                titles.append(title)

    win32gui.EnumWindows(_callback, None)
    return titles


def run() -> None:
    titles = list_visible_window_titles()
    print(f"{len(titles)} fenêtres visibles :\n")
    for title in titles:
        print(f"  - {title!r}")


if __name__ == "__main__":
    run()
