"""
main_loop_example.py — Comment tout s'articule : capture -> état -> décision
Phase 7 — Bot Poker Académique (exemple d'assemblage, pas un module testé)

Montre où et QUAND chaque pièce déjà construite intervient :

  ocr.vision (capture + régions)
      -> table_read_from_frame()      [MANQUANT : card_reader / text_reader,
                                        à construire une fois regions.json +
                                        templates/ disponibles]
      -> ocr.state_builder.ingest()   [déjà prêt — tourne à CHAQUE frame]
      -> is_to_act sur le siège hero  [déjà dans le modèle de données]
      -> ehs_bot.decide()             [déjà prêt (phase 5/6) — appelé UNE
                                        FOIS par tour, pas à chaque frame]
      -> exécuter_action()            [MANQUANT : clic sur le bon bouton
                                        du client, pas construit ici]

Ce fichier n'est pas testé unitairement comme le reste du package (cf.
tests/test_ocr_*.py) : il orchestre des pièces déjà testées séparément
avec un vrai client de poker en face, ce qui ne se reproduit pas dans
une suite de tests automatisée. Une fois table_read_from_frame() et
exécuter_action() écrits, ce fichier (ou une version qui en dérive)
devient la boucle réelle du bot.
"""

from __future__ import annotations

import time

from ocr.hand_lifecycle import HandLifecycleTracker
from ocr.state_builder import StateBuilderError, TableStateBuilder
from ocr.vision.frame_capture import MssFrameGrabber, TableFrame, capture_table_frame
from ocr.vision.region_config import RegionConfig
from ocr.vision.window_anchor import Win32WindowLocator
from ocr.vision_types import TableRead  # noqa: F401  (import pour la signature ci-dessous)

# from ocr.vision.card_reader import read_card              # à construire
# from ocr.vision.text_reader import read_amount, read_pseudo  # à construire
# from core.bots.ehs_bot import EHSBot
# from core.player_db.player_db import PlayerDB
# from core.player_db.hand_recorder import record_hand


def table_read_from_frame(frame: TableFrame) -> "TableRead":
    """
    Le seul maillon qui manque encore : appelle card_reader (template
    matching) sur les régions hero_card_*/board_card_*, text_reader (OCR)
    sur pot/stack/bet/pseudo, et une simple détection de couleur/luminosité
    sur dealer_button/to_act_indicator (pas besoin d'OCR pour un simple
    "cette zone est-elle surlignée ?"). Assemble le tout en TableRead.
    """
    raise NotImplementedError(
        "À construire une fois regions.json + templates/ disponibles."
    )


def run(title_substring: str, regions_path: str, poll_seconds: float = 0.3) -> None:
    locator = Win32WindowLocator(title_substring)
    grabber = MssFrameGrabber()
    regions = RegionConfig.from_json(regions_path)

    builder = TableStateBuilder(session_id=1)
    tracker = HandLifecycleTracker(builder)
    # bot = EHSBot()

    # Détection de FRONT montant, pas de niveau : sans ça, decide() serait
    # rappelé à chaque frame tant que c'est encore notre tour (pendant que
    # le clic est en cours d'exécution, par ex.) — une seule décision par
    # tour, déclenchée au moment précis où is_to_act passe à True.
    was_hero_to_act = False

    while True:
        frame = capture_table_frame(locator, grabber, regions)
        if frame is None:
            time.sleep(poll_seconds)
            continue

        table_read = table_read_from_frame(frame)

        if builder.current_state is None:
            try:
                builder.start_new_hand(table_read)
            except StateBuilderError:
                pass  # pas encore assez d'infos lisibles pour démarrer, on retente au tour suivant
            time.sleep(poll_seconds)
            continue

        finished = tracker.observe(table_read)
        if finished is not None:
            # record_hand(db, finished, finished.board, player_id_map, position_map, our_player_id)
            builder.start_new_hand(table_read)
            was_hero_to_act = False
            time.sleep(poll_seconds)
            continue

        # Tourne à CHAQUE frame, que ce soit notre tour ou non : c'est ce
        # qui construit l'historique des actions adverses en continu.
        builder.ingest(table_read)

        hero_seat = table_read.hero()
        is_hero_to_act_now = bool(hero_seat and hero_seat.is_to_act)

        if is_hero_to_act_now and not was_hero_to_act:
            state = builder.current_state
            # action = bot.decide(state)
            # exécuter_action(action)   # clic sur le bon bouton — à construire séparément
            print(f"[main d'hand {state.hand_id}] à nous de jouer — to_call={state.to_call}")

        was_hero_to_act = is_hero_to_act_now
        time.sleep(poll_seconds)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage : python -m ocr.main_loop_example \"motif fenêtre\" regions.json")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
