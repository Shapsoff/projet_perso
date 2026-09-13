"""
ocr/vision/ — Localisation, capture et découpage de la table de poker
Phase 7 — Bot Poker Académique

  window_anchor.py     — localise la fenêtre du client (indépendant de sa
                          POSITION à l'écran).
  region_config.py      — zones calibrées en fractions 0..1 de la fenêtre
                          (indépendant de sa TAILLE).
  frame_capture.py      — combine les deux : capture + découpe en régions,
                          prêtes pour un lecteur (template matching /
                          OCR — à construire une fois les assets du
                          client cible disponibles, cf. card_reader.py /
                          text_reader.py non encore présents ici).
  capture_reference.py  — script CLI : sauvegarde UNE capture de la zone
                          cliente, à utiliser comme image de référence
                          pour la calibration.
  calibration_tool.py   — script CLI interactif (clic-glisser via
                          cv2.selectROI) : dessine les zones sur l'image
                          de référence, exporte un regions.json.

Ces deux derniers scripts nécessitent un affichage réel (ils ne
tournent pas dans un environnement headless) et pywin32 pour
capture_reference.py (Windows uniquement) — à exécuter en local, sur la
machine où tourne le client de poker, pas dans ce conteneur de
développement.
"""
