"""
ocr/ — Phase 7 : reconstruction d'un GameState depuis la table de poker

Ce package ne contient aucune capture d'écran ni reconnaissance d'image
(template matching, OCR de texte). Il définit et implémente tout ce qui
se trouve EN AVAL de la vision brute :

  vision_types.py       — contrat de données que la couche vision (à
                           construire séparément, une fois les assets du
                           client cible disponibles) doit produire à
                           chaque frame.
  validation.py          — sanity checks + lissage temporel (FrameStabilizer)
                           avant d'accepter une lecture comme fiable.
  seating.py             — attribution des positions (BTN/SB/BB/...) à
                           partir des sièges physiques et du bouton.
  player_identity.py     — résolution d'une identité stable par siège
                           pour la Player DB (phase 6), robuste aux
                           rotations de joueurs et aux erreurs de lecture
                           ponctuelles du pseudo.
  action_inference.py    — déduction des actions jouées (fold/check/call/
                           raise/allin) par comparaison de deux lectures
                           successives de la table.
  state_builder.py       — orchestrateur : construit et maintient un
                           core.game_state.GameState au fil des frames.
  hand_lifecycle.py      — détection de fin de main + pont vers
                           core.player_db.hand_recorder.record_hand().

Le seul contrat avec le reste du projet est core.game_state.GameState :
tout module en aval (EHSBot, RangeBot, best_response_engine...) reçoit
exactement le même objet, qu'il vienne du simulateur (self-play) ou de
ce package (table réelle).
"""
