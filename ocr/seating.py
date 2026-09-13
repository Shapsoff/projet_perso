"""
seating.py — Attribution des positions à partir du bouton
Phase 7 — Bot Poker Académique

core.game_state.Player attend une Position (BTN/SB/BB/UTG/MP/CO...), mais
la vision ne lit jamais directement "ce siège est en BTN" — elle lit
seulement quel siège porte le bouton (SeatRead.is_dealer) et quels
sièges sont occupés. Ce module retrouve les positions par la même
convention que core/simulator.py::PokerTable.POSITIONS_BY_N : l'ordre de
la liste est l'ordre des sièges en partant du bouton et en tournant dans
le sens de distribution des cartes (BTN, puis le siège suivant occupé,
etc.).

POSITIONS_BY_N est dupliqué ici plutôt qu'importé de core.simulator :
core/simulator.py importe poker_engine (moteur C++ compilé) et lève une
RuntimeError à l'import s'il est absent — une dépendance dure que ce
module (et tout le pipeline OCR en amont de la décision) n'a aucune
raison de porter. tests/test_ocr_seating.py contient un test de
synchronisation qui compare les deux tables dès que poker_engine est
disponible, pour détecter toute divergence si l'une des deux est
modifiée sans l'autre.
"""

from __future__ import annotations

from typing import Dict, List

from core.game_state import Position

POSITIONS_BY_N: Dict[int, List[Position]] = {
    2: [Position.BTN, Position.BB],
    3: [Position.BTN, Position.SB, Position.BB],
    4: [Position.BTN, Position.SB, Position.BB, Position.CO],
    5: [Position.BTN, Position.SB, Position.BB, Position.MP, Position.CO],
    6: [Position.BTN, Position.SB, Position.BB, Position.UTG, Position.MP, Position.CO],
}


def positions_from_button(
    occupied_seat_indices: List[int],
    dealer_seat_index: int,
) -> Dict[int, Position]:
    """
    Args:
        occupied_seat_indices : sièges physiques occupés (0..N-1),
                                 PAS nécessairement contigus (un joueur
                                 assis en siège 4 sur une table à 6
                                 sièges où les sièges 1 et 3 sont vides
                                 est courant en cashgame).
        dealer_seat_index     : siège qui porte le bouton.

    Returns:
        {seat_index: Position} pour chaque siège occupé.

    Raises:
        ValueError si le nombre de joueurs n'est pas dans 2..6, ou si
        dealer_seat_index n'est pas dans occupied_seat_indices.
    """
    n = len(occupied_seat_indices)
    if n not in POSITIONS_BY_N:
        raise ValueError(f"{n} joueurs occupés : seul 2 à 6 est supporté")
    if dealer_seat_index not in occupied_seat_indices:
        raise ValueError(
            f"dealer_seat_index={dealer_seat_index} n'est pas un siège occupé "
            f"({occupied_seat_indices})"
        )

    # Ordre des sièges en partant du bouton, dans le sens de la donne.
    # On suppose que les seat_index croissent dans le sens de la donne
    # (convention la plus courante côté clients de poker) et on boucle
    # avec modulo sur le plus grand seat_index + 1 pour ne pas supposer
    # une numérotation 0..N-1 sans trou.
    max_seat = max(occupied_seat_indices)
    ordered = sorted(occupied_seat_indices)
    start_idx = ordered.index(dealer_seat_index)
    rotated = ordered[start_idx:] + ordered[:start_idx]

    # `rotated` suit l'ordre croissant des seat_index à partir du bouton,
    # ce qui correspond au sens de distribution si et seulement si les
    # seat_index sont numérotés dans ce sens. C'est vrai pour tout client
    # de poker connu (numérotation stable et croissante autour de la
    # table) ; si ce n'était pas le cas, seul `rotated` serait à inverser.
    del max_seat  # conservé nommé pour la lisibilité du raisonnement ci-dessus

    positions = POSITIONS_BY_N[n]
    return {seat: pos for seat, pos in zip(rotated, positions)}
