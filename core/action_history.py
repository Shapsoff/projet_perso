"""
action_history.py — Dimension 2 du Bucketing Dynamique : Historique d'Actions
Phase 3 — Bot Poker Académique

Encode la séquence d'actions observées (preflop + streets précédentes) en un
bucket parmi 12 catégories. Ce bucket alimente deux usages :
  1. EHSBot v2 : modulation des seuils de décision selon le contexte
  2. Range Estimator (phase 4) : initialisation des priors bayésiens

Format de l'action_history attendu (section 4.4 du document) :
  [
    {'street': 'preflop', 'player': 1, 'action': 'raise',  'amount': 60},
    {'street': 'preflop', 'player': 2, 'action': 'call',   'amount': 60},
    {'street': 'flop',    'player': 2, 'action': 'check',  'amount': 0},
    {'street': 'flop',    'player': 1, 'action': 'bet',    'amount': 80},
    {'street': 'flop',    'player': 2, 'action': 'call',   'amount': 80},
  ]

Les 12 buckets sont définis par la combinaison des signaux preflop et postflop
les plus informatifs pour la range adverse.

Usage:
    from core.action_history import classify_history, ActionBucket
    bucket = classify_history(game_state['action_history'], villain_id=1)
    print(bucket)  # ActionBucket.CHECK_RAISE_FLOP
"""

from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass, field


# =============================================================================
# Définition des 12 buckets
# =============================================================================

class ActionBucket(Enum):
    """
    12 buckets d'historique couvrant les situations les plus informationnelles
    pour l'estimation de la range adverse.

    Notation :
      Preflop : R=raise/open, 3b=3bet, 4b=4bet, C=call, L=limp
      Postflop: b=bet, c=check, x=call, r=raise/check-raise, f=fold
    """
    # Pas d'action observée
    NO_ACTION               = "no_action"

    # Buckets passifs (limp)
    LIMP_PASSIVE            = "limp_passive"           # L / c-c
    LIMP_CALL               = "limp_call"              # L / b-x

    # Buckets standard (open ou call preflop)
    OPEN_CALL_PASSIVE       = "open_call_passive"      # R ou RC / c-c
    OPEN_CALL_CBET_CALL     = "open_call_cbet_call"    # R ou RC / b-x
    OPEN_CALL_CBET_FOLD     = "open_call_cbet_fold"    # R ou RC / b-f

    # Buckets 3bet pot
    THREBET_POT_PASSIVE     = "3bet_pot_passive"       # R-3b-C / c-c
    THREBET_POT_BET         = "3bet_pot_bet"           # R-3b-C / b-x
    THREBET_POT_CHECK_RAISE = "3bet_pot_check_raise"   # R-3b-C / c-r

    # Buckets check-raise
    CHECK_RAISE_FLOP        = "check_raise_flop"       # / c-r (flop)
    CHECK_RAISE_TURN        = "check_raise_turn"       # / c-r (turn)

    # Bucket premium extrême
    SQUEEZE_4BET            = "squeeze_4bet"           # 4bet ou squeeze


BUCKET_DESCRIPTIONS: Dict[ActionBucket, str] = {
    ActionBucket.NO_ACTION:
        "Aucune action observée — prior uniforme sur toutes les mains",
    ActionBucket.LIMP_PASSIVE:
        "Limp + check postflop — range wide et spéculative",
    ActionBucket.LIMP_CALL:
        "Limp + call cbet — range draws et paires moyennes",
    ActionBucket.OPEN_CALL_PASSIVE:
        "Open ou call preflop, double check postflop — range défensive top pair+",
    ActionBucket.OPEN_CALL_CBET_CALL:
        "Open ou call preflop, bet + call postflop — range large : top pair, draws, floats",
    ActionBucket.OPEN_CALL_CBET_FOLD:
        "Open ou call preflop, bet + fold postflop — range ATC preflop, capitule postflop",
    ActionBucket.THREBET_POT_PASSIVE:
        "3bet pot, check-check postflop — range polarisée qui slow-play ou abandonne",
    ActionBucket.THREBET_POT_BET:
        "3bet pot, cbet + call — range premium : AA, KK, AK, QQ",
    ActionBucket.THREBET_POT_CHECK_RAISE:
        "3bet pot, check-raise flop — range très forte ou bluff : sets, nut flush draws",
    ActionBucket.CHECK_RAISE_FLOP:
        "Check-raise flop — signal fort : sets, deux paires, nut draws",
    ActionBucket.CHECK_RAISE_TURN:
        "Check-raise turn tardif — range très forte ou semi-bluff draw",
    ActionBucket.SQUEEZE_4BET:
        "Squeeze ou 4bet — range premium extrême : AA, KK, parfois AK/QQ",
}

# Fraction approximative de combos dans la range (utilisée par Range Estimator phase 4)
BUCKET_RANGE_WIDTH: Dict[ActionBucket, float] = {
    ActionBucket.NO_ACTION:               1.00,
    ActionBucket.LIMP_PASSIVE:            0.40,
    ActionBucket.LIMP_CALL:               0.30,
    ActionBucket.OPEN_CALL_PASSIVE:       0.25,
    ActionBucket.OPEN_CALL_CBET_CALL:     0.30,
    ActionBucket.OPEN_CALL_CBET_FOLD:     0.40,
    ActionBucket.THREBET_POT_PASSIVE:     0.12,
    ActionBucket.THREBET_POT_BET:         0.08,
    ActionBucket.THREBET_POT_CHECK_RAISE: 0.06,
    ActionBucket.CHECK_RAISE_FLOP:        0.10,
    ActionBucket.CHECK_RAISE_TURN:        0.07,
    ActionBucket.SQUEEZE_4BET:            0.04,
}

# Modificateur sur le seuil EHS de l'EHSBot v2
# Positif = adversaire fort → on est plus défensif
# Négatif = adversaire faible → on est plus agressif
BUCKET_EHS_MODIFIER: Dict[ActionBucket, float] = {
    ActionBucket.NO_ACTION:               0.00,
    ActionBucket.LIMP_PASSIVE:           -0.05,
    ActionBucket.LIMP_CALL:              -0.03,
    ActionBucket.OPEN_CALL_PASSIVE:       0.00,
    ActionBucket.OPEN_CALL_CBET_CALL:     0.02,
    ActionBucket.OPEN_CALL_CBET_FOLD:    -0.02,
    ActionBucket.THREBET_POT_PASSIVE:     0.05,
    ActionBucket.THREBET_POT_BET:         0.08,
    ActionBucket.THREBET_POT_CHECK_RAISE: 0.10,
    ActionBucket.CHECK_RAISE_FLOP:        0.07,
    ActionBucket.CHECK_RAISE_TURN:        0.09,
    ActionBucket.SQUEEZE_4BET:            0.12,
}


# =============================================================================
# Extraction des signaux
# =============================================================================

@dataclass
class ActionSignals:
    """Signaux binaires extraits de l'historique pour la classification."""

    # Preflop
    villain_limped:    bool = False
    villain_raised_pf: bool = False   # open ou re-raise (hors 3bet/4bet)
    villain_3bet:      bool = False
    villain_4bet:      bool = False
    villain_called_pf: bool = False   # call d'un raise adverse preflop

    # Postflop — flop
    villain_checked_flop: bool = False
    villain_bet_flop:     bool = False   # bet en premier
    villain_called_flop:  bool = False
    villain_raised_flop:  bool = False   # check-raise ou raise d'un cbet
    villain_folded_flop:  bool = False

    # Postflop — turn
    villain_checked_turn: bool = False
    villain_bet_turn:     bool = False
    villain_called_turn:  bool = False
    villain_raised_turn:  bool = False
    villain_folded_turn:  bool = False

    # Flags globaux
    saw_flop: bool = False
    saw_turn: bool = False

    # Agressions preflop vues (pour détecter 3bet/4bet via raises successifs)
    _pf_raise_count: int = 0


def extract_signals(
    action_history: List[Dict],
    villain_id: int,
) -> ActionSignals:
    """
    Extrait les signaux binaires de l'historique pour le villain.

    Gère correctement les deux rôles du villain :
      - Villain comme agresseur (opener, 3betteur...)
      - Villain comme appelant (caller d'un open, caller d'un 3bet...)

    Args:
        action_history : liste de dicts {'street', 'player', 'action', 'amount'}
        villain_id     : player_id de l'adversaire à analyser

    Returns:
        ActionSignals remplis.
    """
    signals = ActionSignals()
    streets_seen = set()

    # Compter les raises preflop globaux (tous joueurs) pour positionner
    # le villain dans la séquence (opener, 3betteur, 4betteur)
    pf_raise_count_before_villain: Dict[int, int] = {}
    global_pf_raise_count = 0

    for event in action_history:
        street = event.get('street', '').lower()
        player = event.get('player', -1)
        action = event.get('action', '').lower()

        streets_seen.add(street)

        # Mise forcée (SB/BB) : jamais une décision volontaire, exclue
        # explicitement de tout comptage et de toute extraction de signal.
        # Sans cette exclusion, les 2 blindes de chaque main comptaient
        # comme des "raises" globaux (elles portaient ActionType.RAISE
        # avant le correctif POST_BLIND, cf. game_state.py et
        # simulator.py::_post_blindes) — une simple ouverture du villain
        # se retrouvait alors classée SQUEEZE_4BET (le bucket le plus
        # extrême des 12) puisque n_before comptait à tort les 2 blindes
        # comme des raises précédents. Bug trouvé et corrigé en session 6.
        if action == 'post_blind':
            continue

        # ── Compter les raises preflop globaux ───────────────────────────────
        if street == 'preflop' and action in ('raise', 'open', '3bet', '4bet'):
            if player == villain_id:
                # On mémorise combien de raises avaient eu lieu avant ce villain
                pf_raise_count_before_villain[global_pf_raise_count] = True
            global_pf_raise_count += 1

        if player != villain_id:
            continue

        # ── Signaux preflop ──────────────────────────────────────────────────
        if street == 'preflop':
            if action in ('raise', 'open'):
                # Déterminer si c'est un open, 3bet ou 4bet
                n_before = global_pf_raise_count - 1  # avant CE raise
                if n_before == 0:
                    signals.villain_raised_pf = True    # open raise
                elif n_before == 1:
                    signals.villain_3bet = True
                else:
                    signals.villain_4bet = True

            elif action == '3bet':
                signals.villain_3bet = True

            elif action == '4bet':
                signals.villain_4bet = True

            elif action == 'limp':
                signals.villain_limped = True

            elif action == 'call':
                signals.villain_called_pf = True

        # ── Signaux flop ─────────────────────────────────────────────────────
        elif street == 'flop':
            if action == 'check':
                signals.villain_checked_flop = True
            elif action in ('bet', 'raise'):
                # Si le villain avait checké avant → check-raise
                if signals.villain_checked_flop:
                    signals.villain_raised_flop = True
                else:
                    signals.villain_bet_flop = True
            elif action == 'call':
                signals.villain_called_flop = True
            elif action == 'fold':
                signals.villain_folded_flop = True

        # ── Signaux turn ──────────────────────────────────────────────────────
        elif street == 'turn':
            if action == 'check':
                signals.villain_checked_turn = True
            elif action in ('bet', 'raise'):
                if signals.villain_checked_turn:
                    signals.villain_raised_turn = True
                else:
                    signals.villain_bet_turn = True
            elif action == 'call':
                signals.villain_called_turn = True
            elif action == 'fold':
                signals.villain_folded_turn = True

    signals.saw_flop = 'flop' in streets_seen
    signals.saw_turn = 'turn' in streets_seen

    return signals


# =============================================================================
# Classificateur principal
# =============================================================================

def classify_signals(signals: ActionSignals) -> ActionBucket:
    """
    Mappe les signaux vers un des 12 buckets.
    Ordre de priorité : signal le plus fort en premier.
    """

    # ── 4bet / squeeze ────────────────────────────────────────────────────────
    if signals.villain_4bet:
        return ActionBucket.SQUEEZE_4BET

    # ── 3bet pot ─────────────────────────────────────────────────────────────
    if signals.villain_3bet:
        if signals.villain_raised_flop:
            return ActionBucket.THREBET_POT_CHECK_RAISE
        if signals.saw_flop:
            if signals.villain_bet_flop:
                return ActionBucket.THREBET_POT_BET
            if signals.villain_checked_flop:
                return ActionBucket.THREBET_POT_PASSIVE
        return ActionBucket.THREBET_POT_BET  # 3bet sans flop vu

    # ── Check-raise postflop ─────────────────────────────────────────────────
    if signals.villain_raised_turn:
        return ActionBucket.CHECK_RAISE_TURN
    if signals.villain_raised_flop:
        return ActionBucket.CHECK_RAISE_FLOP

    # ── Limp ─────────────────────────────────────────────────────────────────
    if signals.villain_limped:
        if signals.villain_called_flop or signals.villain_bet_flop:
            return ActionBucket.LIMP_CALL
        return ActionBucket.LIMP_PASSIVE

    # ── Open raise ou call preflop ────────────────────────────────────────────
    # Les deux cas (villain opener / villain caller) mènent aux mêmes buckets
    # postflop — ce qui différencie c'est le comportement sur le flop
    if signals.villain_raised_pf or signals.villain_called_pf:
        if signals.villain_folded_flop:
            return ActionBucket.OPEN_CALL_CBET_FOLD
        if signals.villain_called_flop or signals.villain_bet_flop:
            return ActionBucket.OPEN_CALL_CBET_CALL
        if signals.villain_checked_flop:
            return ActionBucket.OPEN_CALL_PASSIVE
        # Preflop uniquement, pas encore de flop
        return ActionBucket.OPEN_CALL_PASSIVE

    return ActionBucket.NO_ACTION


def classify_history(
    action_history: List[Dict],
    villain_id: int,
) -> ActionBucket:
    """
    Point d'entrée principal.
    Classifie l'historique d'actions d'un adversaire en un des 12 buckets.

    Args:
        action_history : historique complet de la main
        villain_id     : player_id de l'adversaire à analyser

    Returns:
        ActionBucket correspondant au profil observé.

    Exemple:
        history = [
            {'street': 'preflop', 'player': 1, 'action': 'raise',  'amount': 60},
            {'street': 'preflop', 'player': 0, 'action': 'call',   'amount': 60},
            {'street': 'flop',    'player': 1, 'action': 'check',  'amount': 0},
            {'street': 'flop',    'player': 0, 'action': 'bet',    'amount': 80},
            {'street': 'flop',    'player': 1, 'action': 'raise',  'amount': 240},
        ]
        bucket = classify_history(history, villain_id=1)
        # → ActionBucket.CHECK_RAISE_FLOP
    """
    if not action_history:
        return ActionBucket.NO_ACTION

    signals = extract_signals(action_history, villain_id)
    return classify_signals(signals)


# =============================================================================
# Utilitaires
# =============================================================================

def get_bucket_info(bucket: ActionBucket) -> Dict:
    """Retourne toutes les métadonnées d'un bucket."""
    return {
        'bucket':       bucket,
        'name':         bucket.value,
        'description':  BUCKET_DESCRIPTIONS[bucket],
        'range_width':  BUCKET_RANGE_WIDTH[bucket],
        'ehs_modifier': BUCKET_EHS_MODIFIER[bucket],
    }


def classify_all_villains(
    action_history: List[Dict],
    villain_ids: List[int],
) -> Dict[int, ActionBucket]:
    """Classifie tous les adversaires d'une main. Utile en multiway."""
    return {
        vid: classify_history(action_history, vid)
        for vid in villain_ids
    }


def get_villain_bucket_from_state(
    game_state: Dict,
    our_id: int = 0,
) -> Dict[int, ActionBucket]:
    """
    Extrait et classifie tous les adversaires actifs depuis un game_state normalisé.
    """
    history     = game_state.get('action_history', [])
    players     = game_state.get('players', [])
    villain_ids = [p['id'] for p in players if p['id'] != our_id]
    return classify_all_villains(history, villain_ids)


def get_aggregate_bucket(buckets: Dict[int, ActionBucket]) -> ActionBucket:
    """
    En multiway, retourne le bucket le plus menaçant parmi tous les adversaires.
    (celui avec le modificateur EHS le plus élevé)
    """
    if not buckets:
        return ActionBucket.NO_ACTION
    return max(buckets.values(), key=lambda b: BUCKET_EHS_MODIFIER[b])


# =============================================================================
# Tests auto (python action_history.py)
# =============================================================================

if __name__ == "__main__":
    def make_history(*events):
        return [{'street': s, 'player': p, 'action': a, 'amount': amt}
                for s, p, a, amt in events]

    tests = [
        ("Historique vide",
         [], 1, ActionBucket.NO_ACTION),
        ("Limp + double check",
         make_history(('preflop',1,'limp',10),('flop',1,'check',0),('flop',0,'check',0)),
         1, ActionBucket.LIMP_PASSIVE),
        ("Limp + call cbet",
         make_history(('preflop',1,'limp',10),('flop',0,'bet',30),('flop',1,'call',30)),
         1, ActionBucket.LIMP_CALL),
        ("Open + call pf, double check",
         make_history(('preflop',1,'raise',60),('preflop',0,'call',60),('flop',1,'check',0),('flop',0,'check',0)),
         1, ActionBucket.OPEN_CALL_PASSIVE),
        ("Open + call pf, cbet + call",
         make_history(('preflop',1,'raise',60),('preflop',0,'call',60),('flop',1,'bet',80),('flop',0,'call',80)),
         1, ActionBucket.OPEN_CALL_CBET_CALL),
        ("Call pf, face cbet + fold",
         make_history(('preflop',0,'raise',60),('preflop',1,'call',60),('flop',0,'bet',80),('flop',1,'fold',0)),
         1, ActionBucket.OPEN_CALL_CBET_FOLD),
        ("Check-raise flop",
         make_history(('preflop',1,'raise',60),('preflop',0,'call',60),('flop',1,'check',0),('flop',0,'bet',80),('flop',1,'raise',240)),
         1, ActionBucket.CHECK_RAISE_FLOP),
        ("Check-raise turn",
         make_history(('preflop',1,'raise',60),('preflop',0,'call',60),('flop',1,'bet',60),('flop',0,'call',60),('turn',1,'check',0),('turn',0,'bet',120),('turn',1,'raise',360)),
         1, ActionBucket.CHECK_RAISE_TURN),
        ("3bet pot, cbet + call",
         make_history(('preflop',0,'raise',60),('preflop',1,'3bet',180),('preflop',0,'call',180),('flop',1,'bet',200),('flop',0,'call',200)),
         1, ActionBucket.THREBET_POT_BET),
        ("3bet pot, check-check",
         make_history(('preflop',0,'raise',60),('preflop',1,'3bet',180),('preflop',0,'call',180),('flop',1,'check',0),('flop',0,'check',0)),
         1, ActionBucket.THREBET_POT_PASSIVE),
        ("3bet pot, check-raise flop",
         make_history(('preflop',0,'raise',60),('preflop',1,'3bet',180),('preflop',0,'call',180),('flop',1,'check',0),('flop',0,'bet',200),('flop',1,'raise',600)),
         1, ActionBucket.THREBET_POT_CHECK_RAISE),
        ("4bet",
         make_history(('preflop',0,'raise',60),('preflop',1,'3bet',180),('preflop',0,'4bet',480),('preflop',1,'call',480)),
         0, ActionBucket.SQUEEZE_4BET),
        ("Villain absent de l'historique",
         make_history(('preflop',0,'raise',60)),
         1, ActionBucket.NO_ACTION),

        # ── Régression session 6 : les blindes (post_blind) ne doivent JAMAIS
        # être comptées comme des raises. Avant le correctif, ces 2 tests
        # cassaient : une simple ouverture (tag générique 'raise', ce que les
        # vrais bots produisent TOUJOURS — ils ne taguent jamais explicitement
        # '3bet'/'4bet') se retrouvait classée SQUEEZE_4BET à cause des 2
        # blindes comptées comme raises précédents.
        ("[post_blind] Open (tag générique 'raise') après les blindes — PAS un 4bet",
         make_history(('preflop',1,'post_blind',5),('preflop',0,'post_blind',10),
                      ('preflop',1,'raise',30),('preflop',0,'call',30),
                      ('flop',1,'check',0),('flop',0,'check',0)),
         1, ActionBucket.OPEN_CALL_PASSIVE),
        ("[post_blind] Vrai 3bet (tag générique 'raise') après les blindes reste un 3bet",
         make_history(('preflop',1,'post_blind',5),('preflop',0,'post_blind',10),
                      ('preflop',0,'raise',30),('preflop',1,'raise',90),
                      ('preflop',0,'call',90),
                      ('flop',1,'bet',100),('flop',0,'call',100)),
         1, ActionBucket.THREBET_POT_BET),
        ("[post_blind] Vrai 4bet (tag générique 'raise') après les blindes reste un 4bet",
         make_history(('preflop',1,'post_blind',5),('preflop',0,'post_blind',10),
                      ('preflop',1,'raise',30),('preflop',0,'raise',90),
                      ('preflop',1,'raise',270)),
         1, ActionBucket.SQUEEZE_4BET),
    ]

    passed = failed = 0
    print("\n=== Tests action_history.py ===\n")
    for desc, history, vid, expected in tests:
        result = classify_history(history, vid)
        if result == expected:
            print(f"  ✓ {desc}")
            passed += 1
        else:
            print(f"  ✗ {desc}")
            print(f"      Attendu : {expected.value}")
            print(f"      Obtenu  : {result.value}")
            failed += 1

    print(f"\n  {passed}/{passed+failed} tests passés")
    print("  ✅ OK\n" if failed == 0 else f"  ⚠ {failed} échec(s)\n")
