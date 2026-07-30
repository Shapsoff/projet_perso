"""
test_action_history.py — Tests unitaires du module Action History Bucket
Phase 3 — Bot Poker Académique

Placement : tests/
Commande   : depuis la racine du projet
    python tests/test_action_history.py
    pytest tests/test_action_history.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.action_history import (
    ActionBucket,
    ActionSignals,
    classify_history,
    classify_all_villains,
    get_bucket_info,
    get_aggregate_bucket,
    get_villain_bucket_from_state,
    extract_signals,
    BUCKET_DESCRIPTIONS,
    BUCKET_RANGE_WIDTH,
    BUCKET_EHS_MODIFIER,
)


# =============================================================================
# Helper
# =============================================================================

def make_history(*events):
    """Crée un historique depuis des tuples (street, player, action, amount)."""
    return [
        {'street': s, 'player': p, 'action': a, 'amount': amt}
        for s, p, a, amt in events
    ]


# =============================================================================
# Tests de classification
# =============================================================================

class TestClassification:

    def test_empty_history(self):
        assert classify_history([], 1) == ActionBucket.NO_ACTION

    def test_empty_list(self):
        assert classify_history([], 99) == ActionBucket.NO_ACTION

    def test_villain_not_in_history(self):
        h = make_history(('preflop', 0, 'raise', 60))
        assert classify_history(h, 1) == ActionBucket.NO_ACTION

    # ── Limp ────────────────────────────────────────────────────────────────
    def test_limp_passive(self):
        h = make_history(
            ('preflop', 1, 'limp',  10),
            ('flop',    1, 'check', 0),
            ('flop',    0, 'check', 0),
        )
        assert classify_history(h, 1) == ActionBucket.LIMP_PASSIVE

    def test_limp_call(self):
        h = make_history(
            ('preflop', 1, 'limp', 10),
            ('flop',    0, 'bet',  30),
            ('flop',    1, 'call', 30),
        )
        assert classify_history(h, 1) == ActionBucket.LIMP_CALL

    def test_limp_only_no_flop(self):
        h = make_history(('preflop', 1, 'limp', 10))
        assert classify_history(h, 1) == ActionBucket.LIMP_PASSIVE

    # ── Open / call preflop ──────────────────────────────────────────────────
    def test_open_call_passive(self):
        h = make_history(
            ('preflop', 1, 'raise', 60),
            ('preflop', 0, 'call',  60),
            ('flop',    1, 'check', 0),
            ('flop',    0, 'check', 0),
        )
        assert classify_history(h, 1) == ActionBucket.OPEN_CALL_PASSIVE

    def test_open_call_cbet_call_villain_opener(self):
        """Villain open et mise le flop."""
        h = make_history(
            ('preflop', 1, 'raise', 60),
            ('preflop', 0, 'call',  60),
            ('flop',    1, 'bet',   80),
            ('flop',    0, 'call',  80),
        )
        assert classify_history(h, 1) == ActionBucket.OPEN_CALL_CBET_CALL

    def test_open_call_cbet_call_villain_caller(self):
        """Villain call preflop et call la mise flop."""
        h = make_history(
            ('preflop', 0, 'raise', 60),
            ('preflop', 1, 'call',  60),
            ('flop',    0, 'bet',   80),
            ('flop',    1, 'call',  80),
        )
        assert classify_history(h, 1) == ActionBucket.OPEN_CALL_CBET_CALL

    def test_open_call_cbet_fold(self):
        """Villain call preflop puis fold face à une mise."""
        h = make_history(
            ('preflop', 0, 'raise', 60),
            ('preflop', 1, 'call',  60),
            ('flop',    0, 'bet',   80),
            ('flop',    1, 'fold',  0),
        )
        assert classify_history(h, 1) == ActionBucket.OPEN_CALL_CBET_FOLD

    # ── Check-raise ──────────────────────────────────────────────────────────
    def test_check_raise_flop(self):
        h = make_history(
            ('preflop', 1, 'raise', 60),
            ('preflop', 0, 'call',  60),
            ('flop',    1, 'check', 0),
            ('flop',    0, 'bet',   80),
            ('flop',    1, 'raise', 240),
        )
        assert classify_history(h, 1) == ActionBucket.CHECK_RAISE_FLOP

    def test_check_raise_turn(self):
        h = make_history(
            ('preflop', 1, 'raise', 60),
            ('preflop', 0, 'call',  60),
            ('flop',    1, 'bet',   60),
            ('flop',    0, 'call',  60),
            ('turn',    1, 'check', 0),
            ('turn',    0, 'bet',   120),
            ('turn',    1, 'raise', 360),
        )
        assert classify_history(h, 1) == ActionBucket.CHECK_RAISE_TURN

    def test_check_raise_priorite_sur_open_call(self):
        """Un check-raise doit primer sur le bucket open_call."""
        h = make_history(
            ('preflop', 1, 'raise', 60),
            ('preflop', 0, 'call',  60),
            ('flop',    1, 'check', 0),
            ('flop',    0, 'bet',   80),
            ('flop',    1, 'raise', 240),
        )
        result = classify_history(h, 1)
        assert result == ActionBucket.CHECK_RAISE_FLOP
        assert result != ActionBucket.OPEN_CALL_CBET_CALL

    # ── 3bet ────────────────────────────────────────────────────────────────
    def test_3bet_pot_passive(self):
        h = make_history(
            ('preflop', 0, 'raise', 60),
            ('preflop', 1, '3bet',  180),
            ('preflop', 0, 'call',  180),
            ('flop',    1, 'check', 0),
            ('flop',    0, 'check', 0),
        )
        assert classify_history(h, 1) == ActionBucket.THREBET_POT_PASSIVE

    def test_3bet_pot_bet(self):
        h = make_history(
            ('preflop', 0, 'raise', 60),
            ('preflop', 1, '3bet',  180),
            ('preflop', 0, 'call',  180),
            ('flop',    1, 'bet',   200),
            ('flop',    0, 'call',  200),
        )
        assert classify_history(h, 1) == ActionBucket.THREBET_POT_BET

    def test_3bet_pot_check_raise(self):
        h = make_history(
            ('preflop', 0, 'raise', 60),
            ('preflop', 1, '3bet',  180),
            ('preflop', 0, 'call',  180),
            ('flop',    1, 'check', 0),
            ('flop',    0, 'bet',   200),
            ('flop',    1, 'raise', 600),
        )
        assert classify_history(h, 1) == ActionBucket.THREBET_POT_CHECK_RAISE

    def test_3bet_no_flop_yet(self):
        """3bet sans flop → bucket THREBET_POT_BET par défaut."""
        h = make_history(
            ('preflop', 0, 'raise', 60),
            ('preflop', 1, '3bet',  180),
            ('preflop', 0, 'call',  180),
        )
        assert classify_history(h, 1) == ActionBucket.THREBET_POT_BET

    # ── 4bet ────────────────────────────────────────────────────────────────
    def test_4bet(self):
        h = make_history(
            ('preflop', 0, 'raise', 60),
            ('preflop', 1, '3bet',  180),
            ('preflop', 0, '4bet',  480),
            ('preflop', 1, 'call',  480),
        )
        assert classify_history(h, 0) == ActionBucket.SQUEEZE_4BET

    def test_4bet_villain_is_4bettor(self):
        h = make_history(
            ('preflop', 0, 'raise', 60),
            ('preflop', 1, '3bet',  180),
            ('preflop', 0, 'call',  180),
        )
        # villain 1 a 3bet → pas de 4bet → THREBET_POT_BET
        assert classify_history(h, 1) == ActionBucket.THREBET_POT_BET


# =============================================================================
# Tests des métadonnées
# =============================================================================

class TestBucketMetadata:

    def test_all_12_buckets_exist(self):
        assert len(ActionBucket) == 12

    def test_all_buckets_have_description(self):
        for bucket in ActionBucket:
            assert bucket in BUCKET_DESCRIPTIONS
            assert len(BUCKET_DESCRIPTIONS[bucket]) > 10

    def test_all_buckets_have_range_width(self):
        for bucket in ActionBucket:
            assert bucket in BUCKET_RANGE_WIDTH
            w = BUCKET_RANGE_WIDTH[bucket]
            assert 0.0 < w <= 1.0, f"range_width hors bornes pour {bucket}: {w}"

    def test_all_buckets_have_ehs_modifier(self):
        for bucket in ActionBucket:
            assert bucket in BUCKET_EHS_MODIFIER
            m = BUCKET_EHS_MODIFIER[bucket]
            assert -0.20 <= m <= 0.20, f"ehs_modifier hors bornes pour {bucket}: {m}"

    def test_range_width_ordering(self):
        """Les hands fortes ont une range plus étroite."""
        assert BUCKET_RANGE_WIDTH[ActionBucket.SQUEEZE_4BET] < \
               BUCKET_RANGE_WIDTH[ActionBucket.OPEN_CALL_CBET_CALL]
        assert BUCKET_RANGE_WIDTH[ActionBucket.THREBET_POT_BET] < \
               BUCKET_RANGE_WIDTH[ActionBucket.LIMP_PASSIVE]
        assert BUCKET_RANGE_WIDTH[ActionBucket.NO_ACTION] == 1.00

    def test_ehs_modifier_ordering(self):
        """Les buckets forts augmentent le seuil EHS."""
        assert BUCKET_EHS_MODIFIER[ActionBucket.SQUEEZE_4BET] > \
               BUCKET_EHS_MODIFIER[ActionBucket.NO_ACTION]
        assert BUCKET_EHS_MODIFIER[ActionBucket.LIMP_PASSIVE] < \
               BUCKET_EHS_MODIFIER[ActionBucket.CHECK_RAISE_FLOP]
        assert BUCKET_EHS_MODIFIER[ActionBucket.LIMP_PASSIVE] < 0

    def test_get_bucket_info_keys(self):
        info = get_bucket_info(ActionBucket.CHECK_RAISE_FLOP)
        for key in ('bucket', 'name', 'description', 'range_width', 'ehs_modifier'):
            assert key in info, f"Clé manquante : {key}"

    def test_get_bucket_info_values(self):
        info = get_bucket_info(ActionBucket.SQUEEZE_4BET)
        assert info['bucket'] == ActionBucket.SQUEEZE_4BET
        assert info['name'] == "squeeze_4bet"
        assert info['range_width'] < 0.10
        assert info['ehs_modifier'] > 0.10


# =============================================================================
# Tests multiway
# =============================================================================

class TestMultiway:

    def test_classify_all_villains_basic(self):
        h = make_history(
            ('preflop', 1, 'raise', 60),
            ('preflop', 2, 'limp',  10),
        )
        buckets = classify_all_villains(h, [1, 2])
        assert 1 in buckets and 2 in buckets
        assert buckets[2] == ActionBucket.LIMP_PASSIVE

    def test_classify_all_empty_history(self):
        buckets = classify_all_villains([], [1, 2, 3])
        assert all(b == ActionBucket.NO_ACTION for b in buckets.values())

    def test_classify_all_empty_villains(self):
        buckets = classify_all_villains([], [])
        assert buckets == {}

    def test_get_aggregate_bucket_empty(self):
        assert get_aggregate_bucket({}) == ActionBucket.NO_ACTION

    def test_get_aggregate_bucket_single(self):
        assert get_aggregate_bucket({1: ActionBucket.CHECK_RAISE_FLOP}) == \
               ActionBucket.CHECK_RAISE_FLOP

    def test_get_aggregate_bucket_returns_most_threatening(self):
        buckets = {
            1: ActionBucket.LIMP_PASSIVE,
            2: ActionBucket.SQUEEZE_4BET,
            3: ActionBucket.OPEN_CALL_CBET_CALL,
        }
        assert get_aggregate_bucket(buckets) == ActionBucket.SQUEEZE_4BET

    def test_get_villain_bucket_from_state(self):
        gs = {
            'action_history': [
                {'street': 'preflop', 'player': 1, 'action': 'raise', 'amount': 60},
                {'street': 'preflop', 'player': 0, 'action': 'call',  'amount': 60},
                {'street': 'flop',    'player': 1, 'action': 'check', 'amount': 0},
                {'street': 'flop',    'player': 0, 'action': 'bet',   'amount': 80},
                {'street': 'flop',    'player': 1, 'action': 'raise', 'amount': 240},
            ],
            'players': [{'id': 1}, {'id': 2}],
        }
        buckets = get_villain_bucket_from_state(gs, our_id=0)
        assert 1 in buckets
        assert buckets[1] == ActionBucket.CHECK_RAISE_FLOP

    def test_get_villain_bucket_excludes_us(self):
        gs = {
            'action_history': [],
            'players': [{'id': 0}, {'id': 1}, {'id': 2}],
        }
        buckets = get_villain_bucket_from_state(gs, our_id=0)
        assert 0 not in buckets
        assert 1 in buckets and 2 in buckets


# =============================================================================
# Tests des signaux extraits
# =============================================================================

class TestSignalExtraction:

    def test_3bet_detected_via_keyword(self):
        h = make_history(
            ('preflop', 0, 'raise', 60),
            ('preflop', 1, '3bet',  180),
        )
        signals = extract_signals(h, villain_id=1)
        assert signals.villain_3bet is True
        assert signals.villain_raised_pf is False
        assert signals.villain_limped is False

    def test_4bet_detected(self):
        h = make_history(
            ('preflop', 0, 'raise', 60),
            ('preflop', 1, '3bet',  180),
            ('preflop', 0, '4bet',  480),
        )
        signals = extract_signals(h, villain_id=0)
        assert signals.villain_4bet is True

    def test_check_raise_flop_detected(self):
        h = make_history(
            ('flop', 1, 'check', 0),
            ('flop', 0, 'bet',   80),
            ('flop', 1, 'raise', 240),
        )
        signals = extract_signals(h, villain_id=1)
        assert signals.villain_raised_flop is True
        assert signals.villain_checked_flop is True
        assert signals.villain_bet_flop is False

    def test_check_raise_turn_detected(self):
        h = make_history(
            ('turn', 1, 'check', 0),
            ('turn', 0, 'bet',   100),
            ('turn', 1, 'raise', 300),
        )
        signals = extract_signals(h, villain_id=1)
        assert signals.villain_raised_turn is True
        assert signals.villain_checked_turn is True

    def test_saw_flop_turn_flags(self):
        h = make_history(
            ('preflop', 1, 'raise', 60),
            ('flop',    1, 'bet',   50),
            ('turn',    1, 'check', 0),
        )
        signals = extract_signals(h, villain_id=1)
        assert signals.saw_flop is True
        assert signals.saw_turn is True

    def test_villain_absent(self):
        h = make_history(('preflop', 0, 'raise', 60))
        signals = extract_signals(h, villain_id=1)
        assert signals.villain_raised_pf is False
        assert signals.villain_limped is False
        assert signals.villain_called_pf is False


# =============================================================================
# Runner standalone
# =============================================================================

def run_all_tests():
    test_classes = [
        TestClassification,
        TestBucketMetadata,
        TestMultiway,
        TestSignalExtraction,
    ]

    total_passed = 0
    total_failed = 0

    for cls in test_classes:
        print(f"\n{'─'*50}")
        print(f"  {cls.__name__}")
        print(f"{'─'*50}")
        instance = cls()
        methods  = sorted(m for m in dir(cls) if m.startswith('test_'))

        for method_name in methods:
            method = getattr(instance, method_name)
            try:
                method()
                print(f"  ✓ {method_name}")
                total_passed += 1
            except AssertionError as e:
                print(f"  ✗ {method_name} : {e}")
                total_failed += 1
            except Exception as e:
                print(f"  ✗ {method_name} : ERREUR {type(e).__name__}: {e}")
                total_failed += 1

    print(f"\n{'='*50}")
    print(f"  RÉSULTATS : {total_passed} passés | {total_failed} échoués")
    if total_failed == 0:
        print("  ✅ Tous les tests ActionHistoryBucket passent.")
    else:
        print(f"  ⚠ {total_failed} test(s) en échec.")
    print(f"{'='*50}\n")

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
