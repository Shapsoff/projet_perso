"""
test_range_bot.py — Tests unitaires des RangeBots
Phase 3 — Bot Poker Académique

Placement : tests/
Commande   : depuis la racine du projet
    python tests/test_range_bot.py
    pytest tests/test_range_bot.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.bots.range_definitions import (
    expand_hand_class,
    expand_range,
    hand_to_class,
    combo_in_range,
    get_range,
    get_all_configs,
    ARCHETYPES,
    MUTATIONS,
)
from core.bots.range_bot import (
    RangeBot,
    make_range_bot,
    make_all_range_bots,
    Action,
)


# =============================================================================
# Helpers
# =============================================================================

def make_gs(
    hand:    list  = None,
    board:   list  = None,
    street:  str   = 'preflop',
    pot:     float = 100,
    to_call: float = 0,
    stack:   float = 1000,
    bb:      float = 10,
    players: list  = None,
) -> dict:
    """Crée un game_state minimal pour les tests."""
    return {
        'hand':        hand    or ['Ah', 'Kd'],
        'board':       board   or [],
        'street':      street,
        'pot':         pot,
        'to_call':     to_call,
        'stack':       stack,
        'big_blind':   bb,
        'players':     players or [{'id': 1, 'stack': 900}],
        'action_history': [],
        'player_id':   0,
    }


VALID_ACTIONS = {'fold', 'check', 'call', 'bet', 'raise', 'allin'}


# =============================================================================
# Tests range_definitions
# =============================================================================

class TestRangeDefinitions:

    def test_expand_pair(self):
        assert len(expand_hand_class('AA')) == 6

    def test_expand_suited(self):
        assert len(expand_hand_class('AKs')) == 4

    def test_expand_offsuit(self):
        assert len(expand_hand_class('AKo')) == 12

    def test_expand_both(self):
        assert len(expand_hand_class('AK')) == 16

    def test_expand_pair_plus(self):
        # TT+ = TT JJ QQ KK AA = 5 × 6 = 30
        assert len(expand_hand_class('TT+')) == 30

    def test_expand_suited_plus(self):
        # ATs+ = ATs AJs AQs AKs = 4 × 4 = 16
        assert len(expand_hand_class('ATs+')) == 16

    def test_hand_to_class_offsuit(self):
        assert hand_to_class('Ah', 'Kd') == 'AKo'

    def test_hand_to_class_suited(self):
        assert hand_to_class('Ah', 'Kh') == 'AKs'

    def test_hand_to_class_pair(self):
        assert hand_to_class('Ah', 'As') == 'AA'

    def test_hand_to_class_reversed(self):
        # Rang le plus haut en premier
        assert hand_to_class('Kd', 'Ah') == 'AKo'

    def test_combo_in_range_true(self):
        combos = get_range('TAG', 0).preflop_combos
        assert combo_in_range('Ah', 'Ks', combos)

    def test_combo_in_range_false(self):
        combos = get_range('TAG', 0).preflop_combos
        assert not combo_in_range('2h', '3d', combos)

    def test_combo_in_range_reversed(self):
        combos = get_range('TAG', 0).preflop_combos
        # Les deux ordres doivent fonctionner
        assert combo_in_range('Ks', 'Ah', combos)

    def test_9_configs(self):
        assert len(get_all_configs()) == 9

    def test_tag_tighter_than_lag(self):
        tag = get_range('TAG', 0).range_pct
        lag = get_range('LAG', 0).range_pct
        assert tag < lag

    def test_tag_tighter_than_calling(self):
        tag  = get_range('TAG', 0).range_pct
        call = get_range('CALLING_STATION', 0).range_pct
        assert tag < call

    def test_lag_mutations_increasing(self):
        p0 = get_range('LAG', 0).range_pct
        p1 = get_range('LAG', 1).range_pct
        p2 = get_range('LAG', 2).range_pct
        assert p0 <= p1 <= p2

    def test_calling_mutations_increasing(self):
        p0 = get_range('CALLING_STATION', 0).range_pct
        p1 = get_range('CALLING_STATION', 1).range_pct
        p2 = get_range('CALLING_STATION', 2).range_pct
        assert p0 <= p1 <= p2

    def test_all_range_pct_valid(self):
        for arch in ARCHETYPES:
            for mut in MUTATIONS:
                cfg = get_range(arch, mut)
                assert 0 < cfg.range_pct < 1, (
                    f"{arch} mut{mut} range_pct={cfg.range_pct} hors bornes"
                )

    def test_invalid_archetype_raises(self):
        try:
            get_range('UNKNOWN', 0)
            assert False, "Doit lever ValueError"
        except ValueError:
            pass

    def test_invalid_mutation_raises(self):
        try:
            get_range('TAG', 5)
            assert False, "Doit lever ValueError"
        except ValueError:
            pass


# =============================================================================
# Tests RangeBot — preflop
# =============================================================================

class TestRangeBotPreflop:

    def setup_method(self):
        self.tag  = make_range_bot('TAG',  mutation=0)
        self.lag  = make_range_bot('LAG',  mutation=0)
        self.call = make_range_bot('CALLING_STATION', mutation=0)

    def test_tag_raises_with_aa(self):
        gs = make_gs(hand=['Ah', 'As'], street='preflop', to_call=0)
        action = self.tag.decide(gs)
        assert action.action_type == 'raise', f"AA doit → raise, obtenu {action}"

    def test_tag_folds_72o(self):
        gs = make_gs(hand=['7h', '2d'], street='preflop', to_call=20)
        action = self.tag.decide(gs)
        assert action.action_type == 'fold', f"72o doit → fold pour TAG"

    def test_tag_raises_ak(self):
        gs = make_gs(hand=['Ah', 'Kd'], street='preflop', to_call=0)
        action = self.tag.decide(gs)
        assert action.action_type == 'raise'

    def test_tag_folds_small_pairs(self):
        # TAG base : TT+ donc 22-99 sont fold
        gs = make_gs(hand=['9h', '9d'], street='preflop', to_call=20)
        action = self.tag.decide(gs)
        assert action.action_type == 'fold', "99 doit → fold pour TAG base (TT+)"

    def test_lag_plays_small_pairs(self):
        # LAG joue 22+
        gs = make_gs(hand=['2h', '2d'], street='preflop', to_call=0)
        action = self.lag.decide(gs)
        assert action.action_type == 'raise', "22 doit → raise pour LAG"

    def test_lag_plays_suited_connectors(self):
        gs = make_gs(hand=['Jh', 'Th'], street='preflop', to_call=0)
        action = self.lag.decide(gs)
        assert action.action_type == 'raise', "JTs doit → raise pour LAG"

    def test_lag_folds_out_of_range(self):
        # LAG base ne joue pas 72o
        gs = make_gs(hand=['7h', '2d'], street='preflop', to_call=20)
        action = self.lag.decide(gs)
        assert action.action_type == 'fold'

    def test_calling_station_plays_wide(self):
        # Calling Station joue 22+
        gs = make_gs(hand=['3h', '3d'], street='preflop', to_call=0)
        action = self.call.decide(gs)
        assert action.action_type == 'raise'

    def test_spr_too_low_allin(self):
        # SPR < 2 → allin avec main dans range
        gs = make_gs(hand=['Ah', 'As'], street='preflop',
                     stack=80, pot=200, to_call=0,
                     players=[{'id': 1, 'stack': 80}])
        action = self.tag.decide(gs)
        assert action.action_type == 'allin'

    def test_fold_out_of_range_no_allin(self):
        # Main hors range + SPR bas → fold quand même
        gs = make_gs(hand=['7h', '2d'], street='preflop',
                     stack=80, pot=200, to_call=50,
                     players=[{'id': 1, 'stack': 80}])
        action = self.tag.decide(gs)
        assert action.action_type == 'fold'

    def test_valid_action_types_preflop(self):
        for arch in ARCHETYPES:
            bot = make_range_bot(arch, 0)
            for hand in [['Ah', 'As'], ['7h', '2d'], ['Kh', 'Qs']]:
                gs = make_gs(hand=hand, street='preflop')
                action = bot.decide(gs)
                assert action.action_type in VALID_ACTIONS, (
                    f"{arch} action invalide : {action.action_type}"
                )

    def test_raise_amount_positive(self):
        gs = make_gs(hand=['Ah', 'As'], street='preflop', to_call=0)
        action = self.tag.decide(gs)
        assert action.amount > 0

    def test_raise_amount_within_stack(self):
        gs = make_gs(hand=['Ah', 'As'], street='preflop', stack=500)
        action = self.tag.decide(gs)
        assert action.amount <= 500


# =============================================================================
# Tests RangeBot — postflop
# =============================================================================

class TestRangeBotPostflop:

    def setup_method(self):
        self.tag  = make_range_bot('TAG',  mutation=0)
        self.lag  = make_range_bot('LAG',  mutation=0)
        self.call = make_range_bot('CALLING_STATION', mutation=0)

    def test_valid_action_postflop(self):
        """Toute décision postflop doit retourner un type valide."""
        for arch in ARCHETYPES:
            bot = make_range_bot(arch, 0)
            gs = make_gs(
                hand=['Ah', 'Kd'],
                board=['As', '7h', '2c'],
                street='flop',
            )
            action = bot.decide(gs)
            assert action.action_type in VALID_ACTIONS

    def test_spr_too_low_postflop_aggressive(self):
        """SPR < 2 + profil agressif → allin."""
        gs = make_gs(
            hand=['Ah', 'Kd'], board=['As', '7h', '2c'],
            street='flop', stack=80, pot=200,
            players=[{'id': 1, 'stack': 80}],
        )
        action = self.tag.decide(gs)
        assert action.action_type == 'allin'

    def test_spr_too_low_postflop_passive_with_bet(self):
        """SPR < 2 + calling station + face à une mise → call."""
        gs = make_gs(
            hand=['Ah', 'Kd'], board=['As', '7h', '2c'],
            street='flop', stack=80, pot=200, to_call=80,
            players=[{'id': 1, 'stack': 80}],
        )
        action = self.call.decide(gs)
        assert action.action_type in ('call', 'allin')

    def test_no_negative_amounts(self):
        """Les montants ne doivent jamais être négatifs."""
        for arch in ARCHETYPES:
            bot = make_range_bot(arch, 0)
            for street in ['flop', 'turn', 'river']:
                board = {
                    'flop':  ['As', '7h', '2c'],
                    'turn':  ['As', '7h', '2c', 'Kd'],
                    'river': ['As', '7h', '2c', 'Kd', '5s'],
                }[street]
                gs = make_gs(hand=['Ah', 'Kd'], board=board, street=street)
                action = bot.decide(gs)
                assert action.amount >= 0, (
                    f"{arch} {street} amount négatif : {action.amount}"
                )

    def test_amount_within_stack(self):
        """Le montant ne peut pas dépasser le stack."""
        for arch in ARCHETYPES:
            bot = make_range_bot(arch, 0)
            gs = make_gs(
                hand=['Ah', 'Kd'], board=['As', '7h', '2c'],
                street='flop', stack=200,
            )
            action = bot.decide(gs)
            assert action.amount <= 200 + 0.01, (
                f"{arch} amount {action.amount} > stack 200"
            )

    def test_calling_station_calls_more_than_tag(self):
        """
        La Calling Station doit call plus souvent que le TAG.
        Sur une séquence de décisions, CALL doit avoir plus de calls.
        """
        hands_boards = [
            (['7h', '6h'], ['As', 'Kd', '2c']),  # draw faible vs board sec
            (['Th', '9h'], ['Jh', '8c', '2d']),  # draw fort
            (['2h', '2d'], ['As', 'Kd', 'Qc']),  # paire de 2 sur board fort
        ]
        tag_calls  = 0
        call_calls = 0
        for hand, board in hands_boards:
            gs = make_gs(hand=hand, board=board, street='flop', to_call=50)
            a_tag  = self.tag.decide(gs)
            a_call = self.call.decide(gs)
            if a_tag.action_type  == 'call': tag_calls  += 1
            if a_call.action_type == 'call': call_calls += 1

        assert call_calls >= tag_calls, (
            f"CALLING_STATION doit call >= TAG : {call_calls} vs {tag_calls}"
        )


# =============================================================================
# Tests factory
# =============================================================================

class TestFactory:

    def test_make_range_bot(self):
        bot = make_range_bot('TAG', mutation=0)
        assert isinstance(bot, RangeBot)
        assert bot.config.archetype == 'TAG'
        assert bot.config.mutation  == 0

    def test_make_all_range_bots(self):
        bots = make_all_range_bots()
        assert len(bots) == 9
        for key in ['TAG_0', 'TAG_1', 'TAG_2',
                    'LAG_0', 'LAG_1', 'LAG_2',
                    'CALLING_STATION_0', 'CALLING_STATION_1', 'CALLING_STATION_2']:
            assert key in bots, f"Clé manquante : {key}"

    def test_all_bots_are_rangebot(self):
        bots = make_all_range_bots()
        for key, bot in bots.items():
            assert isinstance(bot, RangeBot), f"{key} n'est pas un RangeBot"

    def test_repr(self):
        bot = make_range_bot('TAG', 0)
        r = repr(bot)
        assert 'TAG' in r and 'RangeBot' in r

    def test_reset_no_crash(self):
        bot = make_range_bot('LAG', 1)
        bot.reset()

    def test_get_action_format(self):
        bot = make_range_bot('TAG', 0)
        gs = make_gs(hand=['Ah', 'As'], street='preflop')
        result = bot.get_action(gs)
        assert 'action' in result and 'amount' in result
        assert isinstance(result['action'], str)
        assert isinstance(result['amount'], (int, float))

    def test_n_sims_propagated(self):
        bot = make_range_bot('TAG', 0, n_sims=500)
        assert bot.n_sims == 500

    def test_invalid_archetype_raises(self):
        try:
            make_range_bot('UNKNOWN', 0)
            assert False
        except ValueError:
            pass


# =============================================================================
# Tests de cohérence entre archétypes
# =============================================================================

class TestArchetypeCoherence:

    def test_tag_plays_fewer_hands_than_lag(self):
        """TAG doit jouer moins de mains que LAG sur un grand échantillon."""
        tag = make_range_bot('TAG', 0)
        lag = make_range_bot('LAG', 0)

        import random
        random.seed(42)
        ranks  = '23456789TJQKA'
        suits  = 'cdhs'
        cards  = [r + s for r in ranks for s in suits]

        tag_plays  = 0
        lag_plays  = 0
        n_hands    = 500

        for _ in range(n_hands):
            hand = random.sample(cards, 2)
            gs   = make_gs(hand=hand, street='preflop', to_call=20)
            if tag.decide(gs).action_type != 'fold': tag_plays  += 1
            if lag.decide(gs).action_type != 'fold': lag_plays  += 1

        assert tag_plays < lag_plays, (
            f"TAG doit jouer moins de mains : TAG={tag_plays} LAG={lag_plays}"
        )

    def test_mutations_expand_range(self):
        """Mutation 1 et 2 ont plus de combos que mutation 0 (sauf TAG)."""
        # On vérifie directement les combos définis, pas une simulation
        # (la simulation avec stub EHS=0.5 crée de la variance artificielle)
        for arch in ['LAG', 'CALLING_STATION']:
            c0 = len(get_range(arch, 0).preflop_combos)
            c1 = len(get_range(arch, 1).preflop_combos)
            c2 = len(get_range(arch, 2).preflop_combos)
            assert c0 <= c1 <= c2, (
                f"{arch} mutations non croissantes : {c0} {c1} {c2}"
            )

    def test_calling_station_folds_less_than_tag(self):
        """Calling Station doit fold moins souvent que TAG."""
        import random
        random.seed(42)
        ranks = '23456789TJQKA'
        suits = 'cdhs'
        cards = [r + s for r in ranks for s in suits]

        tag  = make_range_bot('TAG', 0)
        call = make_range_bot('CALLING_STATION', 0)

        tag_folds  = 0
        call_folds = 0

        for _ in range(500):
            hand = random.sample(cards, 2)
            gs   = make_gs(hand=hand, street='preflop', to_call=20)
            if tag.decide(gs).action_type  == 'fold': tag_folds  += 1
            if call.decide(gs).action_type == 'fold': call_folds += 1

        assert call_folds < tag_folds, (
            f"CALLING_STATION doit fold moins que TAG : {call_folds} vs {tag_folds}"
        )


# =============================================================================
# Runner standalone
# =============================================================================

def run_all_tests():
    test_classes = [
        TestRangeDefinitions,
        TestRangeBotPreflop,
        TestRangeBotPostflop,
        TestFactory,
        TestArchetypeCoherence,
    ]

    total_passed = 0
    total_failed = 0

    for cls in test_classes:
        print(f"\n{'─'*55}")
        print(f"  {cls.__name__}")
        print(f"{'─'*55}")
        instance = cls()
        methods  = sorted(m for m in dir(cls) if m.startswith('test_'))

        for method_name in methods:
            if hasattr(instance, 'setup_method'):
                instance.setup_method()
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

    print(f"\n{'='*55}")
    print(f"  RÉSULTATS : {total_passed} passés | {total_failed} échoués")
    if total_failed == 0:
        print("  ✅ Tous les tests RangeBot passent.")
    else:
        print(f"  ⚠ {total_failed} test(s) en échec.")
    print(f"{'='*55}\n")

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
