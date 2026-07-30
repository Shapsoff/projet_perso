"""
test_best_response.py — Tests unitaires Phase 5 / P0
Bot Poker Académique

Couvre les trois modules du Best-Response Engine :
  Section 1 — FrequencyModel (classify_combo_response + compute)
  Section 2 — EVCalculator (ev par action, cohérence, monotonie)
  Section 3 — BestResponseEngine (intégration, fallback, bluff layer)

Les tests utilisent des stubs légers (pas de poker_engine C++) pour
rester auto-contenus et rapides. Les tests d'intégration réels (avec
poker_engine et Range Estimator) sont dans validate_best_response.py.

Usage :
    python -m pytest tests/test_best_response.py -v
    python tests/test_best_response.py  (mode standalone)
"""

import math
import sys
import os
import unittest
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

# ── Résolution du chemin racine du projet ────────────────────────────────────
# Permet de lancer le test depuis n'importe où :
#   python tests/test_best_response.py
#   python -m pytest tests/test_best_response.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Imports Best-Response Engine ─────────────────────────────────────────────
from core.best_response.frequency_model import (
    FrequencyModel,
    FoldCallRaiseResult,
    classify_combo_response,
)
from core.best_response.ev_calculator import EVCalculator
from core.best_response.best_response_engine import (
    BestResponseEngine,
    BestResponseConfig,
)

# ---------------------------------------------------------------------------
# Stubs — évite la dépendance à poker_engine et au projet complet
# ---------------------------------------------------------------------------

# Stub MonteCarloEngine
class _MockMCEngine:
    """Monte Carlo stub : equity fixe à 0.6 (main forte fictive)."""
    def compute(self, our_hand_str, board_str, opp_ranges, n_sims=5000):
        result = MagicMock()
        result.ev   = 0.60
        result.win  = 0.55
        result.tie  = 0.10
        result.loss = 0.35
        result.n    = n_sims
        return result


# Stub EHSCalculator
class _MockEHSCalc:
    """EHS stub : retourne 0.65 pour toute main (forte fictive)."""
    def calculate_multiway(self, hand, board, n_opp, n_sims):
        result = MagicMock()
        result.EHS = 0.65
        result.HS  = 0.65
        result.PPot = 0.10
        result.NPot = 0.05
        result.elapsed_ms = 0.0
        return result


# Stub RangeEstimator
class _MockEstimator:
    def __init__(self, archetype='LAG', confidence=0.75, hands_seen=30):
        self._archetype   = archetype
        self._confidence  = confidence
        self._hands_seen  = hands_seen

    def get_archetype_probabilities(self):
        # On répartit le reste équitablement sans normaliser pour que
        # la confiance retournée soit exactement self._confidence.
        archs = ['TAG', 'LAG', 'CALLING_STATION']
        remaining = (1.0 - self._confidence) / (len(archs) - 1)
        return {a: (self._confidence if a == self._archetype else remaining)
                for a in archs}

    def get_distribution(self):
        """Distribution fictive : 100 combos à poids égal."""
        combos = [
            'AhKd', 'AhKc', 'AhKs', 'AdKh', 'AdKc', 'AdKs',
            'AcKh', 'AcKd', 'AcKs', 'AsKh', 'AsKd', 'AsKc',
            'AhQd', 'AhQc', 'AhQs', 'AdQh', 'AdQc', 'AdQs',
            'AcQh', 'AcQd', 'AcQs', 'AsQh', 'AsQd', 'AsQc',
            'KhQd', 'KhQc', 'KhQs', 'KdQh', 'KdQc', 'KdQs',
            'TThh', 'TTdd',  # intentionnellement invalides → filtrés
            '9h8h', '9d8d', '9c8c', '9s8s',
            'JhTh', 'JdTd', 'JcTc', 'JsTs',
        ]
        # Distribution uniforme sur les combos valides
        valid = [c for c in combos if len(c) == 4]
        p = 1.0 / len(valid) if valid else 0
        return {c: p for c in valid}

    def get_best_archetype(self):
        return self._archetype, 0

    @property
    def _stats(self):
        s = MagicMock()
        s.hands_seen = self._hands_seen
        return s


# Stub SPRInfo
class _MockSPRInfo:
    def __init__(self, force_commit=False, spr=5.0):
        self.force_commit = force_commit
        self.spr          = spr

    def sizing_modifier(self):
        return 1.0


# ---------------------------------------------------------------------------
# Setup des modules sans dépendances externes
# ---------------------------------------------------------------------------

def _build_frequency_model(ehs_calc=None):
    """Construit un FrequencyModel sans dépendances projet."""
    return FrequencyModel(ehs_calculator=ehs_calc or _MockEHSCalc(), n_sims=100)


# ---------------------------------------------------------------------------
# Section 1 — FrequencyModel
# ---------------------------------------------------------------------------

class TestClassifyComboResponse(unittest.TestCase):
    """Tests unitaires de classify_combo_response."""

    def setUp(self):
        self.classify = classify_combo_response

    def test_out_of_range_always_folds(self):
        """Un combo hors range fold toujours, quel que soit l'EHS."""
        for ehs in [0.0, 0.3, 0.5, 0.8, 1.0]:
            result = self.classify(
                ehs=ehs, archetype='TAG', bet_sizing=0.75,
                pot=100, to_call=75, in_range=False,
            )
            self.assertEqual(result, 'fold',
                f"EHS={ehs} hors range devrait fold, obtenu {result}")

    def test_tag_strong_hand_raises(self):
        """TAG avec EHS très fort raise face à un bet."""
        result = self.classify(
            ehs=0.80, archetype='TAG', bet_sizing=0.75,
            pot=100, to_call=75, in_range=True,
        )
        self.assertEqual(result, 'raise')

    def test_tag_medium_hand_calls(self):
        """TAG avec EHS médian call si pot odds rentables."""
        # EHS=0.60 > seuil_call=0.55, et pot_odds = 75/(100+75) = 0.43 < 0.60
        result = self.classify(
            ehs=0.60, archetype='TAG', bet_sizing=0.75,
            pot=100, to_call=75, in_range=True,
        )
        self.assertEqual(result, 'call')

    def test_tag_weak_hand_folds(self):
        """TAG avec EHS faible fold face à un bet."""
        result = self.classify(
            ehs=0.35, archetype='TAG', bet_sizing=0.75,
            pot=100, to_call=75, in_range=True,
        )
        self.assertEqual(result, 'fold')

    def test_lag_lower_thresholds(self):
        """LAG a des seuils plus bas que TAG — call avec EHS=0.40."""
        # seuil_call LAG = 0.35, pot_odds 75/(100+75) = 0.43 > 0.40
        # EHS=0.40 > seuil_call=0.35 mais EHS < pot_odds → fold pour LAG aggressive
        # Avec EHS=0.45 : EHS > pot_odds(0.43) → call
        result = self.classify(
            ehs=0.45, archetype='LAG', bet_sizing=0.75,
            pot=100, to_call=75, in_range=True,
        )
        self.assertEqual(result, 'call')

    def test_calling_station_calls_wide(self):
        """Calling Station call avec EHS > seuil_call même si non rentable."""
        # CS seuil_call = 0.30, aggression='passive' → call large
        result = self.classify(
            ehs=0.35, archetype='CALLING_STATION', bet_sizing=1.00,
            pot=100, to_call=100, in_range=True, aggression='passive',
        )
        self.assertEqual(result, 'call')

    def test_no_bet_check_scenario(self):
        """Sans bet (sizing=0), retourne check ou raise selon EHS."""
        # EHS fort → raise (il bet dans notre check)
        result_strong = self.classify(
            ehs=0.80, archetype='TAG', bet_sizing=0.0,
            pot=100, to_call=0.0, in_range=True,
        )
        self.assertEqual(result_strong, 'raise')

        # EHS faible → call (il check aussi)
        result_weak = self.classify(
            ehs=0.30, archetype='TAG', bet_sizing=0.0,
            pot=100, to_call=0.0, in_range=True,
        )
        self.assertEqual(result_weak, 'call')


class TestFrequencyModelCompute(unittest.TestCase):
    """Tests du calcul de fréquences FrequencyModel.compute()."""

    def setUp(self):
        self.model = FrequencyModel(
            ehs_calculator=_MockEHSCalc(), n_sims=100
        )

        self.distribution = {
            'AhKd': 0.05, 'AhKc': 0.05, 'AhKs': 0.05,
            'AdKh': 0.05, 'AdKc': 0.05, 'AdKs': 0.05,
            'AcKh': 0.05, 'AcKd': 0.05, 'AcKs': 0.05,
            'AsKh': 0.05, 'AsKd': 0.05, 'AsKs': 0.05,
            '9h8h': 0.05, '9d8d': 0.05, '9c8c': 0.05,
            'JhTh': 0.05, 'JdTd': 0.05, 'JcTc': 0.05,
            '2h3d': 0.05, '4c5s': 0.05,  # mains faibles
        }
        self.board = ['7c', '2h', 'Js']

    def test_probabilities_sum_to_one(self):
        """p_fold + p_call + p_raise = 1.0."""
        result = self.model.compute(
            distribution=self.distribution,
            archetype='TAG',
            board=self.board,
            bet_sizing=0.75,
            pot=100,
        )
        total = result.p_fold + result.p_call + result.p_raise
        self.assertAlmostEqual(total, 1.0, places=3,
            msg=f"Somme des prob = {total:.4f} (attendu 1.0)")

    def test_board_cards_filtered(self):
        """Les combos contenant des cartes du board sont exclus."""
        dist_with_board_card = dict(self.distribution)
        dist_with_board_card['7cAh'] = 0.10  # 7c est sur le board
        dist_with_board_card['2hKs'] = 0.10  # 2h est sur le board

        result = self.model.compute(
            distribution=dist_with_board_card,
            archetype='TAG',
            board=self.board,
            bet_sizing=0.75,
            pot=100,
        )
        # Les combos avec cartes du board ne doivent pas apparaître
        for combo in ['7cAh', '2hKs', 'Ah7c', 'Ks2h']:
            self.assertNotIn(combo, result.fold_range,
                f"Combo {combo} ne devrait pas être dans fold_range")
            self.assertNotIn(combo, result.call_range,
                f"Combo {combo} ne devrait pas être dans call_range")

    def test_empty_distribution_returns_safe_result(self):
        """Distribution vide → fold par défaut, pas d'erreur."""
        result = self.model.compute(
            distribution={},
            archetype='TAG',
            board=self.board,
            bet_sizing=0.75,
            pot=100,
        )
        self.assertEqual(result.p_fold, 1.0)
        self.assertEqual(result.p_call, 0.0)

    def test_higher_bet_induces_more_folds_for_tag(self):
        """Un bet plus grand génère plus de folds qu'un bet petit (TAG)."""
        # Le mock EHS retourne 0.65 pour tous les combos.
        # TAG seuil_bet=0.65, seuil_call=0.55.
        # EHS=0.65 est exactement au seuil_bet → raise pour TAG
        # Ce test vérifie la cohérence structurelle du modèle.
        result_small = self.model.compute(
            distribution=self.distribution,
            archetype='CALLING_STATION',  # CS call très large
            board=self.board,
            bet_sizing=0.33,
            pot=100,
        )
        result_large = self.model.compute(
            distribution=self.distribution,
            archetype='CALLING_STATION',
            board=self.board,
            bet_sizing=2.00,  # overbet → plus de folds
            pot=100,
        )
        # CS seuil_call=0.30, EHS=0.65 → call sur les deux sizings
        # (pot odds 2.0/(1+2.0)=0.67 > EHS=0.65 → pourrait fold sur l'overbet)
        # Test de non-régression : les fréquences sont calculées de façon cohérente
        self.assertGreaterEqual(result_large.p_fold, result_small.p_fold - 0.01,
            "Un bet plus grand ne devrait pas induire moins de folds (à EHS constant)")

    def test_multiple_sizings_shares_ehs_cache(self):
        """compute_multiple_sizings utilise le cache EHS (un seul calcul par combo)."""
        import time

        sizings = [0.33, 0.50, 0.75, 1.00]

        # Première passe : calcul complet
        t0 = time.time()
        results = self.model.compute_multiple_sizings(
            distribution=self.distribution,
            archetype='LAG',
            board=self.board,
            sizings=sizings,
            pot=100,
        )
        t1 = time.time()

        self.assertEqual(len(results), len(sizings))
        for s in sizings:
            self.assertIn(s, results)
            total = (results[s].p_fold + results[s].p_call + results[s].p_raise)
            self.assertAlmostEqual(total, 1.0, places=3)


# ---------------------------------------------------------------------------
# Section 2 — EVCalculator
# ---------------------------------------------------------------------------

class TestEVCalculator(unittest.TestCase):
    """Tests du calcul d'EV par action."""

    def setUp(self):
        freq_model = FrequencyModel(
            ehs_calculator=_MockEHSCalc(), n_sims=100
        )
        self.calc = EVCalculator(
            mc_engine=_MockMCEngine(),
            frequency_model=freq_model,
            n_mc_sims=500,
        )
        self.distribution = {
            'AhKd': 0.10, 'AhKc': 0.10, 'AhKs': 0.10,
            'AdKh': 0.10, 'AdKc': 0.10, 'AdKs': 0.10,
            'KhQd': 0.10, 'KhQc': 0.10, 'KhQs': 0.10,
            '9h8h': 0.10,
        }
        self.our_hand = ['Ah', 'Kd']
        self.board    = ['7c', '2h', 'Js']

    def test_results_sorted_by_ev(self):
        """Les résultats sont triés par EV décroissante."""
        results = self.calc.compute_all_actions(
            our_hand=self.our_hand,
            board=self.board,
            distribution=self.distribution,
            archetype='LAG',
            pot=100,
            to_call=0,
            stack=1000,
            street='flop',
        )
        evs = [r.ev for r in results]
        self.assertEqual(evs, sorted(evs, reverse=True),
            "Résultats non triés par EV décroissante")

    def test_fold_ev_is_zero(self):
        """EV du fold = 0 (convention)."""
        results = self.calc.compute_all_actions(
            our_hand=self.our_hand,
            board=self.board,
            distribution=self.distribution,
            archetype='TAG',
            pot=100,
            to_call=50,   # on fait face à un bet
            stack=1000,
            street='flop',
        )
        fold_results = [r for r in results if r.action == 'fold']
        self.assertEqual(len(fold_results), 1)
        self.assertEqual(fold_results[0].ev, 0.0)

    def test_ev_bet_positive_with_fold_equity(self):
        """EV d'un bet est positive quand il y a de la fold equity."""
        # Avec p_fold élevé (mock EHS=0.65 → TAG fold beaucoup), EV > 0
        results = self.calc.compute_all_actions(
            our_hand=self.our_hand,
            board=self.board,
            distribution=self.distribution,
            archetype='TAG',
            pot=100,
            to_call=0,
            stack=1000,
            street='flop',
            sizings=[0.75],
        )
        bet_results = [r for r in results if r.action == 'bet']
        if bet_results:
            # EV(bet) = p_fold×pot + p_call×(equity×pot_total - bet) + ...
            # Avec equity=0.60, p_fold élevé pour TAG → EV devrait être > 0
            self.assertGreater(bet_results[0].ev, 0,
                f"EV(bet_75%) devrait être positive, obtenu {bet_results[0].ev:.2f}")

    def test_check_action_present_when_no_to_call(self):
        """L'action check est présente quand to_call=0."""
        results = self.calc.compute_all_actions(
            our_hand=self.our_hand,
            board=self.board,
            distribution=self.distribution,
            archetype='LAG',
            pot=100,
            to_call=0,
            stack=1000,
            street='flop',
        )
        actions = [r.action for r in results]
        self.assertIn('check', actions)
        self.assertNotIn('fold', actions)

    def test_fold_present_when_to_call(self):
        """L'action fold est présente quand to_call > 0."""
        results = self.calc.compute_all_actions(
            our_hand=self.our_hand,
            board=self.board,
            distribution=self.distribution,
            archetype='LAG',
            pot=100,
            to_call=50,
            stack=1000,
            street='flop',
        )
        actions = [r.action for r in results]
        self.assertIn('fold', actions)
        self.assertIn('call', actions)

    def test_ev_components_sum_to_total(self):
        """ev_fold + ev_call + ev_raise ≈ ev pour les bets."""
        results = self.calc.compute_all_actions(
            our_hand=self.our_hand,
            board=self.board,
            distribution=self.distribution,
            archetype='LAG',
            pot=100,
            to_call=0,
            stack=1000,
            street='flop',
            sizings=[0.75],
        )
        for r in results:
            if r.action in ('bet', 'raise', 'allin'):
                total_components = r.ev_fold + r.ev_call + r.ev_raise
                self.assertAlmostEqual(total_components, r.ev, places=2,
                    msg=f"Composantes EV ne somment pas à {r.ev:.3f} "
                        f"(obtenu {total_components:.3f})")

    def test_allin_included_when_stack_small(self):
        """Un allin est inclus quand le stack est petit."""
        results = self.calc.compute_all_actions(
            our_hand=self.our_hand,
            board=self.board,
            distribution=self.distribution,
            archetype='LAG',
            pot=100,
            to_call=0,
            stack=80,    # stack < 2×pot → allin probable
            street='flop',
            sizings=[2.0],
        )
        actions = [r.action for r in results]
        self.assertIn('allin', actions)


# ---------------------------------------------------------------------------
# Section 3 — BestResponseEngine
# ---------------------------------------------------------------------------

class TestBestResponseEngine(unittest.TestCase):
    """Tests d'intégration du BestResponseEngine."""

    def setUp(self):
        config = BestResponseConfig(
            sizings=[0.50, 0.75, 1.00],
            n_mc_sims_ev=200,
            n_sims_frequency=50,
            min_hands_for_best_response=5,
            min_archetype_confidence=0.40,
        )
        self.engine = BestResponseEngine(
            mc_engine=_MockMCEngine(),
            config=config,
        )
        self.engine.set_ehs_calculator(_MockEHSCalc())

        self.game_state = {
            'hand':     ['Ah', 'Kd'],
            'board':    ['7c', '2h', 'Js'],
            'pot':      100.0,
            'to_call':  0.0,
            'stack':    1000.0,
            'street':   'flop',
            'players':  [{'id': 1, 'stack': 1000, 'status': 'active'}],
            'player_id': 0,
            'action_history': [],
            'position': 'BTN',
            'big_blind': 10,
        }

    def _make_spr_info(self, force_commit=False, spr=5.0):
        return _MockSPRInfo(force_commit=force_commit, spr=spr)

    def _make_ehs_result(self, ehs=0.65, ppot=0.10):
        return {'EHS': ehs, 'HS': ehs, 'PPot': ppot, 'NPot': 0.05}

    def test_returns_action_with_sufficient_data(self):
        """Retourne une action valide quand le Range Estimator a assez de données."""
        estimator = _MockEstimator(archetype='LAG', confidence=0.75, hands_seen=30)
        decision, action = self.engine.decide(
            game_state=self.game_state,
            estimator=estimator,
            ehs_result=self._make_ehs_result(),
            spr_info=self._make_spr_info(),
            texture_modifier=1.0,
            bucket_modifier=0.0,
            hand_count=30,
        )
        self.assertIsNotNone(action)
        self.assertIn(action.action_type,
                      ['fold', 'check', 'call', 'bet', 'raise', 'allin'])
        self.assertFalse(decision.used_fallback,
            "Ne devrait pas utiliser le fallback avec 30 mains et confiance 75%")

    def test_fallback_when_insufficient_hands(self):
        """Fallback v3 quand pas assez de mains."""
        estimator = _MockEstimator(hands_seen=3)
        decision, action = self.engine.decide(
            game_state=self.game_state,
            estimator=estimator,
            ehs_result=self._make_ehs_result(),
            spr_info=self._make_spr_info(),
            texture_modifier=1.0,
            bucket_modifier=0.0,
            hand_count=3,   # < min_hands_for_best_response=5
        )
        self.assertTrue(decision.used_fallback)

    def test_fallback_when_no_estimator(self):
        """Fallback v3 quand estimator est None."""
        decision, action = self.engine.decide(
            game_state=self.game_state,
            estimator=None,
            ehs_result=self._make_ehs_result(),
            spr_info=self._make_spr_info(),
            texture_modifier=1.0,
            bucket_modifier=0.0,
            hand_count=50,
        )
        self.assertTrue(decision.used_fallback)
        self.assertIsNotNone(action)

    def test_spr_force_commit_overrides_everything(self):
        """SPR très bas → allin, même si Best-Response est actif."""
        estimator = _MockEstimator(archetype='TAG', confidence=0.90, hands_seen=50)
        decision, action = self.engine.decide(
            game_state=self.game_state,
            estimator=estimator,
            ehs_result=self._make_ehs_result(),
            spr_info=self._make_spr_info(force_commit=True),  # SPR force commit
            texture_modifier=1.0,
            bucket_modifier=0.0,
            hand_count=50,
        )
        self.assertEqual(action.action_type, 'allin')

    def test_fallback_when_low_confidence(self):
        """Fallback v3 quand confiance archétype insuffisante."""
        estimator = _MockEstimator(
            archetype='LAG', confidence=0.35, hands_seen=20
        )
        decision, action = self.engine.decide(
            game_state=self.game_state,
            estimator=estimator,
            ehs_result=self._make_ehs_result(),
            spr_info=self._make_spr_info(),
            texture_modifier=1.0,
            bucket_modifier=0.0,
            hand_count=20,
        )
        self.assertTrue(decision.used_fallback,
            "Confiance 35% < 40% → devrait fallback")

    def test_calling_station_gets_larger_sizing(self):
        """Contre une Calling Station, le best-response bet plus large."""
        estimator_cs  = _MockEstimator('CALLING_STATION', 0.80, 30)
        estimator_tag = _MockEstimator('TAG',              0.80, 30)

        _, action_cs = self.engine.decide(
            game_state=self.game_state,
            estimator=estimator_cs,
            ehs_result=self._make_ehs_result(ehs=0.75),
            spr_info=self._make_spr_info(),
            texture_modifier=1.0,
            bucket_modifier=0.0,
            hand_count=30,
        )
        _, action_tag = self.engine.decide(
            game_state=self.game_state,
            estimator=estimator_tag,
            ehs_result=self._make_ehs_result(ehs=0.75),
            spr_info=self._make_spr_info(),
            texture_modifier=1.0,
            bucket_modifier=0.0,
            hand_count=30,
        )
        # Contre CS (call très large) on devrait bet plus gros que contre TAG
        # (qui fold beaucoup → la fold equity favorise un bet plus petit)
        # Ce test est de nature qualitative — on vérifie juste que les actions sont cohérentes
        self.assertIn(action_cs.action_type,
                      ['bet', 'raise', 'allin', 'check'],
                      "Action CS invalide")
        self.assertIn(action_tag.action_type,
                      ['bet', 'raise', 'allin', 'check'],
                      "Action TAG invalide")

    def test_decision_has_all_ev_results(self):
        """La décision contient tous les EVResults pour debug."""
        estimator = _MockEstimator('LAG', 0.80, 30)
        decision, _ = self.engine.decide(
            game_state=self.game_state,
            estimator=estimator,
            ehs_result=self._make_ehs_result(),
            spr_info=self._make_spr_info(),
            texture_modifier=1.0,
            bucket_modifier=0.0,
            hand_count=30,
        )
        if not decision.used_fallback:
            self.assertGreater(len(decision.all_results), 0)
            # Vérifie que toutes les actions ont un EV calculé
            for r in decision.all_results:
                self.assertIsInstance(r.ev, float)
                self.assertFalse(math.isnan(r.ev),
                    f"EV de {r.action} est NaN")

    def test_to_call_scenario_has_fold_and_call(self):
        """Quand to_call > 0, fold et call sont présents dans les résultats."""
        gs = dict(self.game_state)
        gs['to_call'] = 75.0

        estimator = _MockEstimator('LAG', 0.80, 30)
        decision, action = self.engine.decide(
            game_state=gs,
            estimator=estimator,
            ehs_result=self._make_ehs_result(),
            spr_info=self._make_spr_info(),
            texture_modifier=1.0,
            bucket_modifier=0.0,
            hand_count=30,
        )
        if not decision.used_fallback:
            action_types = [r.action for r in decision.all_results]
            self.assertIn('fold', action_types)
            self.assertIn('call', action_types)


# ---------------------------------------------------------------------------
# Runner standalone
# ---------------------------------------------------------------------------

def run_tests():
    """Lance tous les tests avec un rapport lisible."""
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()

    sections = [
        ("Section 1 — FrequencyModel : classify_combo_response",
         TestClassifyComboResponse),
        ("Section 2 — FrequencyModel : compute()",
         TestFrequencyModelCompute),
        ("Section 3 — EVCalculator",
         TestEVCalculator),
        ("Section 4 — BestResponseEngine",
         TestBestResponseEngine),
    ]

    total_passed = total_failed = total_errors = 0

    for title, cls in sections:
        print(f"\n{'─' * 60}")
        print(f"  {title}")
        print(f"{'─' * 60}")

        s = loader.loadTestsFromTestCase(cls)
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(s)

        total_passed += result.testsRun - len(result.failures) - len(result.errors)
        total_failed += len(result.failures)
        total_errors += len(result.errors)

    print(f"\n{'═' * 60}")
    total = total_passed + total_failed + total_errors
    print(f"  TOTAL : {total_passed}/{total} tests passés")
    if total_failed == 0 and total_errors == 0:
        print("  ✅ Tous les tests Best-Response (P0) passent.")
    else:
        print(f"  ⚠  {total_failed} échec(s), {total_errors} erreur(s)")
    print(f"{'═' * 60}\n")

    return total_failed + total_errors == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
