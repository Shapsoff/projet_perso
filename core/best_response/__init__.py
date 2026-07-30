"""
core/best_response — Best-Response Engine
Phase 5 — Bot Poker Académique

Modules :
    frequency_model     : Calcul analytique P(fold|bet), P(call|bet), P(raise|bet)
    ev_calculator       : EV par action (fold, check, call, bet_X%, allin)
    best_response_engine: Orchestration + sélection de l'action optimale
"""

from core.best_response.frequency_model import (
    FrequencyModel,
    FoldCallRaiseResult,
    classify_combo_response,
    PlayerRange,
)
from core.best_response.ev_calculator import (
    EVCalculator,
    EVResult,
    DEFAULT_SIZINGS,
)
from core.best_response.best_response_engine import (
    BestResponseEngine,
    BestResponseConfig,
    BestResponseDecision,
)
from core.best_response.sizing_optimizer import (
    SizingOptimizer,
    OptimalSizingResult,
)
from core.best_response.bluff_layer import (
    BluffLayer,
    BluffLayerConfig,
    BluffEvalResult,
)

__all__ = [
    # frequency_model
    'FrequencyModel',
    'FoldCallRaiseResult',
    'classify_combo_response',
    'PlayerRange',
    # ev_calculator
    'EVCalculator',
    'EVResult',
    'DEFAULT_SIZINGS',
    # best_response_engine
    'BestResponseEngine',
    'BestResponseConfig',
    'BestResponseDecision',
    # sizing_optimizer
    'SizingOptimizer',
    'OptimalSizingResult',
    # bluff_layer
    'BluffLayer',
    'BluffLayerConfig',
    'BluffEvalResult',
]
