"""
Test that all modules import correctly.

These tests verify that:
1. All dependencies are installed correctly
2. No syntax errors in application code
3. Module structure is valid
"""


def test_schemas_import():
    """Verify all schemas import correctly."""
    from app.models.schemas import (
        DGRTechnique,
        GapLevel,
        IssueSeverity,
        IssueCategory,
        ImpactLevel,
        ReferenceImageStats,
        ReferenceImageSet,
        ReferenceUploadResponse,
        ReferenceListResponse,
        MetricsRequest,
        ChannelStats,
        ColorDistribution,
        MetricsResult,
        MetricsCompareRequest,
        MetricsCompareResponse,
        AnalyzeRequest,
        GapAnalysis,
        GapIssue,
        ParameterSuggestion,
        RandomizationConfig,
        RandomizationApplyRequest,
        RandomizationBatchRequest,
        RandomizationResponse,
        StyleTransferConfig,
        CycleGANTrainConfig,
        HealthResponse,
        InfoResponse,
        JobStatusResponse,
    )
    assert len(DGRTechnique) > 0
    assert len(GapLevel) > 0
    assert len(IssueSeverity) > 0
    assert len(IssueCategory) > 0
    assert len(ImpactLevel) > 0


def test_engines_import():
    """Verify engine modules import correctly."""
    from app.engines.metrics_engine import MetricsEngine
    from app.engines.advisor_engine import AdvisorEngine
    from app.engines.randomization_engine import RandomizationEngine
    assert MetricsEngine is not None
    assert AdvisorEngine is not None
    assert RandomizationEngine is not None


def test_reference_manager_import():
    """Verify reference manager imports correctly."""
    from app.reference_manager import ReferenceManager
    assert ReferenceManager is not None


def test_main_app_imports():
    """Verify main app module imports without errors."""
    from app.main import app
    assert app is not None
