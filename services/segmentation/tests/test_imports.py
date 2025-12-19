"""
Test that all modules import correctly.

These tests verify that:
1. All dependencies are installed correctly
2. No syntax errors in application code
3. Module structure is valid
"""


def test_main_app_imports():
    """Verify main app module imports without errors."""
    from app.main import app
    assert app is not None


def test_schemas_import():
    """Verify all schemas import correctly."""
    from app.models.schemas import (
        SceneRegionType,
        AnalyzeSceneRequest,
        AnalyzeSceneResponse,
        CompatibilityCheckRequest,
        CompatibilityCheckResponse,
        SuggestPlacementRequest,
        SuggestPlacementResponse,
        SegmentTextRequest,
        SegmentTextResponse,
        HealthResponse,
        DebugAnalyzeRequest,
        DebugAnalyzeResponse,
        DebugCompatibilityRequest,
        DebugCompatibilityResponse,
        PlacementDecisionInfo,
    )
    # Verify enum has values
    assert len(SceneRegionType) > 0
