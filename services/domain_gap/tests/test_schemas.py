"""
Test Pydantic schema validation.

These tests verify that:
1. Schemas accept valid data
2. Schemas have correct default values
3. Validation rules work properly
"""

import pytest
from app.models.schemas import (
    DGRTechnique,
    GapLevel,
    IssueSeverity,
    IssueCategory,
    ImpactLevel,
    ReferenceImageStats,
    ReferenceImageSet,
    MetricsRequest,
    ChannelStats,
    ColorDistribution,
    MetricsResult,
    GapIssue,
    ParameterSuggestion,
    RandomizationConfig,
    RandomizationApplyRequest,
    AnalyzeRequest,
    HealthResponse,
    InfoResponse,
)


class TestEnums:
    """Test enum values are correct."""

    def test_dgr_technique_values(self):
        assert DGRTechnique.DOMAIN_RANDOMIZATION.value == "domain_randomization"
        assert DGRTechnique.STYLE_TRANSFER.value == "neural_style_transfer"
        assert DGRTechnique.CYCLEGAN.value == "cyclegan_translation"

    def test_gap_level_values(self):
        assert GapLevel.LOW.value == "low"
        assert GapLevel.MEDIUM.value == "medium"
        assert GapLevel.HIGH.value == "high"
        assert GapLevel.CRITICAL.value == "critical"

    def test_issue_severity_values(self):
        assert IssueSeverity.LOW.value == "low"
        assert IssueSeverity.HIGH.value == "high"

    def test_issue_category_values(self):
        assert IssueCategory.COLOR.value == "color"
        assert IssueCategory.BRIGHTNESS.value == "brightness"
        assert IssueCategory.EDGES.value == "edges"
        assert IssueCategory.TEXTURE.value == "texture"
        assert IssueCategory.FREQUENCY.value == "frequency"


class TestReferenceImageStats:
    """Test ReferenceImageStats schema."""

    def test_creation(self):
        stats = ReferenceImageStats(
            channel_means_lab=[128.0, 128.0, 128.0],
            channel_stds_lab=[20.0, 10.0, 10.0],
            channel_means_rgb=[120.0, 110.0, 100.0],
            channel_stds_rgb=[30.0, 25.0, 20.0],
            avg_edge_variance=500.0,
            avg_brightness=128.0,
            image_count=10,
        )
        assert stats.image_count == 10
        assert len(stats.channel_means_lab) == 3
        assert stats.avg_brightness == 128.0


class TestMetricsRequest:
    """Test MetricsRequest schema."""

    def test_defaults(self):
        req = MetricsRequest(
            synthetic_dir="/synthetic",
            reference_set_id="abc123",
        )
        assert req.max_images == 100
        assert req.compute_fid is True
        assert req.compute_kid is True
        assert req.compute_color_distribution is True

    def test_custom_values(self):
        req = MetricsRequest(
            synthetic_dir="/syn",
            reference_set_id="ref1",
            max_images=50,
            compute_fid=False,
        )
        assert req.max_images == 50
        assert req.compute_fid is False

    def test_max_images_bounds(self):
        with pytest.raises(Exception):
            MetricsRequest(
                synthetic_dir="/syn",
                reference_set_id="ref1",
                max_images=3,  # Below minimum of 5
            )


class TestMetricsResult:
    """Test MetricsResult schema."""

    def test_creation(self):
        result = MetricsResult(
            fid_score=45.2,
            kid_score=0.035,
            kid_std=0.005,
            overall_gap_score=38.5,
            gap_level=GapLevel.MEDIUM,
            synthetic_count=50,
            real_count=30,
        )
        assert result.fid_score == 45.2
        assert result.gap_level == GapLevel.MEDIUM
        assert result.overall_gap_score == 38.5

    def test_optional_fields(self):
        result = MetricsResult(
            overall_gap_score=0.0,
            gap_level=GapLevel.LOW,
            synthetic_count=10,
            real_count=10,
        )
        assert result.fid_score is None
        assert result.kid_score is None
        assert result.color_distribution is None


class TestGapIssue:
    """Test GapIssue schema."""

    def test_creation(self):
        issue = GapIssue(
            category=IssueCategory.BRIGHTNESS,
            severity=IssueSeverity.HIGH,
            description="Synthetic images are too bright",
            metric_name="l_channel_mean_diff",
            metric_value=180.0,
            reference_value=140.0,
        )
        assert issue.category == IssueCategory.BRIGHTNESS
        assert issue.severity == IssueSeverity.HIGH
        assert issue.metric_value == 180.0


class TestParameterSuggestion:
    """Test ParameterSuggestion schema."""

    def test_creation(self):
        suggestion = ParameterSuggestion(
            parameter_path="effects.color_correction.intensity",
            current_value=0.12,
            suggested_value=0.25,
            reason="Color mismatch detected",
            expected_impact=ImpactLevel.HIGH,
        )
        assert suggestion.parameter_path == "effects.color_correction.intensity"
        assert suggestion.suggested_value == 0.25
        assert suggestion.expected_impact == ImpactLevel.HIGH

    def test_null_current_value(self):
        suggestion = ParameterSuggestion(
            parameter_path="effects.blur_matching.strength",
            suggested_value=0.5,
            reason="Edges too sharp",
            expected_impact=ImpactLevel.MEDIUM,
        )
        assert suggestion.current_value is None


class TestRandomizationConfig:
    """Test RandomizationConfig schema."""

    def test_defaults(self):
        config = RandomizationConfig()
        assert config.num_variants == 3
        assert config.intensity == 0.5
        assert config.preserve_annotations is True
        assert config.color_jitter == 0.3
        assert config.noise_intensity == 0.02
        assert config.reference_set_id is None
        assert config.histogram_match_strength == 0.5

    def test_custom_values(self):
        config = RandomizationConfig(
            num_variants=5,
            intensity=0.8,
            noise_intensity=0.05,
        )
        assert config.num_variants == 5
        assert config.intensity == 0.8

    def test_bounds(self):
        with pytest.raises(Exception):
            RandomizationConfig(num_variants=0)
        with pytest.raises(Exception):
            RandomizationConfig(intensity=1.5)


class TestAnalyzeRequest:
    """Test AnalyzeRequest schema."""

    def test_defaults(self):
        req = AnalyzeRequest(
            synthetic_dir="/syn",
            reference_set_id="ref1",
        )
        assert req.max_images == 50
        assert req.current_config is None

    def test_with_config(self):
        req = AnalyzeRequest(
            synthetic_dir="/syn",
            reference_set_id="ref1",
            current_config={"effects": {"lighting": {"intensity": 0.5}}},
        )
        assert req.current_config is not None


class TestHealthResponse:
    """Test HealthResponse schema."""

    def test_creation(self):
        health = HealthResponse(
            status="healthy",
            gpu_available=True,
            gpu_name="RTX 4090",
            engines_loaded={
                "metrics": True,
                "advisor": True,
                "randomization": True,
                "reference_manager": True,
            },
        )
        assert health.status == "healthy"
        assert health.gpu_available is True
        assert health.engines_loaded["metrics"] is True


class TestInfoResponse:
    """Test InfoResponse schema."""

    def test_defaults(self):
        info = InfoResponse()
        assert info.service == "domain_gap"
        assert info.version == "1.0.0"
        assert len(info.techniques) == 3
        assert "fid" in info.metrics_available
