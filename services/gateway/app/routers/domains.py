"""
Domains Router

REST API endpoints for managing domain configurations.
"""

import logging
from typing import List, Optional, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.domain_registry import get_domain_registry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/domains", tags=["domains"])


# ============================================
# Pydantic Models for API
# ============================================

class RegionInput(BaseModel):
    """Input model for region definition."""
    id: str
    name: str
    display_name: str
    color_rgb: List[int] = Field(default=[128, 128, 128], min_length=3, max_length=3)
    sam3_prompt: Optional[str] = None
    detection_heuristics: Optional[dict] = None


class ObjectInput(BaseModel):
    """Input model for object definition."""
    class_name: str
    display_name: Optional[str] = None
    real_world_size_meters: float = Field(gt=0)
    keywords: List[str] = Field(default_factory=list)
    physics_properties: dict = Field(default_factory=dict)


class EffectsInput(BaseModel):
    """Input model for effects configuration."""
    domain_specific: List[dict] = Field(default_factory=list)
    disabled: List[str] = Field(default_factory=list)
    universal_overrides: dict = Field(default_factory=dict)


class PhysicsInput(BaseModel):
    """Input model for physics configuration."""
    physics_type: str = "neutral"
    medium_density: float = 1.0
    float_threshold: Optional[float] = None
    sink_threshold: Optional[float] = None
    surface_zone: Optional[float] = None
    bottom_zone: Optional[float] = None
    gravity_direction: str = "down"


class PresetInput(BaseModel):
    """Input model for effect preset."""
    id: str
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    config: dict = Field(default_factory=dict)


class LabelingTemplateInput(BaseModel):
    """Input model for labeling template."""
    id: str
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    classes: List[str]


class DomainCreateRequest(BaseModel):
    """Request model for creating a domain."""
    domain_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str = Field(min_length=1, max_length=100)
    description: str = ""
    version: str = "1.0.0"
    icon: str = "Box"
    regions: List[RegionInput]
    objects: List[ObjectInput] = Field(default_factory=list)
    compatibility_matrix: dict = Field(default_factory=dict)
    effects: EffectsInput = Field(default_factory=EffectsInput)
    physics: PhysicsInput = Field(default_factory=PhysicsInput)
    presets: List[PresetInput] = Field(default_factory=list)
    labeling_templates: List[LabelingTemplateInput] = Field(default_factory=list)


class DomainUpdateRequest(BaseModel):
    """Request model for updating a domain."""
    name: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None
    icon: Optional[str] = None
    regions: Optional[List[RegionInput]] = None
    objects: Optional[List[ObjectInput]] = None
    compatibility_matrix: Optional[dict] = None
    effects: Optional[EffectsInput] = None
    physics: Optional[PhysicsInput] = None
    presets: Optional[List[PresetInput]] = None
    labeling_templates: Optional[List[LabelingTemplateInput]] = None


class DomainSummary(BaseModel):
    """Summary info for a domain."""
    domain_id: str
    name: str
    description: str
    icon: str
    version: str
    is_builtin: bool
    region_count: int
    object_count: int


class DomainActivateResponse(BaseModel):
    """Response after activating a domain."""
    success: bool
    active_domain_id: str
    message: str


class CompatibilityRequest(BaseModel):
    """Request for checking object-region compatibility."""
    object_class: str
    region_id: str
    domain_id: Optional[str] = None


class CompatibilityResponse(BaseModel):
    """Response with compatibility score."""
    object_class: str
    region_id: str
    domain_id: str
    score: float


# ============================================
# Endpoints
# ============================================

@router.get("", response_model=List[DomainSummary])
async def list_domains():
    """
    List all available domains.

    Returns both built-in and user-defined domains with summary information.
    """
    registry = get_domain_registry()
    domains = registry.list_domains()
    return domains


@router.get("/active")
async def get_active_domain():
    """
    Get the currently active domain.

    Returns the full domain configuration that is currently active.
    """
    registry = get_domain_registry()
    domain = registry.get_active_domain()
    if not domain:
        raise HTTPException(status_code=404, detail="No active domain set")

    return {
        "active_domain_id": registry.get_active_domain_id(),
        "domain": domain.to_dict()
    }


@router.get("/{domain_id}")
async def get_domain(domain_id: str):
    """
    Get a specific domain by ID.

    Returns the full domain configuration including regions, objects,
    compatibility matrix, effects, and physics settings.
    """
    registry = get_domain_registry()
    domain = registry.get_domain_dict(domain_id)
    if not domain:
        raise HTTPException(status_code=404, detail=f"Domain not found: {domain_id}")
    return domain


@router.post("", status_code=201)
async def create_domain(request: DomainCreateRequest):
    """
    Create a new user domain.

    Creates a new domain configuration. Built-in domain IDs cannot be used.
    """
    registry = get_domain_registry()

    # Convert to dict for storage
    domain_data = request.model_dump()

    try:
        domain = registry.create_domain(domain_data)
        if domain:
            return domain.to_dict()
        raise HTTPException(status_code=400, detail="Failed to create domain")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{domain_id}")
async def update_domain(domain_id: str, request: DomainUpdateRequest):
    """
    Update an existing user domain.

    Only non-built-in domains can be modified directly.
    For built-in domains, use the /override endpoint instead.
    """
    registry = get_domain_registry()

    # Get existing domain
    existing = registry.get_domain(domain_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Domain not found: {domain_id}")

    # If it's a built-in domain, redirect to override
    if existing.is_builtin:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot modify built-in domain directly. Use POST /{domain_id}/override instead."
        )

    # Merge updates with existing
    existing_data = existing.to_dict()
    update_data = request.model_dump(exclude_none=True)

    for key, value in update_data.items():
        existing_data[key] = value

    try:
        domain = registry.update_domain(domain_id, existing_data)
        if domain:
            return domain.to_dict()
        raise HTTPException(status_code=400, detail="Failed to update domain")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class BuiltinOverrideRequest(BaseModel):
    """Request model for creating/updating a built-in domain override."""
    regions: Optional[List[RegionInput]] = None
    objects: Optional[List[ObjectInput]] = None
    compatibility_matrix: Optional[dict] = None
    effects: Optional[EffectsInput] = None
    physics: Optional[PhysicsInput] = None
    presets: Optional[List[PresetInput]] = None
    labeling_templates: Optional[List[LabelingTemplateInput]] = None
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None


class BuiltinOverrideResponse(BaseModel):
    """Response after creating/updating a built-in override."""
    success: bool
    domain_id: str
    message: str
    has_override: bool
    domain: Optional[dict] = None


@router.post("/{domain_id}/override", response_model=BuiltinOverrideResponse)
async def create_builtin_override(domain_id: str, request: BuiltinOverrideRequest):
    """
    Create or update an override for a built-in domain.

    This endpoint allows modifying built-in domains (like 'underwater', 'aerial_birds')
    by creating a user-space copy that takes precedence over the original.

    Use this to:
    - Modify SAM3 prompts for scene regions
    - Add/update object definitions
    - Adjust compatibility scores
    - Customize effects or physics settings

    The override can be reset to restore the original built-in configuration.
    """
    registry = get_domain_registry()

    # Check domain exists
    existing = registry.get_domain(domain_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Domain not found: {domain_id}")

    # If not built-in and has override, treat as update
    if not existing.is_builtin and registry.has_override(domain_id):
        # This is already an overridden domain, update it
        pass

    # Get updates as dict
    updates = request.model_dump(exclude_none=True)

    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    try:
        domain = registry.create_builtin_override(domain_id, updates)
        if domain:
            return BuiltinOverrideResponse(
                success=True,
                domain_id=domain_id,
                message=f"Override created for domain '{domain_id}'",
                has_override=True,
                domain=domain.to_dict()
            )
        raise HTTPException(status_code=400, detail="Failed to create override")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{domain_id}/override")
async def reset_builtin_override(domain_id: str):
    """
    Reset a built-in domain override to its original configuration.

    This removes any user customizations and restores the original
    built-in domain settings.
    """
    registry = get_domain_registry()

    if not registry.has_override(domain_id):
        raise HTTPException(
            status_code=404,
            detail=f"No override exists for domain: {domain_id}"
        )

    try:
        domain = registry.reset_builtin_override(domain_id)
        if domain:
            return {
                "success": True,
                "domain_id": domain_id,
                "message": f"Override removed, domain '{domain_id}' restored to original",
                "has_override": False,
                "domain": domain.to_dict()
            }
        raise HTTPException(status_code=400, detail="Failed to reset override")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{domain_id}/override-status")
async def get_override_status(domain_id: str):
    """
    Check if a domain has a user override.

    Returns information about whether the domain has been customized
    from its original built-in configuration.
    """
    registry = get_domain_registry()

    existing = registry.get_domain(domain_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Domain not found: {domain_id}")

    has_override = registry.has_override(domain_id)

    # Get original if there's an override
    original = None
    if has_override:
        original_domain = registry.get_original_builtin(domain_id)
        if original_domain:
            original = original_domain.to_dict()

    return {
        "domain_id": domain_id,
        "has_override": has_override,
        "is_builtin": existing.is_builtin or has_override,
        "current_version": existing.version,
        "original": original
    }


@router.delete("/{domain_id}")
async def delete_domain(domain_id: str):
    """
    Delete a user domain.

    Built-in domains cannot be deleted.
    """
    registry = get_domain_registry()

    try:
        success = registry.delete_domain(domain_id)
        if success:
            return {"success": True, "message": f"Domain '{domain_id}' deleted"}
        raise HTTPException(status_code=404, detail=f"Domain not found: {domain_id}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{domain_id}/activate", response_model=DomainActivateResponse)
async def activate_domain(domain_id: str):
    """
    Activate a domain for the current session.

    The active domain determines which regions, compatibility rules,
    effects, and physics are used during generation.
    """
    registry = get_domain_registry()

    success = registry.set_active_domain(domain_id)
    if success:
        return DomainActivateResponse(
            success=True,
            active_domain_id=domain_id,
            message=f"Domain '{domain_id}' is now active"
        )
    raise HTTPException(status_code=404, detail=f"Domain not found: {domain_id}")


@router.get("/{domain_id}/export")
async def export_domain(domain_id: str):
    """
    Export a domain as JSON.

    Returns the complete domain configuration ready for import elsewhere.
    """
    registry = get_domain_registry()
    domain_data = registry.export_domain(domain_id)
    if not domain_data:
        raise HTTPException(status_code=404, detail=f"Domain not found: {domain_id}")
    return domain_data


@router.post("/import")
async def import_domain(
    domain_data: dict,
    overwrite: bool = Query(default=False, description="Overwrite existing domain if present")
):
    """
    Import a domain from JSON data.

    Imports a previously exported domain configuration.
    """
    registry = get_domain_registry()

    try:
        domain = registry.import_domain(domain_data, overwrite=overwrite)
        if domain:
            return domain.to_dict()
        raise HTTPException(status_code=400, detail="Failed to import domain")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/compatibility", response_model=CompatibilityResponse)
async def check_compatibility(request: CompatibilityRequest):
    """
    Check compatibility score for object-region pair.

    Returns a score from 0 (incompatible) to 1 (perfect match).
    """
    registry = get_domain_registry()

    domain_id = request.domain_id or registry.get_active_domain_id()
    score = registry.get_compatibility_score(
        request.object_class,
        request.region_id,
        domain_id
    )

    return CompatibilityResponse(
        object_class=request.object_class,
        region_id=request.region_id,
        domain_id=domain_id,
        score=score
    )


@router.get("/{domain_id}/regions")
async def get_domain_regions(domain_id: str):
    """
    Get all regions for a domain.

    Returns the list of scene regions with their SAM3 prompts and heuristics.
    """
    registry = get_domain_registry()
    domain = registry.get_domain(domain_id)
    if not domain:
        raise HTTPException(status_code=404, detail=f"Domain not found: {domain_id}")

    return {
        "domain_id": domain_id,
        "regions": [
            {
                "id": r.id,
                "name": r.name,
                "display_name": r.display_name,
                "color_rgb": r.color_rgb,
                "sam3_prompt": r.sam3_prompt,
                "detection_heuristics": r.detection_heuristics
            }
            for r in domain.regions
        ]
    }


@router.get("/{domain_id}/objects")
async def get_domain_objects(domain_id: str):
    """
    Get all object types for a domain.

    Returns the list of object classes with their sizes and physics properties.
    """
    registry = get_domain_registry()
    domain = registry.get_domain(domain_id)
    if not domain:
        raise HTTPException(status_code=404, detail=f"Domain not found: {domain_id}")

    return {
        "domain_id": domain_id,
        "objects": [
            {
                "class_name": o.class_name,
                "display_name": o.display_name or o.class_name,
                "real_world_size_meters": o.real_world_size_meters,
                "keywords": o.keywords,
                "physics_properties": o.physics_properties
            }
            for o in domain.objects
        ]
    }


@router.get("/{domain_id}/effects")
async def get_domain_effects(domain_id: str):
    """
    Get effects configuration for a domain.

    Returns domain-specific effects, disabled effects, and universal overrides.
    """
    registry = get_domain_registry()
    effects = registry.get_domain_effects(domain_id)
    if effects["domain_specific"] == [] and effects["disabled"] == []:
        # Check if domain exists
        if not registry.get_domain(domain_id):
            raise HTTPException(status_code=404, detail=f"Domain not found: {domain_id}")

    return {
        "domain_id": domain_id,
        **effects
    }


@router.get("/{domain_id}/sam3-prompts")
async def get_domain_sam3_prompts(domain_id: str):
    """
    Get SAM3 segmentation prompts for a domain.

    Returns the text prompts used for semantic scene analysis.
    """
    registry = get_domain_registry()
    prompts = registry.get_sam3_prompts(domain_id)

    if not prompts:
        if not registry.get_domain(domain_id):
            raise HTTPException(status_code=404, detail=f"Domain not found: {domain_id}")

    return {
        "domain_id": domain_id,
        "prompts": [
            {"text": prompt, "region_id": region_id}
            for prompt, region_id in prompts
        ]
    }


@router.get("/{domain_id}/presets")
async def get_domain_presets(domain_id: str):
    """
    Get effect presets for a domain.

    Returns pre-configured effect settings for common use cases.
    """
    registry = get_domain_registry()
    domain = registry.get_domain(domain_id)
    if not domain:
        raise HTTPException(status_code=404, detail=f"Domain not found: {domain_id}")

    return {
        "domain_id": domain_id,
        "presets": domain.presets
    }


@router.get("/{domain_id}/labeling-templates")
async def get_domain_labeling_templates(domain_id: str):
    """
    Get labeling templates for a domain.

    Returns predefined class lists for auto-labeling.
    """
    registry = get_domain_registry()
    domain = registry.get_domain(domain_id)
    if not domain:
        raise HTTPException(status_code=404, detail=f"Domain not found: {domain_id}")

    return {
        "domain_id": domain_id,
        "labeling_templates": domain.labeling_templates
    }
