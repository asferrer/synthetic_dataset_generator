# Synthetic Dataset Generator Pro

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![Docker](https://img.shields.io/badge/Docker-Microservices-blue)
![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-green)
<a href="mailto:asanc.tech@gmail.com">
    <img alt="email" src="https://img.shields.io/badge/contact-email-yellow">
</a>

**Professional tool for photorealistic synthetic data generation with AI-powered realism effects and microservices architecture**

Generate high-quality synthetic datasets for underwater object detection with intelligent scene understanding, depth-aware composition, and physics-based lighting simulation.

---

## Table of Contents

- [Features](#-features)
- [Frontend UI](#-frontend-ui)
- [Workflow System](#-workflow-system)
- [Effects Presets](#-effects-presets)
- [Post-Processing Tools](#-post-processing-tools)
- [Architecture](#-architecture)
- [Generation Pipeline](#-generation-pipeline)
- [Scene Analysis & Placement](#-scene-analysis--placement-decisions)
- [Object Composition Pipeline](#-object-composition-pipeline)
- [Debug Output](#-debug-output-explainability)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)

---

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **AI-Powered Depth Estimation** | Depth Anything V3 for accurate monocular depth maps |
| **Semantic Scene Analysis** | SAM3 text-prompted segmentation with heuristic fallback |
| **Intelligent Object Placement** | Physics-aware positioning based on scene understanding |
| **Multi-Light Source Detection** | Advanced lighting estimation with shadow generation |
| **Quality Validation** | LPIPS perceptual metrics and physics consistency checks |
| **Microservices Architecture** | Scalable, GPU-optimized Docker services |

### AI Models Integrated

| Model | Service | Purpose |
|-------|---------|---------|
| **Depth Anything V3** | Depth Service | High-quality monocular depth estimation |
| **SAM3** | Segmentation Service | Text-prompted scene segmentation |
| **LPIPS** | Augmentor Service | Perceptual quality validation |

### Realism Effects Pipeline

1. **Depth-Aware Composition** - Objects scaled and blurred based on depth
2. **Multi-Light Shadow Generation** - Realistic shadows from detected light sources
3. **Adaptive Color Matching** - Histogram matching and tone adaptation
4. **Underwater Caustics** - Pre-cached Perlin noise patterns (500-1000x faster)
5. **Motion Blur** - Vectorized kernel generation
6. **Poisson Blending** - Seamless boundary integration
7. **Edge Smoothing** - Distance transform-based feathering

### Image Composition Quality

| Feature | Implementation | Purpose |
|---------|---------------|---------|
| **Edge Feathering** | Distance transform + gradient alpha | Seamless object-background transitions |
| **Anti-aliasing** | LANCZOS4/AREA interpolation | No jagged edges on scaled objects |
| **Small Object Handling** | Pre-blur + minimum size filter | Prevent low-res artifacts |
| **Large Upscaling** | Post-smoothing for >2x scale | Blend interpolation artifacts |

### Docker Build Testing

All services include automated unit tests that run during Docker build:

```bash
# Build with tests (default) - fails if tests fail
docker-compose -f docker-compose.microservices.yml build

# Skip tests for emergency builds
docker build --target production services/augmentor/
```

---

## Frontend UI

### Professional Interface

The Streamlit-based frontend provides a modern, professional interface for synthetic data generation:

| Feature | Description |
|---------|-------------|
| **6-Step Workflow Wizard** | Guided process from analysis to final splits |
| **Visual Progress Stepper** | Track progress through the generation pipeline |
| **Light/Dark Themes** | Native Streamlit theme support |
| **Responsive Design** | Optimized for various screen sizes |
| **Real-time Monitoring** | Live job progress with ETA estimates |

### Navigation Structure

```
┌─────────────────────────┐
│       🔬 SDG            │
│   Synthetic Data Gen    │
├─────────────────────────┤
│  🏠 Home                │  ← Quick start + status
├─────────────────────────┤
│  WORKFLOW PRINCIPAL     │
│  ─────────────────────  │
│  ① Análisis            │  ← Upload and analyze JSON
│  ② Configuración       │  ← Configure generation
│  ③ Generación          │  ← Batch + monitor
│  ④ Exportar            │  ← Export formats
│  ⑤ Combinar            │  ← Merge datasets
│  ⑥ Splits              │  ← Train/Val/Test
├─────────────────────────┤
│  HERRAMIENTAS           │
│  ─────────────────────  │
│  🏷️ Etiquetas          │  ← Label management
│  📊 Monitor            │  ← Job monitoring
│  🔧 Servicios          │  ← Microservices status
│  📚 Docs               │  ← Documentation
├─────────────────────────┤
│  [Estado: 5/5 ✓]       │  ← Mini service status
└─────────────────────────┘
```

---

## Workflow System

### 6-Step Generation Workflow

The application follows a structured workflow for synthetic data generation:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SYNTHETIC GENERATION FLOW                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [1. ANÁLISIS]  →  [2. CONFIGURAR]  →  [3. GENERAR]  →  [4. EXPORTAR]  │
│                                                                          │
│  Upload COCO       Configure          Launch batch      Export to       │
│  JSON and          effects and        and monitor       YOLO/VOC/etc    │
│  analyze           balancing          progress                           │
│                                                    ↓                     │
│                                            [5. COMBINAR]                 │
│                                                                          │
│                                            Merge with original           │
│                                            or other datasets             │
│                                                    ↓                     │
│                                            [6. SPLITS]                   │
│                                                                          │
│                                            Train/Val/Test                │
│                                            or K-Fold                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Step Details

| Step | Page | Description |
|------|------|-------------|
| **1. Análisis** | Upload & analyze | Load COCO JSON, view class distribution, configure balancing strategy |
| **2. Configurar** | Configure | Set effects, directories, intensity sliders, advanced options |
| **3. Generar** | Generate | Launch batch job, monitor progress, view previews |
| **4. Exportar** | Export | Export to COCO, YOLO, Pascal VOC, CreateML formats |
| **5. Combinar** | Combine | Merge synthetic dataset with original or other datasets |
| **6. Splits** | Splits | Create train/val/test splits or K-Fold cross-validation |

### Balancing Strategies

| Strategy | Description |
|----------|-------------|
| **Complete** | Balance all classes to match the maximum count |
| **Partial** | Balance to 75% of the maximum count |
| **Minority** | Only augment underrepresented classes (below median) |
| **Custom** | Manually specify samples per class |

---

## Effects Presets

### Reusable Configuration Presets

Save and load effects configurations as reusable presets. Ideal for replicating successful settings across different datasets.

**Key Features:**
- Save only effects and advanced options (no dataset-specific info)
- Auto-save preset when generation completes
- Backward compatibility with legacy config format
- Import/export via JSON files

### Preset Structure

```json
{
  "preset_version": "1.0",
  "preset_type": "effects_config",
  "created_at": "2024-12-20T01:08:00",
  "description": "Preset de efectos y opciones avanzadas reutilizable",

  "effects": {
    "enabled": ["color_correction", "blur_matching", "shadows", "caustics", "underwater", "edge_smoothing"],
    "intensities": {
      "color_intensity": 0.7,
      "blur_strength": 1.0,
      "shadow_opacity": 0.4,
      "underwater_intensity": 0.25,
      "caustics_intensity": 0.15,
      "edge_feather": 5,
      "lighting_intensity": 0.5,
      "motion_blur_probability": 0.2
    },
    "parameters": {
      "lighting_type": "ambient",
      "water_color": [20, 80, 120],
      "water_clarity": "clear"
    }
  },

  "advanced_options": {
    "max_objects_per_image": 5,
    "overlap_threshold": 0.1,
    "depth_aware": true
  },

  "validation": {
    "validate_quality": false,
    "validate_physics": false
  },

  "debug": {
    "save_pipeline_debug": false
  }
}
```

### Using Presets

1. **Export Current Settings**: Click "Descargar Preset" in the Configure page
2. **Import Preset**: Upload a JSON preset file and click "Aplicar Preset"
3. **Auto-saved**: Presets are automatically saved to `effects_preset.json` in the job output folder

---

## Post-Processing Tools

### Independent Tools

The application includes standalone post-processing tools that work with any COCO dataset:

| Tool | Description |
|------|-------------|
| **🏷️ Labels** | Rename, merge, delete, and reorder categories |
| **📤 Export** | Convert to YOLO, Pascal VOC, CreateML formats |
| **✂️ Splits** | Create train/val/test or K-Fold splits |
| **⚖️ Balancing** | Oversample/undersample classes |
| **📊 Monitor** | View all generation jobs and their status |

### Export Formats

| Format | Output | Use Case |
|--------|--------|----------|
| **COCO JSON** | `annotations.json` | Default format, native to the tool |
| **YOLO** | `.txt` + `data.yaml` | Ultralytics YOLO training |
| **Pascal VOC** | `.xml` per image | Classic object detection format |
| **CreateML** | `annotations.json` | Apple Core ML training |

### Dataset Splitting

**Train/Val/Test Split:**
- Configurable ratios (default: 70/20/10)
- Stratified splitting to maintain class distribution
- Random seed for reproducibility

**K-Fold Cross-Validation:**
- 2-10 folds supported
- Stratified K-Fold option
- Generates separate JSON files per fold

### Dataset Combination

Merge multiple COCO datasets with intelligent ID management:

| Strategy | Description |
|----------|-------------|
| **ID Offset** | Add offset to avoid ID collisions |
| **ID Reassign** | Reassign all IDs starting from 1 |
| **Category Unify** | Merge categories with same name |
| **Category Separate** | Keep categories separate with suffix |

---

## Architecture

### Microservices Overview

```
                                    +------------------+
                                    |    Frontend      |
                                    |   (Streamlit)    |
                                    |   Port: 8501     |
                                    +--------+---------+
                                             |
                                             v
                                    +------------------+
                                    |    Gateway       |
                                    |   (FastAPI)      |
                                    |   Port: 8000     |
                                    +--------+---------+
                                             |
                    +------------------------+------------------------+
                    |                        |                        |
                    v                        v                        v
          +------------------+    +------------------+    +------------------+
          |  Depth Service   |    |  Segmentation    |    |  Effects Service |
          | Depth Anything V3|    |  Service (SAM3)  |    |  (Realism FX)    |
          |   Port: 8001     |    |   Port: 8002     |    |   Port: 8003     |
          +------------------+    +------------------+    +------------------+
                    |                        |                        |
                    +------------------------+------------------------+
                                             |
                                             v
                                    +------------------+
                                    |   Augmentor      |
                                    |   (Composer)     |
                                    |   Port: 8004     |
                                    +------------------+
```

### Service Descriptions

| Service | Port | GPU | Description |
|---------|------|-----|-------------|
| **Gateway** | 8000 | No | API orchestration and routing |
| **Depth** | 8001 | Yes | Depth Anything V3 inference |
| **Segmentation** | 8002 | Yes | SAM3 + Scene analysis |
| **Effects** | 8003 | No | Lighting, caustics, blur effects |
| **Augmentor** | 8004 | Yes | Image composition and validation |
| **Frontend** | 8501 | No | Streamlit web interface |

### Data Flow

```
[Background Image] + [Object Images]
            |
            v
    +---------------+
    | Depth Service |-----> Depth Map (normalized 0-1)
    +---------------+
            |
            v
    +-------------------+
    | Segmentation Svc  |-----> Scene Analysis:
    +-------------------+       - Region Map (pixel-wise)
            |                   - Region Scores (%)
            |                   - Dominant Region
            |                   - Water Clarity
            v                   - Color Temperature
    +-----------------+
    | Effects Service |-----> Lighting Info:
    +-----------------+       - Light Positions
            |                 - Light Intensities
            |                 - Shadow Parameters
            v
    +-----------------+
    | Augmentor       |-----> Composed Image + Annotations
    +-----------------+       - COCO Format JSON
                              - Bounding Boxes
                              - Segmentation Masks
```

---

## Generation Pipeline

### Complete Flow Diagram

```
START
  |
  v
[1] LOAD INPUTS
  |-- Background image (JPG/PNG)
  |-- Object images with masks (PNG with alpha)
  |-- Configuration parameters
  |
  v
[2] DEPTH ESTIMATION (Depth Service)
  |-- Input: Background image
  |-- Model: Depth Anything V3 (Base/Large)
  |-- Output: Depth map (H x W, float32, 0-1)
  |-- Decision: Objects in foreground get larger scale
  |
  v
[3] SCENE ANALYSIS (Segmentation Service)
  |-- Input: Background image
  |-- Method: SAM3 (if available) or Heuristics
  |-- Output:
  |     |-- Region Map: pixel-wise classification
  |     |-- Region Scores: {open_water: 0.45, seafloor: 0.35, ...}
  |     |-- Dominant Region: "open_water"
  |     |-- Scene Properties: brightness, clarity, temperature
  |-- Decision: Determines valid placement zones per object type
  |
  v
[4] LIGHTING ANALYSIS (Effects Service)
  |-- Input: Background image
  |-- Analysis:
  |     |-- Sobel gradients for light direction
  |     |-- Multi-light source detection
  |     |-- Color temperature estimation
  |-- Output:
  |     |-- Light positions [(x, y, intensity), ...]
  |     |-- Shadow parameters (angle, distance, blur)
  |-- Decision: Shadow direction and intensity per object
  |
  v
[5] FOR EACH OBJECT:
  |
  |  [5.1] COMPATIBILITY CHECK
  |    |-- Input: Object class, proposed position
  |    |-- Lookup: SCENE_COMPATIBILITY[class][region]
  |    |-- Score: 0.0 (incompatible) to 1.0 (perfect)
  |    |-- Decision:
  |          score >= 0.8  -> ACCEPTED (green)
  |          score >= 0.4  -> MARGINAL (proceed with warning)
  |          score <  0.4  -> REJECTED or RELOCATED (orange/red)
  |
  |  [5.2] POSITION SELECTION
  |    |-- If ACCEPTED: Use proposed position
  |    |-- If RELOCATED: Find best alternative in compatible region
  |    |-- Constraints:
  |          - Minimum distance from other objects
  |          - Within image bounds (with margin)
  |          - Depth-appropriate zone
  |
  |  [5.3] SIZE CALCULATION
  |    |-- Base: Original object size
  |    |-- Depth factor: depth_map[y, x] -> scale multiplier
  |    |-- Constraints:
  |          - min_area_ratio: 0.5% of image
  |          - max_area_ratio: 40% of image
  |          - max_upscale_ratio: 2.0x
  |    |-- Output: Final (width, height)
  |
  |  [5.4] APPLY TRANSFORMATIONS (Anti-aliasing)
  |    |-- Pre-processing:
  |          - Small objects (<100px): Pre-blur to reduce jaggies
  |    |-- Rotation: INTER_CUBIC interpolation
  |          - Random -45 to +45 degrees
  |          - High-quality anti-aliased rotation
  |    |-- Scaling:
  |          - Upscaling: INTER_LANCZOS4 (best quality)
  |          - Downscaling: INTER_AREA (prevents moire)
  |          - Large upscale (>2x): Post-smoothing applied
  |    |-- Quality gate: Reject objects < 30px
  |
  |  [5.5] APPLY EFFECTS
  |    |-- Color Matching: Histogram adaptation to background
  |    |-- Blur Consistency: Match background blur level (conservative)
  |    |-- Shadow Generation: Based on lighting analysis
  |    |-- Depth Blur: Gaussian blur based on depth difference
  |    |-- Edge Smoothing: Distance transform-based feathering
  |
  |  [5.6] COMPOSITE (Edge Blending)
  |    |-- Mask refinement:
  |          - Distance transform for edge detection
  |          - Gradual alpha falloff at boundaries
  |    |-- Alpha blending with feathered edges (4px default)
  |    |-- Color harmonization at transition zone
  |    |-- Z-ordering: Back-to-front rendering
  |    |-- Update placement mask for overlap detection
  |
  v
[6] POST-PROCESSING
  |-- Underwater effects (if enabled):
  |     |-- Color tint extraction
  |     |-- Caustics overlay
  |-- Global adjustments:
  |     |-- Brightness normalization
  |     |-- Contrast enhancement
  |
  v
[7] QUALITY VALIDATION (Optional)
  |-- LPIPS score: Perceptual similarity check
  |-- Physics check: Object-scene consistency
  |-- Decision:
  |     score > threshold -> ACCEPT
  |     score <= threshold -> REJECT (regenerate)
  |
  v
[8] ANNOTATION GENERATION
  |-- Bounding boxes: [x, y, width, height]
  |-- Segmentation masks: RLE or polygon format
  |-- COCO JSON structure
  |
  v
[9] OUTPUT
  |-- composed_image.jpg
  |-- annotations.json (COCO format)
  |-- debug/ (if enabled):
        |-- 01_original_background.jpg
        |-- 02_lighting_analysis.jpg
        |-- 02b_scene_analysis.jpg
        |-- 03_depth_map.jpg
        |-- 04-07_object_placement_*.jpg
        |-- 08_final_composite.jpg
        |-- 08b_placement_decisions.jpg
  |
  v
END
```

---

## Scene Analysis & Placement Decisions

### Region Types

The scene analyzer classifies underwater images into these regions:

| Region | Value | Description | Typical Location |
|--------|-------|-------------|------------------|
| `open_water` | 1 | Blue/cyan water column | Upper-middle area |
| `seafloor` | 2 | Bottom substrate | Lower 40% |
| `surface` | 3 | Water surface/sky interface | Top 20% |
| `vegetation` | 4 | Seaweed, kelp, plants | Variable |
| `rocky` | 5 | Rocky substrate | Lower area |
| `sandy` | 6 | Sandy bottom | Lower area |
| `murky` | 7 | Low visibility water | Any |

### Detection Methods

#### Method 1: SAM3 Text-Prompted Segmentation

When SAM3 is available, the system uses text prompts:

```python
region_prompts = [
    ("water", OPEN_WATER),
    ("seafloor", SEAFLOOR),
    ("water surface", SURFACE),
    ("seaweed", VEGETATION),
    ("rock", ROCKY),
    ("sand", SANDY),
]
```

Each prompt generates a confidence mask, and pixels are assigned to the highest-confidence region.

#### Method 2: Heuristic Analysis (Fallback)

When SAM3 is unavailable, color and texture heuristics are used:

| Region | Detection Criteria |
|--------|-------------------|
| **Surface** | `brightness > 0.75` AND `y_position < 30%` |
| **Seafloor** | `y_position > 55%` AND (`R > B*0.9` OR `saturation < 100`) |
| **Vegetation** | `30 < hue < 90` AND `saturation > 30` AND `G > B` |
| **Rocky** | `saturation < 60` AND high Laplacian texture AND `y > 40%` |
| **Murky** | Global contrast `std < 35` |
| **Open Water** | `B > R*1.1` OR cyan hue `80 < hue < 135` |

### Object-Scene Compatibility Rules

Compatibility scores (0.0 - 1.0) determine if an object fits in a region:

```python
SCENE_COMPATIBILITY = {
    'fish': {
        OPEN_WATER: 1.0,    # Perfect - fish swim in water
        SURFACE: 0.8,       # Good - near surface feeding
        VEGETATION: 0.7,    # Acceptable - hiding in plants
        SEAFLOOR: 0.5,      # Marginal - bottom feeders only
    },
    'can': {
        SEAFLOOR: 1.0,      # Perfect - sinks to bottom
        SANDY: 1.0,         # Perfect - rests on sand
        ROCKY: 0.8,         # Good - lodged in rocks
        OPEN_WATER: 0.3,    # Bad - cans don't float
        SURFACE: 0.1,       # Very bad - unrealistic
    },
    'plastic_bag': {
        OPEN_WATER: 1.0,    # Perfect - floats/drifts
        SURFACE: 1.0,       # Perfect - floating debris
        SEAFLOOR: 0.5,      # Marginal - waterlogged bags sink
    },
    'jellyfish': {
        OPEN_WATER: 1.0,    # Perfect - drifting
        SURFACE: 0.9,       # Good - near surface
        SEAFLOOR: 0.2,      # Bad - jellyfish don't rest on bottom
    },
    # ... more objects
}
```

### Placement Decision Flow

```
Input: object_class="can", position=(500, 200)
                |
                v
        Query region_map[200, 500]
                |
                v
        region_value = 1 (OPEN_WATER)
                |
                v
        SCENE_COMPATIBILITY["can"][OPEN_WATER] = 0.3
                |
                v
        0.3 < 0.4 (min_threshold)
                |
                v
        Decision: REJECTED or RELOCATED
                |
                v
        Find alternatives in SEAFLOOR region
                |
                v
        Best alternative: (500, 650) with score 1.0
                |
                v
        Final Decision: RELOCATED to (500, 650)
```

---

## Object Composition Pipeline

### Mask Refinement Process

The mask refinement creates smooth transitions between objects and backgrounds:

```
Input: Binary mask (0 or 255 values)
         |
         v
+---------------------------+
| Distance Transform        |
| - Calculate distance from |
|   each pixel to edge      |
+---------------------------+
         |
         v
+---------------------------+
| Alpha Gradient Creation   |
| - Core (far from edge):   |
|   alpha = 1.0 (opaque)    |
| - Edge zone (< feather):  |
|   alpha = distance/feather|
| - Outside: alpha = 0.0    |
+---------------------------+
         |
         v
+---------------------------+
| Gaussian Smoothing        |
| - Blur alpha channel      |
| - Creates natural falloff |
+---------------------------+
         |
         v
Output: Smooth alpha mask (0.0-1.0 float)
```

### Interpolation Methods

| Scenario | Method | Why |
|----------|--------|-----|
| **Upscaling** | INTER_LANCZOS4 | Best quality, 8x8 kernel, minimal artifacts |
| **Downscaling** | INTER_AREA | Pixel area resampling, prevents moire patterns |
| **Rotation** | INTER_CUBIC | Good balance of quality and performance |
| **Small objects** | Pre-blur + above | Reduces input jaggies before transformation |

### Edge Blending Algorithm

```python
# Simplified edge blending logic
dist_inside = distance_transform(mask)     # Distance from edge (inward)
feather_px = 4                              # Default feather radius

# Create alpha gradient
alpha = clip(dist_inside / feather_px, 0, 1)
alpha = gaussian_blur(alpha, sigma=1.5)

# Ensure core stays opaque
core_mask = dist_inside > feather_px * 1.5
alpha[core_mask] = 1.0

# Final blend
result = object * alpha + background * (1 - alpha)
```

### Quality Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `edge_feather` | 4 | 1-15 | Pixels of edge transition |
| `blur_strength` | 0.5 | 0-2 | Background blur matching |
| `shadow_opacity` | 0.12 | 0-0.5 | Shadow darkness |
| `shadow_blur` | 31 | 3-61 | Shadow softness |
| `min_object_size` | 30px | - | Reject smaller objects |

---

## Debug Output (Explainability)

When debug mode is enabled, the pipeline generates visualization images:

### Debug Images

| File | Description | Contents |
|------|-------------|----------|
| `01_original_background.jpg` | Input background | Unmodified background image |
| `02_lighting_analysis.jpg` | Lighting estimation | Light sources, directions, intensity map |
| `02b_scene_analysis.jpg` | Scene segmentation | 3-panel: Original \| Region Overlay \| Info |
| `03_depth_map.jpg` | Depth estimation | Colorized depth map (near=warm, far=cool) |
| `04-07_object_*.jpg` | Per-object placement | Transform stages for first object |
| `08_final_composite.jpg` | Final result | Complete composed image |
| `08b_placement_decisions.jpg` | Decision visualization | Bounding boxes with decision colors |

### 02b_scene_analysis.jpg Format

```
+------------------+------------------+------------------+
|                  |                  |                  |
|    Original      |  Region Overlay  |   Analysis Info  |
|    Image         |  (40% opacity)   |                  |
|                  |                  |  Dominant: seafloor
|                  |  [colored by     |  open_water: 45%
|                  |   region type]   |  seafloor: 35%
|                  |                  |  vegetation: 12%
|                  |                  |  Method: heuristic
|                  |                  |  Brightness: 0.52
|                  |                  |  Clarity: moderate
|                  |                  |  Time: 45.2ms
+------------------+------------------+------------------+
```

**Region Colors (BGR):**
- `open_water`: Blue (255, 150, 50)
- `seafloor`: Sandy brown (50, 150, 200)
- `surface`: Light cyan (255, 255, 200)
- `vegetation`: Green (50, 200, 50)
- `rocky`: Gray (100, 100, 100)
- `sandy`: Sand (100, 180, 220)
- `murky`: Dark (80, 80, 60)

### 08b_placement_decisions.jpg Format

```
+-----------------------------------------------+
|                                               |
|    [Object 1]          [Object 2]             |
|    +----------+        +----------+           |
|    | can      |        | fish     |           |
|    | [accepted]|       | [relocated]          |
|    | 95%      |        | 72%      |           |
|    | @seafloor|        | @open_water          |
|    +----------+        +----------+           |
|                                               |
|  +------------------------------------------+ |
|  | PLACEMENT DECISIONS                      | |
|  | O Accepted   O Relocated   O Rejected    | |
|  | Total: 5 | OK: 3 | Moved: 2              | |
|  +------------------------------------------+ |
+-----------------------------------------------+
```

**Decision Colors:**
- Green: `accepted` - Good placement for region
- Orange: `relocated` - Moved to better position
- Red: `rejected` - Incompatible, not placed

---

## Prerequisites

Before running the system, ensure you have completed the following setup steps:

### 1. System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA GTX 1080 (8GB VRAM) | NVIDIA RTX 3090+ (24GB VRAM) |
| **CUDA** | 12.0+ | 12.4+ |
| **Docker** | 20.10+ | 24.0+ |
| **Docker Compose** | 2.0+ | 2.24+ |
| **Disk Space** | 30GB | 50GB+ (for model checkpoints) |

### 2. HuggingFace Account & Token

The system uses AI models from HuggingFace that require authentication:

1. **Create a HuggingFace account** at [huggingface.co](https://huggingface.co/join)
2. **Generate an access token**:
   - Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
   - Create a new token with **read** permissions
   - Copy the token (starts with `hf_`)

3. **Set the token** in your environment:
   ```bash
   # Option A: Environment variable
   export HF_TOKEN=hf_your_token_here

   # Option B: Create .env file
   echo "HF_TOKEN=hf_your_token_here" > .env.microservices
   ```

### 3. SAM3 Model Access (Required for Text-Prompted Segmentation)

The SAM3 model (`facebook/sam3`) is a **gated model** that requires explicit access approval:

1. **Visit the model page**: [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
2. **Click "Request access"** and fill out the form
3. **Wait for approval** (usually within 24-48 hours)
4. Once approved, the model will load automatically when the service starts

> **Note**: While waiting for SAM3 access, the system will fall back to **heuristic-based scene analysis**, which still provides good results for most use cases.

**How SAM3 Authentication Works**:
- The segmentation container automatically logs in to HuggingFace using your `HF_TOKEN`
- This login happens at container startup, before loading any models
- The startup logs will show: `"HuggingFace login successful"` when authentication works
- If you see `"SAM3 failed to load"`, verify your token has access to the SAM3 model

### 4. Verify GPU Access for Docker

Ensure Docker can access your NVIDIA GPU:

```bash
# Check NVIDIA driver
nvidia-smi

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If the above fails, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### 5. Create Required Directories

```bash
mkdir -p volumes/shared volumes/checkpoints volumes/cache output
```

---

## Quick Start

### Option 1: Docker Microservices (Recommended)

```bash
# Clone repository
git clone https://github.com/asferrer/synthetic_dataset_generator.git
cd synthetic_dataset_generator

# Create required directories
mkdir -p volumes/shared volumes/checkpoints volumes/cache

# Start all services
docker-compose -f docker-compose.microservices.yml up -d

# Check service health
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Access frontend
# Open http://localhost:8501 in browser
```

### Option 2: Single Container

```bash
docker-compose up --build
# Access at http://localhost:8501
```

### Option 3: Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app/main.py
```

---

## Configuration

### docker-compose.microservices.yml

Key environment variables:

```yaml
services:
  depth:
    environment:
      - DEPTH_MODEL=DA3-BASE      # DA3-BASE, DA3-LARGE
      - CUDA_VISIBLE_DEVICES=0

  segmentation:
    environment:
      - HF_TOKEN=${HF_TOKEN}      # For SAM3 model download
      - CUDA_VISIBLE_DEVICES=0

  augmentor:
    environment:
      - DEPTH_SERVICE_URL=http://depth:8001
      - SEGMENTATION_SERVICE_URL=http://segmentation:8002
      - EFFECTS_SERVICE_URL=http://effects:8003
```

### Scene Compatibility Configuration

Edit `services/segmentation/app/scene_analyzer.py`:

```python
SCENE_COMPATIBILITY = {
    'your_new_object': {
        SceneRegion.SEAFLOOR: 1.0,      # Best region
        SceneRegion.OPEN_WATER: 0.5,    # Acceptable
        SceneRegion.SURFACE: 0.2,       # Poor fit
    },
}
```

### Heuristic Thresholds

Adjust detection sensitivity in `_analyze_with_heuristics()`:

```python
# Surface detection
surface_brightness_threshold = 0.75  # Increase for stricter detection
surface_y_limit = 0.3               # Top 30% of image

# Seafloor detection
seafloor_y_start = 0.55             # Bottom 45% of image
warm_color_ratio = 0.9              # R > B * 0.9
```

---

## API Reference

### Gateway API (Port 8000)

```bash
# Health check
GET /health

# Generate synthetic image
POST /generate
{
  "background_path": "/app/datasets/Backgrounds_filtered/bg001.jpg",
  "objects": [
    {"class_name": "can", "image_path": "/app/datasets/Objects/can/can001.png"},
    {"class_name": "fish", "image_path": "/app/datasets/Objects/fish/fish001.png"}
  ],
  "config": {
    "max_objects": 5,
    "debug_output": true
  }
}
```

### Segmentation API (Port 8002)

```bash
# Scene analysis
POST /analyze
{
  "image_path": "/shared/temp/background.jpg"
}

# Response
{
  "success": true,
  "dominant_region": "seafloor",
  "region_scores": {
    "open_water": 0.35,
    "seafloor": 0.45,
    "vegetation": 0.12,
    "surface": 0.08
  },
  "scene_brightness": 0.52,
  "water_clarity": "moderate",
  "color_temperature": "cool"
}

# Debug analysis with visualization
POST /debug/analyze
{
  "image_path": "/shared/temp/background.jpg",
  "save_visualization": true,
  "image_id": "test_001"
}

# Compatibility check
POST /check-compatibility
{
  "image_path": "/shared/temp/background.jpg",
  "object_class": "can",
  "position_x": 500,
  "position_y": 200
}

# Response
{
  "success": true,
  "score": 0.3,
  "reason": "can is incompatible with open_water",
  "is_compatible": false
}
```

### Depth API (Port 8001)

```bash
# Estimate depth
POST /estimate
{
  "image_path": "/shared/temp/background.jpg"
}

# Response
{
  "success": true,
  "depth_path": "/shared/depth/background_depth.npy",
  "min_depth": 0.0,
  "max_depth": 1.0,
  "processing_time_ms": 125.4
}
```

---

## Project Structure

```
synthetic_dataset_generator/
|
+-- services/                      # Microservices
|   +-- gateway/                   # API Gateway
|   |   +-- app/main.py
|   |   +-- Dockerfile
|   |
|   +-- depth/                     # Depth Anything V3
|   |   +-- app/main.py
|   |   +-- app/depth_estimator.py
|   |   +-- Dockerfile
|   |
|   +-- segmentation/              # SAM3 + Scene Analysis
|   |   +-- app/main.py
|   |   +-- app/scene_analyzer.py  # <- Compatibility rules here
|   |   +-- app/models.py
|   |   +-- Dockerfile
|   |
|   +-- effects/                   # Realism Effects
|   |   +-- app/main.py
|   |   +-- app/lighting.py
|   |   +-- app/caustics.py
|   |   +-- Dockerfile
|   |
|   +-- augmentor/                 # Image Composition
|       +-- app/main.py
|       +-- app/composer.py        # <- Main composition logic
|       +-- app/segmentation_client.py
|       +-- Dockerfile
|
+-- frontend/                      # Streamlit UI
|   +-- app/
|   |   +-- main.py               # Application entry point
|   |   +-- config/
|   |   |   +-- theme.py          # Light/Dark theme system
|   |   |   +-- styles.py         # Custom CSS styles
|   |   +-- components/
|   |   |   +-- ui.py             # Reusable UI components
|   |   |   +-- api_client.py     # Gateway API client
|   |   +-- pages/
|   |   |   +-- analysis.py       # Step 1: Dataset analysis
|   |   |   +-- configure.py      # Step 2: Generation config
|   |   |   +-- generation.py     # Step 3: Batch generation
|   |   |   +-- export.py         # Step 4: Export formats
|   |   |   +-- combine.py        # Step 5: Merge datasets
|   |   |   +-- splits.py         # Step 6: Train/Val/Test splits
|   |   |   +-- post_processing.py # Standalone tools
|   |   +-- utils/
|   |       +-- merger.py         # Dataset merger utility
|   |       +-- label_manager.py  # Label management
|   |       +-- export_manager.py # Export utilities
|   |       +-- dataset_splitter.py # Split utilities
|   +-- Dockerfile
|
+-- src/                          # Legacy monolithic code
|   +-- augmentation/
|   +-- analysis/
|   +-- data/
|
+-- volumes/                      # Docker volumes
|   +-- shared/                   # Inter-service file sharing
|   +-- checkpoints/              # Model weights
|   +-- cache/                    # Caustics cache
|
+-- output/                       # Generated images
|   +-- synthetic/
|       +-- job_<id>/
|           +-- images/           # Generated images
|           +-- synthetic_dataset.json
|           +-- effects_preset.json  # Auto-saved preset
|           +-- pipeline_debug/   # Debug images (if enabled)
|
+-- docker-compose.microservices.yml
+-- docker-compose.yml            # Single container mode
+-- README.md
```

---

## Performance

### Benchmarks (RTX 3090, 24GB VRAM)

| Operation | Time | Notes |
|-----------|------|-------|
| Depth estimation | 80-150ms | Depth Anything V3 Base |
| Scene analysis (SAM3) | 200-400ms | Per prompt |
| Scene analysis (heuristic) | 15-30ms | Fallback mode |
| Effects pipeline | 50-100ms | All effects enabled |
| Full composition | 500-800ms | 5 objects, all features |

### Quality Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| LPIPS threshold | 0.3 | Perceptual quality gate |
| FID Score | 18.5 | Distribution similarity |
| Physics consistency | 92% | Object-scene match rate |

---

## Contributing

### Adding New Object Types

1. Add compatibility rules in `scene_analyzer.py`:

```python
SCENE_COMPATIBILITY['new_object'] = {
    SceneRegion.SEAFLOOR: 0.9,
    SceneRegion.OPEN_WATER: 0.4,
    # ...
}
```

2. Add keyword mapping for class normalization:

```python
keyword_map = {
    'new_object': ['new', 'object', 'synonyms'],
}
```

### Adding New Region Types

1. Add to `SceneRegion` enum
2. Add detection logic in `_analyze_with_heuristics()`
3. Add SAM3 prompt in `region_prompts`
4. Update compatibility rules for all objects

---

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

---

## Contact

- **Email**: asanc.tech@gmail.com
- **Issues**: [GitHub Issues](https://github.com/asferrer/synthetic_dataset_generator/issues)

---

<div align="center">

**Made with care for the computer vision community**

</div>
