# Synthetic Dataset Generator

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![Docker](https://img.shields.io/badge/Docker-Microservices-blue)
![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-green)
<a href="mailto:asanc.tech@gmail.com">
    <img alt="email" src="https://img.shields.io/badge/contact-email-yellow">
</a>

**Photorealistic synthetic data generation with AI-powered realism effects and microservices architecture.**

Generate high-quality synthetic datasets for object detection with intelligent scene understanding, depth-aware composition, and physics-based lighting simulation. Domain-agnostic with built-in presets for underwater, urban, and aerial environments.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Workflow (4 Steps)](#workflow)
- [Tools](#tools)
- [Generation Pipeline](#generation-pipeline)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Related Projects](#related-projects)

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
| **Parallel Batch Processing** | Process 4-8 images concurrently for 4x throughput |
| **Domain Gap Reduction** | FID/KID metrics, style transfer, diffusion refinement |
| **Multi-Domain Support** | Configurable domains (underwater, urban, aerial, custom) |
| **Auto-Tune** | Automatic effect optimization to minimize domain gap |

### AI Models

| Model | Service | Purpose |
|-------|---------|---------|
| **Depth Anything V3** | Depth (:8001) | Monocular depth estimation |
| **SAM3** | Segmentation (:8002) | Text-prompted scene segmentation |
| **LPIPS** | Augmentor (:8004) | Perceptual quality validation |

### Realism Effects Pipeline

1. **Depth-Aware Composition** - Objects scaled and blurred based on depth
2. **Multi-Light Shadow Generation** - Realistic shadows from detected light sources
3. **Adaptive Color Matching** - Histogram matching and tone adaptation
4. **Domain-Specific Effects** - Underwater caustics, fire glow, aerial haze
5. **Poisson Blending** - Seamless boundary integration
6. **Edge Smoothing** - Distance transform-based feathering

---

## Architecture

```
                              +------------------+
                              |    Frontend      |
                              |   (Vue 3 SPA)   |
                              |   Port: 3001     |
                              +--------+---------+
                                       |
                                       v
                              +------------------+
                              |    Gateway       |
                              |   (FastAPI)      |
                              |   Port: 8000     |
                              +--------+---------+
                                       |
              +------------+-----------+-----------+------------+
              |            |           |           |            |
              v            v           v           v            v
     +----------+  +------------+  +--------+  +----------+  +----------+
     |  Depth   |  |Segmentation|  |Effects |  |Augmentor |  |Domain Gap|
     |  DA-V3   |  |   SAM3     |  | (CPU)  |  |(Composer)|  | (Metrics)|
     |  :8001   |  |   :8002    |  | :8003  |  |  :8004   |  |  :8005   |
     +----------+  +------------+  +--------+  +----------+  +----------+
```

### Services

| Service | Port | GPU | Description |
|---------|------|-----|-------------|
| **Gateway** | 8000 | No | API orchestration, routing, domain management |
| **Depth** | 8001 | Yes | Depth Anything V3 inference |
| **Segmentation** | 8002 | Yes | SAM3 scene analysis + object extraction |
| **Effects** | 8003 | No | Lighting, caustics, blur, color correction |
| **Augmentor** | 8004 | Yes | Image composition and validation |
| **Domain Gap** | 8005 | Yes | FID/KID metrics, style transfer, diffusion |
| **Frontend** | 3001 | No | Vue 3 + Tailwind CSS web interface |

---

## Quick Start

### Prerequisites

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with 12+ GB VRAM
- 32+ GB RAM

### Launch

```bash
# Clone and start
git clone <repo-url>
cd synthetic_dataset_generator

# Configure environment
cp .env.microservices .env
# Edit .env: set DATASET_PATH, HF_TOKEN

# Start all services
docker-compose -f docker-compose.microservices.yml up -d

# Open browser
open http://localhost:3001
```

### Verify Services

```bash
curl http://localhost:8000/health
```

---

## Workflow

The application follows a streamlined 4-step workflow:

```
  [1. PREPARE]  -->  [2. CONFIGURE]  -->  [3. GENERATE]  -->  [4. EXPORT]
   Analyze &         Effects &            Launch batch        Export, combine,
   select sources    object sizes         + monitor           split & balance
```

### Step 1: Prepare (`/prepare`)

| Tab | Description |
|-----|-------------|
| **Analyze** | Upload/select a COCO JSON dataset to analyze class distribution. Or skip to "Custom Objects Mode" if you only have object cutouts. |
| **Sources** | Select background images directory, object sources directory, and output path. In custom mode, select which object classes to include. |

### Step 2: Configure (`/configure`)

| Tab | Description |
|-----|-------------|
| **Effects & Placement** | Choose a preset (Underwater, Urban, Minimal, Aggressive) or manually configure 15+ effects across 4 categories: Basic Augmentation, Underwater & Realism, Blending & Edges, Object Placement. Enable Auto-Tune for automatic domain gap optimization. |
| **Object Sizes** | Configure real-world size ratios per class. Load presets (Marine Life, Vehicles, Common Objects) or add custom classes. |

### Step 3: Generate (`/generate`)

- Set target image count **per category**
- Configure validation thresholds (quality, physics, anomaly detection)
- Batch processing options (concurrency, VRAM threshold)
- Dataset metadata (name, version, license)
- Real-time progress monitoring with ETA

### Step 4: Export (`/export`)

| Tab | Description |
|-----|-------------|
| **Export** | Convert to YOLO, Pascal VOC, or COCO format |
| **Combine** | Merge multiple datasets (synthetic + real) |
| **Split** | Train/val/test split (percentage or K-Fold) |
| **Balance** | Oversample, undersample, or weight classes |

---

## Tools

Standalone tools accessible from the sidebar:

### Object Extraction

Extract object cutouts from annotated datasets for use as synthetic generation sources. Supports SAM3-enhanced segmentation for precise boundaries.

### Domain Gap Reduction

Measure and reduce the gap between synthetic and real data:

| Feature | Description |
|---------|-------------|
| **Reference Images** | Upload real-world reference sets |
| **Validation** | Compute FID, KID, RADIO-MMD, color distribution metrics |
| **Domain Randomization** | Apply random augmentations to increase variety |
| **Style Transfer** | Transfer visual style from real images |
| **Diffusion Refinement** | Use ControlNet, IP-Adapter, or LoRA to close the gap |

### System Monitor

Monitor all running jobs and service health from a unified dashboard:
- Filter jobs by status/type
- View logs, retry failed jobs, resume interrupted batches
- Real-time service health with GPU/memory info

---

## Generation Pipeline

```
[Background] + [Objects]
        |
        v
+-- Depth Service ----> Depth Map (0-1)
        |
        v
+-- Augmentor --------> Lighting Estimation
        |                     |
        v                     v
    Object Placement    Shadow Generation
    (depth-aware)       (multi-light)
        |
        v
+-- Effects ----------> Color Correction
        |                Blur Matching
        |                Underwater Tint
        |                Caustics
        |                Edge Smoothing
        v
    Quality Validation
    (LPIPS + Physics)
        |
        v
    [Output Image + COCO Annotations]
```

### Performance

| Images | Sequential | Parallel (4x) | Speedup |
|--------|------------|---------------|---------|
| 10 | 60-120s | 15-30s | 4x |
| 100 | 10-20 min | 2.5-5 min | 4x |
| 1000 | 100-200 min | 25-50 min | 4x |

---

## Configuration

### Domain Configuration

Domains define the visual rules for generation. Located in `config/domains/`:

| Domain | Description |
|--------|-------------|
| `underwater.json` | Marine debris detection (master domain) |
| `fire_smoke.json` | Fire and smoke scenarios |
| `aerial_birds.json` | Aerial bird detection |
| `_template.json` | Template for custom domains |

Each domain configures: regions, objects, physics, effects, presets, and domain gap profiles.

### Effects Presets

| Preset | Use Case |
|--------|----------|
| **Underwater Marine** | Caustics, color tinting, water attenuation |
| **Urban Street** | Motion blur, perspective, gradient lighting |
| **Minimal** | Light augmentation for subtle variations |
| **Aggressive** | Heavy augmentation for maximum variety |

### Resource Allocation

| Service | CPU | RAM | GPU VRAM |
|---------|-----|-----|----------|
| Gateway | 2 cores | 4 GB | - |
| Depth | 3 cores | 12 GB | Shared |
| Segmentation | 3 cores | 12 GB | Shared |
| Effects | 2 cores | 4 GB | - |
| Augmentor | 8 cores | 24 GB | Shared |
| Domain Gap | 4 cores | 16 GB | Shared |

---

## API Reference

### Core Endpoints

```
GET  /health                     # Service health check
POST /generate/image             # Generate single image
POST /generate/batch             # Generate batch (async)
GET  /jobs/all                   # List all jobs
GET  /jobs/{job_id}              # Job status
```

### Generation

```
POST /augment/compose            # Compose single image
POST /augment/compose-batch      # Batch composition (async)
POST /augment/validate           # Validate image quality
POST /augment/lighting           # Estimate lighting
```

### Object Extraction

```
POST /segmentation/extract/objects          # Extract objects (async)
POST /segmentation/extract/single-object    # Extract single object
POST /segmentation/extract/analyze-dataset  # Analyze annotation types
```

### Domain Gap

```
POST /domain-gap/references/upload    # Upload reference images
POST /domain-gap/metrics/compute      # Compute FID/KID
POST /domain-gap/randomize/apply      # Domain randomization
POST /domain-gap/style-transfer/apply # Style transfer
POST /domain-gap/diffusion/refine     # Diffusion refinement
POST /domain-gap/lora/train           # Train LoRA adapter
```

### Domains

```
GET  /domains                    # List domains
POST /domains/{id}/activate      # Activate domain
POST /domains                    # Create custom domain
```

### Datasets

```
POST /datasets/analyze           # Analyze dataset
POST /datasets/combine           # Combine datasets
POST /datasets/split             # Train/val/test split
POST /datasets/kfold             # K-Fold split
POST /datasets/export            # Export to YOLO/VOC
```

---

## Related Projects

### Annotation Tool

The labeling and annotation features have been extracted to a standalone tool:

**Repository:** `annotation-tool` (same workspace)

Features:
- Auto Labeling with SAM3
- Visual COCO annotation editor (similar to CVAT)
- Label Manager (rename, delete categories)
- SAM3 bbox-to-mask conversion

The annotation tool shares the SAM3 segmentation service but operates independently with its own lightweight gateway.

---

## License

Apache 2.0
