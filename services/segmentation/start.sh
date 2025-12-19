#!/bin/bash
# Segmentation Service Startup Script
# ====================================
# Handles HuggingFace login for gated models (SAM3) before starting the service

echo "=== Segmentation Service Startup ==="

# Login to HuggingFace if token is provided (non-fatal if fails)
if [ -n "$HF_TOKEN" ]; then
    echo "HF_TOKEN detected, logging in to HuggingFace..."
    if python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)" 2>/dev/null; then
        echo "HuggingFace login successful"
    else
        echo "WARNING: HuggingFace login failed. Token may be invalid or expired."
        echo "SAM3 model will not be available. Service will start with fallback segmentation."
        echo "To fix: Generate a new token at https://huggingface.co/settings/tokens"
    fi
else
    echo "WARNING: HF_TOKEN not set. SAM3 model may not load (gated model)."
    echo "Set HF_TOKEN in .env.microservices to enable SAM3 features."
fi

echo "Starting Segmentation Service..."
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8002
