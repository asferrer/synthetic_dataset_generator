"""
Session state persistence for Streamlit frontend.

Saves key session state values to disk so they survive page refreshes.
Only lightweight keys are persisted (IDs, step numbers) - not large data like COCO JSON.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict

STATE_FILE = os.environ.get(
    "FRONTEND_STATE_PATH",
    os.path.join(os.environ.get("SHARED_PATH", "/shared"), "frontend_state.json"),
)

# Keys that should persist across page refreshes
PERSISTENT_KEYS = [
    "workflow_step",
    "workflow_completed",
    "current_job_id",
    "active_dataset_id",
    "generated_output_dir",
    "generation_source_mode",
    "nav_menu",
]


def save_persistent_state(session_state: dict) -> None:
    """Save selected session_state keys to disk."""
    data = {}
    for k in PERSISTENT_KEYS:
        if k in session_state:
            val = session_state[k]
            # Only serialize JSON-safe values
            if isinstance(val, (str, int, float, bool, list)):
                data[k] = val
    try:
        Path(STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception:
        pass  # Non-critical: don't break the app


def load_persistent_state() -> Dict[str, Any]:
    """Load persisted state from disk."""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def restore_session_state(session_state) -> None:
    """Restore persistent state into Streamlit session_state.

    Only restores keys that are NOT already set in session_state
    (so explicit user actions always take priority).
    """
    saved = load_persistent_state()
    for key, value in saved.items():
        if key not in session_state:
            session_state[key] = value
