"""
Standardized job lifecycle manager.

Wraps JobDatabase + JobLogger with a consistent interface for all services.
Status transitions: queued -> running -> completed/failed/cancelled/interrupted
"""

import time
from datetime import datetime
from typing import Optional, Dict, Any

from .job_database import JobDatabase, get_job_db
from .job_logger import JobLogger


class JobLifecycle:
    """Manages a single job's lifecycle with DB persistence and logging."""

    def __init__(self, job_id: str, db: Optional[JobDatabase] = None):
        self.job_id = job_id
        self.db = db or get_job_db()
        self.logger = JobLogger(job_id, self.db)
        self._start_time: Optional[float] = None

    @classmethod
    def create(
        cls,
        job_id: str,
        job_type: str,
        service: str,
        request_params: Optional[Dict] = None,
        total_items: int = 0,
        output_path: Optional[str] = None,
        db: Optional[JobDatabase] = None,
    ) -> "JobLifecycle":
        """Create a new job in the database. Returns a JobLifecycle instance."""
        db = db or get_job_db()
        db.create_job(job_id, job_type, service, request_params, total_items, output_path)
        lifecycle = cls(job_id, db)
        lifecycle.logger.info("Job created", {"job_type": job_type, "service": service})
        return lifecycle

    def start(self, message: str = "Job started"):
        """Mark job as running."""
        self._start_time = time.time()
        self.db.update_job_status(self.job_id, "running", started_at=datetime.now())
        self.logger.start(message)

    def update_progress(
        self,
        processed: int,
        failed: int = 0,
        current_item: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        """Update job progress in the database."""
        self.db.update_job_progress(self.job_id, processed, failed, current_item, details)

    def complete(self, result_summary: Optional[Dict] = None, message: str = "Job completed"):
        """Mark job as successfully completed."""
        elapsed = self._elapsed_ms()
        self.db.complete_job(
            self.job_id, "completed",
            result_summary=result_summary,
            processing_time_ms=elapsed,
        )
        self.logger.complete(message, result_summary)

    def fail(self, error_message: str):
        """Mark job as failed."""
        elapsed = self._elapsed_ms()
        self.db.complete_job(
            self.job_id, "failed",
            error_message=error_message,
            processing_time_ms=elapsed,
        )
        self.logger.fail(f"Job failed: {error_message}")

    def cancel(self, result_summary: Optional[Dict] = None):
        """Mark job as cancelled."""
        elapsed = self._elapsed_ms()
        self.db.complete_job(
            self.job_id, "cancelled",
            result_summary=result_summary,
            processing_time_ms=elapsed,
        )
        self.logger.info("Job cancelled")

    def interrupt(self, reason: str = "Service restarted"):
        """Mark job as interrupted (resumable)."""
        elapsed = self._elapsed_ms()
        self.db.complete_job(
            self.job_id, "interrupted",
            error_message=reason,
            processing_time_ms=elapsed,
        )
        self.logger.warning(f"Job interrupted: {reason}")

    def _elapsed_ms(self) -> float:
        if self._start_time is not None:
            return (time.time() - self._start_time) * 1000
        return 0.0
