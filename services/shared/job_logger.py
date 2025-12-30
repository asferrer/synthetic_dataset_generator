"""
Per-job logger that writes to the database.
Provides structured logging with job context.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any

from .job_database import JobDatabase, get_job_db


class JobLogger:
    """Logger for a specific job that writes to the database."""

    def __init__(
        self,
        job_id: str,
        db: Optional[JobDatabase] = None,
        also_print: bool = True
    ):
        """
        Initialize a job logger.

        Args:
            job_id: The ID of the job to log for
            db: Optional JobDatabase instance (uses global if not provided)
            also_print: Whether to also print logs to console
        """
        self.job_id = job_id
        self.db = db or get_job_db()
        self.also_print = also_print
        self._console_logger = logging.getLogger(f"job.{job_id}")

    def _log(
        self,
        level: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Internal log method."""
        # Write to database
        try:
            self.db.add_job_log(
                job_id=self.job_id,
                level=level,
                message=message,
                details=details
            )
        except Exception as e:
            # Don't fail silently, at least print the error
            print(f"[JobLogger] Failed to write to database: {e}")

        # Also print to console if enabled
        if self.also_print:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            details_str = f" | {details}" if details else ""
            log_line = f"[{timestamp}] [{self.job_id}] {level}: {message}{details_str}"

            if level == "ERROR":
                self._console_logger.error(log_line)
            elif level == "WARNING":
                self._console_logger.warning(log_line)
            else:
                self._console_logger.info(log_line)

    def info(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Log an INFO message."""
        self._log("INFO", message, details)

    def error(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Log an ERROR message."""
        self._log("ERROR", message, details)

    def warning(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Log a WARNING message."""
        self._log("WARNING", message, details)

    def debug(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Log a DEBUG message (only to console, not to database)."""
        if self.also_print:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            details_str = f" | {details}" if details else ""
            self._console_logger.debug(
                f"[{timestamp}] [{self.job_id}] DEBUG: {message}{details_str}"
            )

    def progress(
        self,
        current: int,
        total: int,
        message: Optional[str] = None
    ):
        """Log a progress update."""
        pct = (current / total * 100) if total > 0 else 0
        msg = message or f"Progress: {current}/{total}"
        self.info(msg, {"current": current, "total": total, "percentage": round(pct, 1)})

    def start(self, message: str = "Job started"):
        """Log job start."""
        self.info(message)

    def complete(self, message: str = "Job completed", result: Optional[Dict] = None):
        """Log job completion."""
        self.info(message, result)

    def fail(self, message: str, error: Optional[Exception] = None):
        """Log job failure."""
        details = {}
        if error:
            details["error_type"] = type(error).__name__
            details["error_message"] = str(error)
        self.error(message, details if details else None)
