"""
SQLite database module for job persistence across services.
Thread-safe implementation with connection pooling.
"""

import sqlite3
import json
import threading
import os
from datetime import datetime
from typing import Optional, Dict, List, Any
from contextlib import contextmanager
from pathlib import Path


# Default database path
DEFAULT_DB_PATH = os.environ.get("JOBS_DB_PATH", "/shared/db/jobs.db")

# Thread-local storage for connections
_local = threading.local()

# Global database instance
_db_instance: Optional["JobDatabase"] = None
_db_lock = threading.Lock()


class JobDatabase:
    """Thread-safe SQLite database for job management."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """Initialize database connection and create tables if needed."""
        self.db_path = db_path
        self._ensure_directory()
        self._init_schema()

    def _ensure_directory(self):
        """Ensure the database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        if not hasattr(_local, "connection") or _local.connection is None:
            _local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            _local.connection.row_factory = sqlite3.Row
            # Enable foreign keys
            _local.connection.execute("PRAGMA foreign_keys = ON")
        return _local.connection

    @contextmanager
    def _cursor(self):
        """Context manager for database cursor with automatic commit/rollback."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_schema(self):
        """Initialize database schema."""
        with self._cursor() as cursor:
            # Jobs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    service TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    request_params TEXT,
                    total_items INTEGER DEFAULT 0,
                    processed_items INTEGER DEFAULT 0,
                    failed_items INTEGER DEFAULT 0,
                    current_item TEXT,
                    progress_details TEXT,
                    output_path TEXT,
                    result_summary TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_time_ms REAL DEFAULT 0
                )
            """)

            # Job logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS job_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT,
                    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_job_type ON jobs(job_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_service ON jobs(service)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_job_logs_job_id ON job_logs(job_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_job_logs_timestamp ON job_logs(timestamp)")

            # Dataset metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dataset_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL UNIQUE,
                    dataset_name TEXT NOT NULL,
                    dataset_type TEXT NOT NULL,

                    coco_json_path TEXT NOT NULL,
                    images_dir TEXT NOT NULL,
                    effects_config_path TEXT,

                    num_images INTEGER NOT NULL,
                    num_annotations INTEGER NOT NULL,
                    num_categories INTEGER NOT NULL,
                    class_distribution TEXT,

                    generation_config TEXT,

                    preview_images TEXT,
                    categories TEXT,

                    created_at TIMESTAMP NOT NULL,
                    file_size_mb REAL,

                    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
                )
            """)

            # Create indexes for dataset_metadata
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dataset_job_id ON dataset_metadata(job_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dataset_type ON dataset_metadata(dataset_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dataset_created ON dataset_metadata(created_at DESC)")

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a database row to a dictionary, parsing JSON fields."""
        if row is None:
            return None

        result = dict(row)

        # Parse JSON fields
        json_fields = ["request_params", "progress_details", "result_summary"]
        for field in json_fields:
            if field in result and result[field]:
                try:
                    result[field] = json.loads(result[field])
                except (json.JSONDecodeError, TypeError):
                    pass

        return result

    # ==================== Job CRUD Operations ====================

    def create_job(
        self,
        job_id: str,
        job_type: str,
        service: str,
        request_params: Optional[Dict] = None,
        total_items: int = 0,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new job in the database."""
        now = datetime.now().isoformat()

        with self._cursor() as cursor:
            cursor.execute("""
                INSERT INTO jobs (
                    id, job_type, service, status, request_params,
                    total_items, output_path, created_at, updated_at
                ) VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, ?)
            """, (
                job_id,
                job_type,
                service,
                json.dumps(request_params) if request_params else None,
                total_items,
                output_path,
                now,
                now
            ))

        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a job by ID."""
        with self._cursor() as cursor:
            cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            return self._row_to_dict(row)

    def update_job_status(
        self,
        job_id: str,
        status: str,
        started_at: Optional[datetime] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """Update job status."""
        now = datetime.now().isoformat()

        with self._cursor() as cursor:
            if started_at:
                cursor.execute("""
                    UPDATE jobs
                    SET status = ?, started_at = ?, updated_at = ?, error_message = ?
                    WHERE id = ?
                """, (status, started_at.isoformat(), now, error_message, job_id))
            else:
                cursor.execute("""
                    UPDATE jobs
                    SET status = ?, updated_at = ?, error_message = ?
                    WHERE id = ?
                """, (status, now, error_message, job_id))

            return cursor.rowcount > 0

    def update_job_progress(
        self,
        job_id: str,
        processed_items: int,
        failed_items: int = 0,
        current_item: Optional[str] = None,
        progress_details: Optional[Dict] = None
    ) -> bool:
        """Update job progress."""
        now = datetime.now().isoformat()

        with self._cursor() as cursor:
            cursor.execute("""
                UPDATE jobs
                SET processed_items = ?, failed_items = ?, current_item = ?,
                    progress_details = ?, updated_at = ?
                WHERE id = ?
            """, (
                processed_items,
                failed_items,
                current_item,
                json.dumps(progress_details) if progress_details else None,
                now,
                job_id
            ))

            return cursor.rowcount > 0

    def complete_job(
        self,
        job_id: str,
        status: str,
        result_summary: Optional[Dict] = None,
        error_message: Optional[str] = None,
        processing_time_ms: float = 0
    ) -> bool:
        """Mark a job as completed (or failed/cancelled)."""
        now = datetime.now().isoformat()

        with self._cursor() as cursor:
            cursor.execute("""
                UPDATE jobs
                SET status = ?, completed_at = ?, result_summary = ?,
                    error_message = ?, processing_time_ms = ?, updated_at = ?
                WHERE id = ?
            """, (
                status,
                now,
                json.dumps(result_summary) if result_summary else None,
                error_message,
                processing_time_ms,
                now,
                job_id
            ))

            return cursor.rowcount > 0

    # ==================== Query Operations ====================

    def list_jobs(
        self,
        service: Optional[str] = None,
        job_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List jobs with optional filters."""
        conditions = []
        params = []

        if service:
            conditions.append("service = ?")
            params.append(service)
        if job_type:
            conditions.append("job_type = ?")
            params.append(job_type)
        if status:
            conditions.append("status = ?")
            params.append(status)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self._cursor() as cursor:
            cursor.execute(f"""
                SELECT * FROM jobs
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, params + [limit, offset])

            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_job_history(
        self,
        days: Optional[int] = None,
        job_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get job history, optionally filtered by days and type."""
        conditions = ["status IN ('completed', 'failed', 'cancelled', 'interrupted')"]
        params = []

        if days:
            conditions.append("created_at >= datetime('now', ?)")
            params.append(f"-{days} days")
        if job_type:
            conditions.append("job_type = ?")
            params.append(job_type)

        where_clause = " AND ".join(conditions)

        with self._cursor() as cursor:
            cursor.execute(f"""
                SELECT * FROM jobs
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ?
            """, params + [limit])

            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_active_jobs(self, service: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all active (queued or processing) jobs."""
        conditions = ["status IN ('queued', 'processing')"]
        params = []

        if service:
            conditions.append("service = ?")
            params.append(service)

        where_clause = " AND ".join(conditions)

        with self._cursor() as cursor:
            cursor.execute(f"""
                SELECT * FROM jobs
                WHERE {where_clause}
                ORDER BY created_at ASC
            """, params)

            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_interrupted_jobs(self, service: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all interrupted jobs that can be resumed."""
        conditions = ["status = 'interrupted'"]
        params = []

        if service:
            conditions.append("service = ?")
            params.append(service)

        where_clause = " AND ".join(conditions)

        with self._cursor() as cursor:
            cursor.execute(f"""
                SELECT * FROM jobs
                WHERE {where_clause}
                ORDER BY created_at DESC
            """, params)

            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_job_request_params(self, job_id: str) -> Optional[Dict]:
        """Get the original request parameters for a job (for retry)."""
        job = self.get_job(job_id)
        if job:
            return job.get("request_params")
        return None

    # ==================== Dataset Metadata Operations ====================

    def create_dataset_metadata(
        self,
        job_id: str,
        dataset_name: str,
        dataset_type: str,
        coco_json_path: str,
        images_dir: str,
        num_images: int,
        num_annotations: int,
        num_categories: int,
        class_distribution: Dict[str, int],
        categories: List[Dict],
        preview_images: Optional[List[str]] = None,
        generation_config: Optional[Dict] = None,
        effects_config_path: Optional[str] = None,
        file_size_mb: Optional[float] = None
    ) -> Dict[str, Any]:
        """Create dataset metadata entry after successful generation."""
        import json
        from datetime import datetime

        now = datetime.now().isoformat()

        with self._cursor() as cursor:
            cursor.execute("""
                INSERT INTO dataset_metadata (
                    job_id, dataset_name, dataset_type, coco_json_path, images_dir,
                    effects_config_path, num_images, num_annotations, num_categories,
                    class_distribution, generation_config, preview_images, categories,
                    created_at, file_size_mb
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id, dataset_name, dataset_type, coco_json_path, images_dir,
                effects_config_path, num_images, num_annotations, num_categories,
                json.dumps(class_distribution) if class_distribution else None,
                json.dumps(generation_config) if generation_config else None,
                json.dumps(preview_images) if preview_images else None,
                json.dumps(categories) if categories else None,
                now, file_size_mb
            ))

        return self.get_dataset_metadata(job_id)

    def get_dataset_metadata(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific dataset."""
        import json

        with self._cursor() as cursor:
            cursor.execute("SELECT * FROM dataset_metadata WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()

            if not row:
                return None

            result = dict(row)

            # Parse JSON fields
            json_fields = ["class_distribution", "generation_config", "preview_images", "categories"]
            for field in json_fields:
                if field in result and result[field]:
                    try:
                        result[field] = json.loads(result[field])
                    except:
                        result[field] = None

            return result

    def delete_dataset_metadata(self, job_id: str) -> bool:
        """Delete dataset metadata for a job. Returns True if deleted, False if not found."""
        with self._cursor() as cursor:
            cursor.execute("DELETE FROM dataset_metadata WHERE job_id = ?", (job_id,))
            return cursor.rowcount > 0

    def list_datasets(
        self,
        dataset_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List all datasets with metadata, optionally filtered by type."""
        import json

        where_clause = "1=1"
        params = []

        if dataset_type:
            where_clause += " AND dataset_type = ?"
            params.append(dataset_type)

        with self._cursor() as cursor:
            cursor.execute(f"""
                SELECT * FROM dataset_metadata
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, params + [limit, offset])

            rows = cursor.fetchall()
            results = []

            for row in rows:
                result = dict(row)

                # Parse JSON fields
                json_fields = ["class_distribution", "generation_config", "preview_images", "categories"]
                for field in json_fields:
                    if field in result and result[field]:
                        try:
                            result[field] = json.loads(result[field])
                        except:
                            result[field] = None

                results.append(result)

            return results

    def update_dataset_preview(
        self,
        job_id: str,
        preview_images: List[str]
    ) -> bool:
        """Update preview images for a dataset."""
        import json

        with self._cursor() as cursor:
            cursor.execute("""
                UPDATE dataset_metadata
                SET preview_images = ?
                WHERE job_id = ?
            """, (json.dumps(preview_images), job_id))

            return cursor.rowcount > 0

    # ==================== Log Operations ====================

    def add_job_log(
        self,
        job_id: str,
        level: str,
        message: str,
        details: Optional[Dict] = None
    ) -> bool:
        """Add a log entry for a job."""
        now = datetime.now().isoformat()

        with self._cursor() as cursor:
            cursor.execute("""
                INSERT INTO job_logs (job_id, timestamp, level, message, details)
                VALUES (?, ?, ?, ?, ?)
            """, (
                job_id,
                now,
                level.upper(),
                message,
                json.dumps(details) if details else None
            ))

            return cursor.rowcount > 0

    def get_job_logs(
        self,
        job_id: str,
        level: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get logs for a specific job."""
        conditions = ["job_id = ?"]
        params = [job_id]

        if level:
            conditions.append("level = ?")
            params.append(level.upper())

        where_clause = " AND ".join(conditions)

        with self._cursor() as cursor:
            cursor.execute(f"""
                SELECT * FROM job_logs
                WHERE {where_clause}
                ORDER BY timestamp ASC
                LIMIT ?
            """, params + [limit])

            logs = []
            for row in cursor.fetchall():
                log = dict(row)
                if log.get("details"):
                    try:
                        log["details"] = json.loads(log["details"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                logs.append(log)

            return logs

    # ==================== Cleanup Operations ====================

    def cleanup_old_jobs(self, days: int = 30) -> int:
        """Delete jobs older than specified days. Returns count of deleted jobs."""
        with self._cursor() as cursor:
            # First delete associated logs (due to foreign key)
            cursor.execute("""
                DELETE FROM job_logs
                WHERE job_id IN (
                    SELECT id FROM jobs
                    WHERE created_at < datetime('now', ?)
                    AND status IN ('completed', 'failed', 'cancelled')
                )
            """, (f"-{days} days",))

            # Then delete the jobs
            cursor.execute("""
                DELETE FROM jobs
                WHERE created_at < datetime('now', ?)
                AND status IN ('completed', 'failed', 'cancelled')
            """, (f"-{days} days",))

            return cursor.rowcount

    def delete_job(self, job_id: str) -> bool:
        """Delete a specific job and its logs."""
        with self._cursor() as cursor:
            cursor.execute("DELETE FROM job_logs WHERE job_id = ?", (job_id,))
            cursor.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            return cursor.rowcount > 0


def get_job_db(db_path: str = DEFAULT_DB_PATH) -> JobDatabase:
    """Get the global JobDatabase instance (singleton pattern)."""
    global _db_instance

    if _db_instance is None:
        with _db_lock:
            if _db_instance is None:
                _db_instance = JobDatabase(db_path)

    return _db_instance


def reset_job_db():
    """Reset the global database instance (useful for testing)."""
    global _db_instance
    with _db_lock:
        _db_instance = None
