"""
Filesystem Router
=================
Endpoints for filesystem operations (directory listing, file listing, image serving).
"""

import logging
import os
from typing import List, Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/fs", tags=["Filesystem"])

# Base paths for data (can be configured via environment)
DATA_BASE_PATH = os.environ.get("DATA_BASE_PATH", "/data")


class DirectoryListResponse(BaseModel):
    """Response for directory listing"""
    directories: List[str]
    path: str


class FileListResponse(BaseModel):
    """Response for file listing"""
    files: List[str]
    path: str


class FileInfo(BaseModel):
    """Information about a file"""
    name: str
    path: str
    size: int
    is_directory: bool
    extension: Optional[str] = None


class DirectoryContentsResponse(BaseModel):
    """Response for directory contents with file info"""
    path: str
    items: List[FileInfo]


def _is_safe_path(path: str) -> bool:
    """Check if path is safe (within data directory)."""
    try:
        resolved = Path(path).resolve()
        base = Path(DATA_BASE_PATH).resolve()
        return str(resolved).startswith(str(base)) or str(resolved).startswith("/data")
    except Exception:
        return False


@router.get("/directories", response_model=DirectoryListResponse)
async def list_directories(
    path: str = Query(DATA_BASE_PATH, description="Base path to list directories from"),
    recursive: bool = Query(False, description="List directories recursively")
):
    """
    List directories in the specified path.

    Returns all subdirectories. For security, only allows listing
    within the data directory.
    """
    logger.info(f"List directories: {path}")

    # Use default path if empty or invalid
    if not path or path == "undefined":
        path = DATA_BASE_PATH

    # Check path safety
    if not _is_safe_path(path):
        # Allow common data paths
        if not path.startswith("/data"):
            raise HTTPException(status_code=403, detail="Access denied: path outside data directory")

    try:
        target_path = Path(path)

        if not target_path.exists():
            logger.warning(f"Path does not exist: {path}")
            return DirectoryListResponse(directories=[], path=path)

        if not target_path.is_dir():
            raise HTTPException(status_code=400, detail="Path is not a directory")

        directories = []

        if recursive:
            # Get all directories recursively
            for item in target_path.rglob("*"):
                if item.is_dir():
                    directories.append(str(item))
        else:
            # Get immediate subdirectories only
            for item in target_path.iterdir():
                if item.is_dir():
                    directories.append(str(item))

        directories.sort()
        return DirectoryListResponse(directories=directories, path=path)

    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except Exception as e:
        logger.error(f"Failed to list directories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files", response_model=FileListResponse)
async def list_files(
    path: str = Query(DATA_BASE_PATH, description="Path to list files from"),
    pattern: Optional[str] = Query(None, description="File pattern (e.g., *.json, *.png)"),
    recursive: bool = Query(False, description="Search recursively")
):
    """
    List files in the specified path.

    Can filter by pattern (e.g., *.json) and search recursively.
    """
    logger.info(f"List files: {path}, pattern: {pattern}")

    # Use default path if empty
    if not path or path == "undefined":
        path = DATA_BASE_PATH

    # Check path safety
    if not _is_safe_path(path):
        if not path.startswith("/data"):
            raise HTTPException(status_code=403, detail="Access denied: path outside data directory")

    try:
        target_path = Path(path)

        if not target_path.exists():
            logger.warning(f"Path does not exist: {path}")
            return FileListResponse(files=[], path=path)

        if not target_path.is_dir():
            raise HTTPException(status_code=400, detail="Path is not a directory")

        files = []
        search_pattern = pattern or "*"

        if recursive:
            for item in target_path.rglob(search_pattern):
                if item.is_file():
                    files.append(str(item))
        else:
            for item in target_path.glob(search_pattern):
                if item.is_file():
                    files.append(str(item))

        files.sort()
        return FileListResponse(files=files, path=path)

    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/contents", response_model=DirectoryContentsResponse)
async def list_directory_contents(
    path: str = Query(DATA_BASE_PATH, description="Path to list contents from"),
    show_hidden: bool = Query(False, description="Show hidden files/directories")
):
    """
    List all contents of a directory with file information.

    Returns both files and directories with size and type info.
    """
    logger.info(f"List directory contents: {path}")

    # Use default path if empty
    if not path or path == "undefined":
        path = DATA_BASE_PATH

    # Check path safety
    if not _is_safe_path(path):
        if not path.startswith("/data"):
            raise HTTPException(status_code=403, detail="Access denied: path outside data directory")

    try:
        target_path = Path(path)

        if not target_path.exists():
            logger.warning(f"Path does not exist: {path}")
            return DirectoryContentsResponse(path=path, items=[])

        if not target_path.is_dir():
            raise HTTPException(status_code=400, detail="Path is not a directory")

        items = []
        for item in target_path.iterdir():
            # Skip hidden files unless requested
            if not show_hidden and item.name.startswith("."):
                continue

            try:
                stat = item.stat()
                items.append(FileInfo(
                    name=item.name,
                    path=str(item),
                    size=stat.st_size if item.is_file() else 0,
                    is_directory=item.is_dir(),
                    extension=item.suffix if item.is_file() else None
                ))
            except (PermissionError, OSError):
                # Skip items we can't stat
                continue

        # Sort: directories first, then files, alphabetically
        items.sort(key=lambda x: (not x.is_directory, x.name.lower()))
        return DirectoryContentsResponse(path=path, items=items)

    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except Exception as e:
        logger.error(f"Failed to list contents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/exists")
async def check_path_exists(path: str):
    """Check if a path exists."""
    logger.info(f"Check path exists: {path}")

    try:
        target_path = Path(path)
        exists = target_path.exists()
        is_file = target_path.is_file() if exists else False
        is_dir = target_path.is_dir() if exists else False

        return {
            "path": path,
            "exists": exists,
            "is_file": is_file,
            "is_directory": is_dir
        }
    except Exception as e:
        logger.error(f"Failed to check path: {e}")
        return {
            "path": path,
            "exists": False,
            "is_file": False,
            "is_directory": False,
            "error": str(e)
        }


# Supported image extensions for serving
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.tif'}


@router.get("/images")
async def serve_image(path: str = Query(..., description="Path to image file")):
    """
    Serve an image file.

    Returns the image file for use in img tags.
    Only serves files from the data directory for security.
    """
    logger.info(f"Serve image: {path}")

    try:
        target_path = Path(path)

        # Security check - must be within data directory
        if not _is_safe_path(path):
            if not path.startswith("/data"):
                raise HTTPException(status_code=403, detail="Access denied: path outside data directory")

        # Check if file exists
        if not target_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")

        if not target_path.is_file():
            raise HTTPException(status_code=400, detail="Path is not a file")

        # Check if it's an image file
        ext = target_path.suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Not a supported image format: {ext}")

        # Determine media type
        media_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff',
        }
        media_type = media_types.get(ext, 'application/octet-stream')

        return FileResponse(
            path=str(target_path),
            media_type=media_type,
            filename=target_path.name
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve image: {e}")
        raise HTTPException(status_code=500, detail=str(e))
