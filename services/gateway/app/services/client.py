"""
HTTP Client for Service Communication

Provides async HTTP client with retry logic for communicating with
microservices (Depth, Effects, Segmentation).
"""
import os
import logging
from typing import Dict, Any, Optional

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

logger = logging.getLogger(__name__)


class ServiceClient:
    """Async HTTP client for microservice communication."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 60.0,
        max_retries: int = 3
    ):
        """Initialize service client.

        Args:
            base_url: Base URL of the service (e.g., http://depth:8001)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException))
    )
    async def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make POST request to service.

        Args:
            endpoint: API endpoint (e.g., /estimate)
            data: JSON data to send

        Returns:
            Response JSON as dictionary

        Raises:
            httpx.HTTPStatusError: If response status is 4xx or 5xx
        """
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"POST {url}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=data)
            response.raise_for_status()
            return response.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException))
    )
    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make GET request to service.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            Response JSON as dictionary
        """
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"GET {url}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException))
    )
    async def delete(self, endpoint: str) -> Dict[str, Any]:
        """Make DELETE request to service.

        Args:
            endpoint: API endpoint

        Returns:
            Response JSON as dictionary
        """
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"DELETE {url}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.delete(url)
            response.raise_for_status()
            return response.json()

    async def health_check(self) -> Dict[str, Any]:
        """Check service health.

        Returns:
            Dictionary with health status and latency
        """
        import time

        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                latency = (time.time() - start) * 1000

                if response.status_code == 200:
                    data = response.json()
                    return {
                        "healthy": True,
                        "status": data.get("status", "healthy"),
                        "latency_ms": latency,
                        "details": data
                    }
                else:
                    return {
                        "healthy": False,
                        "status": "unhealthy",
                        "latency_ms": latency,
                        "error": f"Status {response.status_code}"
                    }
        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.warning(f"Health check failed for {self.base_url}: {e}")
            return {
                "healthy": False,
                "status": "unreachable",
                "latency_ms": latency,
                "error": str(e)
            }

    async def is_healthy(self) -> bool:
        """Quick health check.

        Returns:
            True if service is healthy
        """
        result = await self.health_check()
        return result.get("healthy", False)


class ServiceRegistry:
    """Registry of all available services."""

    def __init__(self):
        """Initialize service registry from environment variables."""
        self.depth = ServiceClient(
            os.environ.get("DEPTH_SERVICE_URL", "http://depth:8001")
        )
        self.effects = ServiceClient(
            os.environ.get("EFFECTS_SERVICE_URL", "http://effects:8003")
        )
        self.segmentation = ServiceClient(
            os.environ.get("SEGMENTATION_SERVICE_URL", "http://segmentation:8002")
        )
        self.augmentor = ServiceClient(
            os.environ.get("AUGMENTOR_SERVICE_URL", "http://augmentor:8004"),
            timeout=120.0  # Longer timeout for composition
        )

    async def check_all_health(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all services.

        Returns:
            Dictionary mapping service name to health status
        """
        import asyncio

        results = await asyncio.gather(
            self.depth.health_check(),
            self.effects.health_check(),
            self.segmentation.health_check(),
            self.augmentor.health_check(),
            return_exceptions=True
        )

        return {
            "depth": results[0] if not isinstance(results[0], Exception) else {"healthy": False, "error": str(results[0])},
            "effects": results[1] if not isinstance(results[1], Exception) else {"healthy": False, "error": str(results[1])},
            "segmentation": results[2] if not isinstance(results[2], Exception) else {"healthy": False, "error": str(results[2])},
            "augmentor": results[3] if not isinstance(results[3], Exception) else {"healthy": False, "error": str(results[3])}
        }


# Global registry instance
_registry: Optional[ServiceRegistry] = None


def get_service_registry() -> ServiceRegistry:
    """Get singleton service registry."""
    global _registry
    if _registry is None:
        _registry = ServiceRegistry()
    return _registry
