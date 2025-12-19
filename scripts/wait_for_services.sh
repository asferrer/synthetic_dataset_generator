#!/bin/bash
#
# Wait for all services to be healthy
#
# Usage: ./scripts/wait_for_services.sh [timeout_seconds]
#

set -e

TIMEOUT="${1:-300}"  # Default 5 minutes
INTERVAL=5
ELAPSED=0

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Services to check (name:url)
SERVICES=(
    "gateway:http://localhost:8000/health"
    "depth:http://localhost:8001/health"
    "segmentation:http://localhost:8002/health"
    "effects:http://localhost:8003/health"
    "augmentor:http://localhost:8004/health"
    "frontend:http://localhost:8501"
)

echo "Waiting for services to be healthy (timeout: ${TIMEOUT}s)..."

for service_entry in "${SERVICES[@]}"; do
    NAME="${service_entry%%:*}"
    URL="${service_entry#*:}"

    echo -n "  Waiting for $NAME... "

    SERVICE_ELAPSED=0
    while [ $SERVICE_ELAPSED -lt $TIMEOUT ]; do
        if curl -s "$URL" > /dev/null 2>&1; then
            echo -e "${GREEN}OK${NC}"
            break
        fi
        sleep $INTERVAL
        SERVICE_ELAPSED=$((SERVICE_ELAPSED + INTERVAL))
    done

    if [ $SERVICE_ELAPSED -ge $TIMEOUT ]; then
        echo -e "${RED}TIMEOUT${NC}"
        echo -e "${RED}Service $NAME did not become healthy within ${TIMEOUT}s${NC}"
        exit 1
    fi
done

echo ""
echo -e "${GREEN}All services are healthy!${NC}"
exit 0
