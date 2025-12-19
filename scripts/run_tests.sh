#!/bin/bash
#
# Test Runner Script for Synthetic Dataset Generator
#
# Usage:
#   ./scripts/run_tests.sh           # Run all tests
#   ./scripts/run_tests.sh unit      # Run unit tests only
#   ./scripts/run_tests.sh integration # Run integration tests only
#   ./scripts/run_tests.sh e2e       # Run e2e tests only
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo -e "${BLUE}=========================================="
echo -e "  Synthetic Dataset Generator - Tests"
echo -e "==========================================${NC}"
echo ""

# Parse arguments
TEST_TYPE="${1:-all}"

run_unit_tests() {
    echo -e "${GREEN}[UNIT] Running unit tests...${NC}"
    pytest tests/unit/ -v --tb=short -m "unit or not (integration or e2e or slow)"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Unit tests failed!${NC}"
        return 1
    fi
    echo -e "${GREEN}Unit tests passed!${NC}"
    return 0
}

run_integration_tests() {
    echo -e "${GREEN}[INTEGRATION] Running integration tests...${NC}"

    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        echo -e "${YELLOW}Docker not found. Skipping integration tests.${NC}"
        return 0
    fi

    # Build Docker images
    echo -e "${BLUE}Building Docker images...${NC}"
    docker compose -f docker-compose.microservices.yml build
    if [ $? -ne 0 ]; then
        echo -e "${RED}Docker build failed!${NC}"
        return 1
    fi

    # Start services
    echo -e "${BLUE}Starting services...${NC}"
    docker compose -f docker-compose.microservices.yml up -d

    # Wait for services
    echo -e "${BLUE}Waiting for services to be healthy...${NC}"
    "$SCRIPT_DIR/wait_for_services.sh"
    WAIT_RESULT=$?

    if [ $WAIT_RESULT -ne 0 ]; then
        echo -e "${RED}Services failed to start!${NC}"
        docker compose -f docker-compose.microservices.yml logs
        docker compose -f docker-compose.microservices.yml down
        return 1
    fi

    # Run integration tests
    pytest tests/integration/ -v --tb=short -m "integration"
    TEST_RESULT=$?

    # Cleanup
    echo -e "${BLUE}Stopping services...${NC}"
    docker compose -f docker-compose.microservices.yml down

    if [ $TEST_RESULT -ne 0 ]; then
        echo -e "${RED}Integration tests failed!${NC}"
        return 1
    fi

    echo -e "${GREEN}Integration tests passed!${NC}"
    return 0
}

run_e2e_tests() {
    echo -e "${GREEN}[E2E] Running end-to-end tests...${NC}"

    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        echo -e "${YELLOW}Docker not found. Skipping e2e tests.${NC}"
        return 0
    fi

    # Check if services are already running
    if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${BLUE}Starting services for E2E tests...${NC}"
        docker compose -f docker-compose.microservices.yml up -d
        "$SCRIPT_DIR/wait_for_services.sh"
        CLEANUP_NEEDED=true
    else
        echo -e "${BLUE}Services already running.${NC}"
        CLEANUP_NEEDED=false
    fi

    # Run e2e tests
    pytest tests/e2e/ -v --tb=short -m "e2e"
    TEST_RESULT=$?

    # Cleanup if we started the services
    if [ "$CLEANUP_NEEDED" = true ]; then
        echo -e "${BLUE}Stopping services...${NC}"
        docker compose -f docker-compose.microservices.yml down
    fi

    if [ $TEST_RESULT -ne 0 ]; then
        echo -e "${RED}E2E tests failed!${NC}"
        return 1
    fi

    echo -e "${GREEN}E2E tests passed!${NC}"
    return 0
}

# Main execution
case "$TEST_TYPE" in
    unit)
        run_unit_tests
        ;;
    integration)
        run_integration_tests
        ;;
    e2e)
        run_e2e_tests
        ;;
    all)
        echo -e "${BLUE}Running all tests...${NC}"
        echo ""

        # Run unit tests first (no Docker needed)
        run_unit_tests
        UNIT_RESULT=$?

        # Run integration tests
        run_integration_tests
        INTEGRATION_RESULT=$?

        # Run e2e tests (services should still be up)
        run_e2e_tests
        E2E_RESULT=$?

        echo ""
        echo -e "${BLUE}=========================================="
        echo -e "               Test Summary"
        echo -e "==========================================${NC}"

        if [ $UNIT_RESULT -eq 0 ]; then
            echo -e "  Unit Tests:        ${GREEN}PASSED${NC}"
        else
            echo -e "  Unit Tests:        ${RED}FAILED${NC}"
        fi

        if [ $INTEGRATION_RESULT -eq 0 ]; then
            echo -e "  Integration Tests: ${GREEN}PASSED${NC}"
        else
            echo -e "  Integration Tests: ${RED}FAILED${NC}"
        fi

        if [ $E2E_RESULT -eq 0 ]; then
            echo -e "  E2E Tests:         ${GREEN}PASSED${NC}"
        else
            echo -e "  E2E Tests:         ${RED}FAILED${NC}"
        fi

        echo ""

        if [ $UNIT_RESULT -eq 0 ] && [ $INTEGRATION_RESULT -eq 0 ] && [ $E2E_RESULT -eq 0 ]; then
            echo -e "${GREEN}All tests passed successfully!${NC}"
            exit 0
        else
            echo -e "${RED}Some tests failed!${NC}"
            exit 1
        fi
        ;;
    *)
        echo "Usage: $0 [unit|integration|e2e|all]"
        exit 1
        ;;
esac
