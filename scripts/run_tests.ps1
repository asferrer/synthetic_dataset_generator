#
# Test Runner Script for Synthetic Dataset Generator (PowerShell)
#
# Usage:
#   .\scripts\run_tests.ps1           # Run all tests
#   .\scripts\run_tests.ps1 unit      # Run unit tests only
#   .\scripts\run_tests.ps1 integration # Run integration tests only
#   .\scripts\run_tests.ps1 e2e       # Run e2e tests only
#

param(
    [string]$TestType = "all"
)

$ErrorActionPreference = "Stop"

# Get script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Set-Location $ProjectRoot

Write-Host "==========================================" -ForegroundColor Blue
Write-Host "  Synthetic Dataset Generator - Tests" -ForegroundColor Blue
Write-Host "==========================================" -ForegroundColor Blue
Write-Host ""

function Run-UnitTests {
    Write-Host "[UNIT] Running unit tests..." -ForegroundColor Green

    & pytest tests/unit/ -v --tb=short

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Unit tests failed!" -ForegroundColor Red
        return $false
    }

    Write-Host "Unit tests passed!" -ForegroundColor Green
    return $true
}

function Run-IntegrationTests {
    Write-Host "[INTEGRATION] Running integration tests..." -ForegroundColor Green

    # Check if Docker is available
    try {
        & docker version | Out-Null
    }
    catch {
        Write-Host "Docker not found. Skipping integration tests." -ForegroundColor Yellow
        return $true
    }

    # Build Docker images
    Write-Host "Building Docker images..." -ForegroundColor Blue
    & docker compose -f docker-compose.microservices.yml build

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Docker build failed!" -ForegroundColor Red
        return $false
    }

    # Start services
    Write-Host "Starting services..." -ForegroundColor Blue
    & docker compose -f docker-compose.microservices.yml up -d

    # Wait for services
    Write-Host "Waiting for services to be healthy..." -ForegroundColor Blue
    & powershell -File "$ScriptDir\wait_for_services.ps1"
    $WaitResult = $LASTEXITCODE

    if ($WaitResult -ne 0) {
        Write-Host "Services failed to start!" -ForegroundColor Red
        & docker compose -f docker-compose.microservices.yml logs
        & docker compose -f docker-compose.microservices.yml down
        return $false
    }

    # Run integration tests
    & pytest tests/integration/ -v --tb=short -m "integration"
    $TestResult = $LASTEXITCODE

    # Cleanup
    Write-Host "Stopping services..." -ForegroundColor Blue
    & docker compose -f docker-compose.microservices.yml down

    if ($TestResult -ne 0) {
        Write-Host "Integration tests failed!" -ForegroundColor Red
        return $false
    }

    Write-Host "Integration tests passed!" -ForegroundColor Green
    return $true
}

function Run-E2ETests {
    Write-Host "[E2E] Running end-to-end tests..." -ForegroundColor Green

    # Check if Docker is available
    try {
        & docker version | Out-Null
    }
    catch {
        Write-Host "Docker not found. Skipping e2e tests." -ForegroundColor Yellow
        return $true
    }

    # Check if services are already running
    $CleanupNeeded = $false
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5 -ErrorAction SilentlyContinue
        Write-Host "Services already running." -ForegroundColor Blue
    }
    catch {
        Write-Host "Starting services for E2E tests..." -ForegroundColor Blue
        & docker compose -f docker-compose.microservices.yml up -d
        & powershell -File "$ScriptDir\wait_for_services.ps1"
        $CleanupNeeded = $true
    }

    # Run e2e tests
    & pytest tests/e2e/ -v --tb=short -m "e2e"
    $TestResult = $LASTEXITCODE

    # Cleanup if we started the services
    if ($CleanupNeeded) {
        Write-Host "Stopping services..." -ForegroundColor Blue
        & docker compose -f docker-compose.microservices.yml down
    }

    if ($TestResult -ne 0) {
        Write-Host "E2E tests failed!" -ForegroundColor Red
        return $false
    }

    Write-Host "E2E tests passed!" -ForegroundColor Green
    return $true
}

# Main execution
switch ($TestType.ToLower()) {
    "unit" {
        $result = Run-UnitTests
        if (-not $result) { exit 1 }
    }
    "integration" {
        $result = Run-IntegrationTests
        if (-not $result) { exit 1 }
    }
    "e2e" {
        $result = Run-E2ETests
        if (-not $result) { exit 1 }
    }
    "all" {
        Write-Host "Running all tests..." -ForegroundColor Blue
        Write-Host ""

        $UnitResult = Run-UnitTests
        $IntegrationResult = Run-IntegrationTests
        $E2EResult = Run-E2ETests

        Write-Host ""
        Write-Host "==========================================" -ForegroundColor Blue
        Write-Host "               Test Summary" -ForegroundColor Blue
        Write-Host "==========================================" -ForegroundColor Blue

        if ($UnitResult) {
            Write-Host "  Unit Tests:        PASSED" -ForegroundColor Green
        } else {
            Write-Host "  Unit Tests:        FAILED" -ForegroundColor Red
        }

        if ($IntegrationResult) {
            Write-Host "  Integration Tests: PASSED" -ForegroundColor Green
        } else {
            Write-Host "  Integration Tests: FAILED" -ForegroundColor Red
        }

        if ($E2EResult) {
            Write-Host "  E2E Tests:         PASSED" -ForegroundColor Green
        } else {
            Write-Host "  E2E Tests:         FAILED" -ForegroundColor Red
        }

        Write-Host ""

        if ($UnitResult -and $IntegrationResult -and $E2EResult) {
            Write-Host "All tests passed successfully!" -ForegroundColor Green
            exit 0
        } else {
            Write-Host "Some tests failed!" -ForegroundColor Red
            exit 1
        }
    }
    default {
        Write-Host "Usage: .\run_tests.ps1 [unit|integration|e2e|all]"
        exit 1
    }
}
