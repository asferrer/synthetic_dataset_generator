#
# Wait for all services to be healthy (PowerShell)
#
# Usage: .\scripts\wait_for_services.ps1 [timeout_seconds]
#

param(
    [int]$Timeout = 300
)

$ErrorActionPreference = "SilentlyContinue"

$Interval = 5

# Services to check
$Services = @(
    @{Name="gateway"; Url="http://localhost:8000/health"},
    @{Name="depth"; Url="http://localhost:8001/health"},
    @{Name="segmentation"; Url="http://localhost:8002/health"},
    @{Name="effects"; Url="http://localhost:8003/health"},
    @{Name="augmentor"; Url="http://localhost:8004/health"},
    @{Name="frontend"; Url="http://localhost:8501"}
)

Write-Host "Waiting for services to be healthy (timeout: ${Timeout}s)..."

foreach ($service in $Services) {
    Write-Host "  Waiting for $($service.Name)... " -NoNewline

    $ServiceElapsed = 0
    $Healthy = $false

    while ($ServiceElapsed -lt $Timeout) {
        try {
            $response = Invoke-WebRequest -Uri $service.Url -TimeoutSec 10 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                $Healthy = $true
                break
            }
        }
        catch {
            # Service not ready yet
        }

        Start-Sleep -Seconds $Interval
        $ServiceElapsed += $Interval
    }

    if ($Healthy) {
        Write-Host "OK" -ForegroundColor Green
    }
    else {
        Write-Host "TIMEOUT" -ForegroundColor Red
        Write-Host "Service $($service.Name) did not become healthy within ${Timeout}s" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "All services are healthy!" -ForegroundColor Green
exit 0
