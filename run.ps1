# PowerShell script to build the Day-11 SVM/360 fisheye demo on Windows
#
# Examples:
#   .\run.ps1
#   .\run.ps1 -Recreate
#   .\run.ps1 -Only object
#   .\run.ps1 -Py C:\path\to\python.exe
#

[CmdletBinding()]
param(
    [switch]$Recreate,
    [switch]$PrefillAnnotation,
    [ValidateSet("object", "freespace", "lines", "ignore", "")]
    [string]$Only = "",
    [string]$Py = "python"
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host ">> Checking Python environment ($Py)..." -ForegroundColor Cyan
try {
    & $Py -c "import cvat_sdk, numpy, PIL, requests, dotenv; print('ok')" | Out-Null
} catch {
    Write-Error "Python dependency check failed. Please run: pip install -r requirements.txt"
    exit 1
}

# Determine which steps to run
$runFishEye = ($Only -eq "" -or $Only -eq "object")
$runWoodScape = ($Only -eq "" -or $Only -in @("freespace", "lines", "ignore"))

if ($runFishEye) {
    Write-Host "`n>> Task A: FishEye8K object subset + golden GT (HuggingFace, no token)..." -ForegroundColor Green
    & $Py scripts/prepare_fisheye8k.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($runWoodScape) {
    Write-Host "`n>> Tasks B/C/D: WoodScape subset + golden GT (Kaggle, needs KAGGLE_API_TOKEN)..." -ForegroundColor Green
    & $Py scripts/prepare_woodscape.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "`n>> Creating CVAT tasks (split by camera_id) + golden GT job for each..." -ForegroundColor Green
$setupArgs = @("scripts/setup_cvat.py")
if ($Recreate) {
    $setupArgs += "--recreate"
}
if ($Only -ne "") {
    $setupArgs += "--only"
    $setupArgs += $Only
}

& $Py $setupArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($PrefillAnnotation) {
    Write-Host "`n>> Pre-filling annotation jobs with Ground Truth..." -ForegroundColor Green
    & $Py scripts/auto_label_as_gt.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "`n>> ALL DONE. Tasks created successfully on CVAT!" -ForegroundColor Cyan
