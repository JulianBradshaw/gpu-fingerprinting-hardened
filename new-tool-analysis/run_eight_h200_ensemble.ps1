<# Collect a matched eight-H200 run and download it for offline analysis. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)] [ValidatePattern('^[1-9][0-9]*$')] [string]$InstanceId,
    [ValidateRange(60,3600)] [int]$GapSeconds = 300,
    [ValidateRange(5,25)] [int]$MaxCaptureMinutes = 16,
    [string]$PrivateKeyPath = "$env:USERPROFILE\.ssh\id_ed25519",
    [string]$LegacyDir = (Join-Path (Split-Path $PSScriptRoot -Parent) 'existing-tool-analysis'),
    [string]$DownloadDir = (Join-Path $PSScriptRoot 'eight_h200_runs')
)

& (Join-Path $PSScriptRoot 'run_two_h200_pilot.ps1') -EightGpuEnsemble -InstanceId $InstanceId `
    -GapSeconds $GapSeconds -MaxCaptureMinutes $MaxCaptureMinutes -PrivateKeyPath $PrivateKeyPath `
    -LegacyDir $LegacyDir -DownloadDir $DownloadDir
