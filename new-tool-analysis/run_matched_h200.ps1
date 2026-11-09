<#
Compare the legacy probe and hardened all mode on an existing two-H200 NVL Vast instance.
Example: .\run_matched_h200.ps1 -InstanceId 1234567 -GapSeconds 600
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)] [ValidatePattern('^[1-9][0-9]*$')] [string]$InstanceId,
    [ValidateRange(60,3600)] [int]$GapSeconds = 600,
    [string]$PrivateKeyPath = "$env:USERPROFILE\.ssh\id_ed25519",
    [string]$PackageDir = (Join-Path $PSScriptRoot 'gpu_fingerprinting_hardened_v1'),
    [string]$LegacyDir = (Join-Path (Split-Path $PSScriptRoot -Parent) 'existing-tool-analysis'),
    [string]$DownloadDir = (Join-Path $PSScriptRoot 'matched_h200_runs')
)

$runner = Join-Path $PSScriptRoot 'run_two_h200_pilot.ps1'
& $runner -InstanceId $InstanceId -GapSeconds $GapSeconds -PrivateKeyPath $PrivateKeyPath `
    -PackageDir $PackageDir -LegacyDir $LegacyDir -DownloadDir $DownloadDir -CompareLegacy
