<#
Rent one Vast instance with two H200 NVLs for run_two_h200_pilot.ps1.
MaxHourlyPrice applies to the entire two-GPU instance in USD per hour.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)] [ValidateRange(0.01,1000)][double]$MaxHourlyPrice,
    [switch]$WholeHostOnly,
    [ValidateRange(0,1)][double]$MinReliability = 0.99,
    [ValidateRange(0,1000000)][double]$MinDiskBandwidth = 3000,
    [ValidateRange(1,3600)][int]$SearchIntervalSeconds = 60,
    [ValidateRange(1,1000)][int]$MaxOfferSearchTries = 60,
    [string]$PrivateKeyPath = "$env:USERPROFILE\.ssh\id_ed25519"
)

$provisioner = Join-Path $PSScriptRoot 'provision_hardened_h200.ps1'
if (-not (Test-Path -LiteralPath $provisioner -PathType Leaf)) { throw "Missing shared provisioner: $provisioner" }
& $provisioner @PSBoundParameters -TwoGpuPilot
