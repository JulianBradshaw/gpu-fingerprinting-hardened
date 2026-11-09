<# Search for up to five minutes and rent one eight-H200 instance within the hourly ceiling. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)] [ValidateRange(0.01,1000)][double]$MaxHourlyPrice,
    [switch]$WholeHostOnly,
    [ValidateRange(1,1000)][int]$MaxOfferSearchTries = 60,
    [string]$PrivateKeyPath = "$env:USERPROFILE\.ssh\id_ed25519"
)

& (Join-Path $PSScriptRoot 'provision_hardened_h200.ps1') -EightGpuEnsemble `
    -MaxHourlyPrice $MaxHourlyPrice -MaxOfferSearchTries $MaxOfferSearchTries -SearchTimeoutSeconds 300 `
    -SearchIntervalSeconds 30 -WholeHostOnly:$WholeHostOnly -PrivateKeyPath $PrivateKeyPath
