<#
Run a two-device pilot on an existing Vast instance with two H200 NVLs and a CUDA development image.
Example: .\run_two_h200_pilot.ps1 -InstanceId 1234567 -GapSeconds 600
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)] [ValidatePattern('^[1-9][0-9]*$')] [string]$InstanceId,
    [ValidateRange(60,3600)] [int]$GapSeconds = 600,
    [string]$PrivateKeyPath = "$env:USERPROFILE\.ssh\id_ed25519",
    [string]$PackageDir = (Join-Path $PSScriptRoot 'gpu_fingerprinting_hardened_v1'),
    [string]$DownloadDir = (Join-Path $PSScriptRoot 'two_h200_runs'),
    [switch]$CompareLegacy,
    [switch]$EightGpuEnsemble,
    [ValidateRange(5,25)] [int]$MaxCaptureMinutes = 16,
    [string]$LegacyDir = (Join-Path (Split-Path $PSScriptRoot -Parent) 'existing-tool-analysis')
)

$ErrorActionPreference = 'Stop'
if ($CompareLegacy -and $EightGpuEnsemble) { throw 'Select one experiment variant.' }
$PSNativeCommandUseErrorActionPreference = $false
$clock = [Diagnostics.Stopwatch]::StartNew()
$token = [guid]::NewGuid().ToString('N')
$runPath = "/workspace/fp-pilot-$token"
$dest = Join-Path $DownloadDir "inst${InstanceId}_$token"
$runCreated = $false

function Write-Status([string]$Message) {
    Write-Host ('[{0:hh\:mm\:ss}] {1}' -f $clock.Elapsed,$Message)
}

function Invoke-External([string]$Command, [string[]]$Arguments, [switch]$Live) {
    $ErrorActionPreference = 'Continue'
    $LASTEXITCODE = 0
    $lines = @(& $Command @Arguments 2>&1 | ForEach-Object {
        $line = [string]$_
        if ($Live) { Write-Host $line }
        $line
    })
    [pscustomobject]@{ Exit = $LASTEXITCODE; Text = $lines -join "`n" }
}

function Invoke-Ssh([string]$RemoteCommand, [switch]$Live) {
    Invoke-External ssh ($sshOptions + @('-p',$sshPort,$hostSpec,$RemoteCommand)) -Live:$Live
}

function Invoke-Stage([string]$Stage) {
    $command = "bash $runPath/two_h200_host.sh $Stage $runPath $GapSeconds"
    if ($CompareLegacy) { $command += ' matched' }
    if ($EightGpuEnsemble) {
        $command += ' ensemble'
        if ($Stage -eq 'pilot') { $command = "timeout --kill-after=10s $($MaxCaptureMinutes*60)s $command" }
    }
    $reply = Invoke-Ssh $command -Live
    if ($reply.Exit -ne 0 -or $reply.Text -cnotmatch "(?m)^PILOT_OK $Stage\r?$") {
        throw "Pilot stage '$Stage' failed (SSH exit $($reply.Exit)). See the output above."
    }
    $reply
}

function Save-Run {
    Write-Status 'Packaging the evidence and checking the download hash.'
    $archive = Invoke-Stage archive
    if ($archive.Text -notmatch '(?m)^ARCHIVE_SHA256=([a-f0-9]{64})\r?$') {
        throw 'The host did not return the archive hash.'
    }
    $expected = $Matches[1]
    $localArchive = Join-Path $dest 'run.tar.gz'
    $partial = "$localArchive.partial"
    for ($attempt=1; $attempt -le 3; $attempt++) {
        Write-Status "Download attempt $attempt/3."
        $transfer = Invoke-External scp ($sshOptions + @('-P',$sshPort,"${hostSpec}:$runPath.tar.gz",$partial)) -Live
        if ($transfer.Exit -eq 0 -and (Test-Path -LiteralPath $partial -PathType Leaf) -and
            (Get-FileHash -Algorithm SHA256 -LiteralPath $partial).Hash -eq $expected) {
            Move-Item -LiteralPath $partial -Destination $localArchive
            Write-Status "Verified archive: $localArchive"
            return
        }
        if ($attempt -lt 3) { Start-Sleep -Seconds 5 }
    }
    throw "Download could not be verified. Remote archive: $runPath.tar.gz"
}

try {
    foreach ($tool in @('vastai','ssh','scp')) { Get-Command $tool -ErrorAction Stop | Out-Null }
    $PrivateKeyPath = (Resolve-Path -LiteralPath $PrivateKeyPath).Path
    if (-not (Test-Path -LiteralPath $PrivateKeyPath -PathType Leaf)) { throw 'SSH identity must be a private key file.' }
    $code = Join-Path $dest 'code'
    New-Item -ItemType Directory -Path $code -Force | Out-Null
    foreach ($file in @('probe_hardened.cu','chacha20_core.h','crypto_selftest.cpp','collect.py','fingerprint.py',
                        'manage.py','evaluate.py','test_hardening.py','requirements.txt')) {
        Copy-Item -LiteralPath (Join-Path $PackageDir $file) -Destination $code
    }
    $helperNames = @('two_h200_host.sh','two_h200_analysis.py')
    if ($CompareLegacy -or $EightGpuEnsemble) {
        foreach ($file in @('campaign_v4.py','atomic_race_probe_batch_v2.cu')) {
            Copy-Item -LiteralPath (Join-Path $LegacyDir $file) -Destination $code
        }
        $helperNames += 'matched_h200.py'
    }
    if ($EightGpuEnsemble) { $helperNames += 'eight_h200_ensemble.py' }
    $helpers = @($helperNames | ForEach-Object {
        $path = Join-Path $dest $_
        $source = [IO.File]::ReadAllText((Join-Path $PSScriptRoot $_)).Replace("`r`n","`n")
        [IO.File]::WriteAllText($path,$source,[Text.UTF8Encoding]::new($false))
        $path
    })

    $sshOptions = @('-i',$PrivateKeyPath,'-o','IdentitiesOnly=yes','-o','BatchMode=yes',
        '-o','StrictHostKeyChecking=accept-new','-o','ConnectTimeout=10',
        '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=8')
    Write-Status "Waiting for SSH on existing instance $InstanceId."
    $ready = $false
    for ($attempt=1; $attempt -le 36; $attempt++) {
        $endpoint = Invoke-External vastai @('ssh-url',$InstanceId)
        if ($endpoint.Exit -eq 0 -and $endpoint.Text.Trim() -cmatch '^ssh://([a-zA-Z0-9_][a-zA-Z0-9_.-]*)@([a-zA-Z0-9][a-zA-Z0-9.-]*):([0-9]{1,5})/?$') {
            $hostSpec = "$($Matches[1])@$($Matches[2])"
            $sshPort = [int]$Matches[3]
            if ($sshPort -ge 1 -and $sshPort -le 65535) {
                $probe = Invoke-Ssh 'true'
                if ($probe.Exit -eq 0) { $ready = $true; break }
            }
        }
        Write-Status "SSH not ready ($attempt/36)."
        if ($attempt -lt 36) { Start-Sleep -Seconds 8 }
    }
    if (-not $ready) { throw 'SSH did not become available. Check the instance and SSH key.' }

    $mkdir = Invoke-Ssh "mkdir -- $runPath"
    if ($mkdir.Exit -ne 0) { throw "Could not create run directory: $($mkdir.Text)" }
    $runCreated = $true
    Write-Status "Uploading probe package and pilot helpers to $runPath."
    $upload = Invoke-External scp ($sshOptions + @('-P',$sshPort,'-r',$code) + $helpers + @("${hostSpec}:$runPath/")) -Live
    if ($upload.Exit -ne 0) { throw "Upload failed (exit $($upload.Exit))." }

    $failure = $null
    try {
        Write-Status 'Checking GPUs, running CPU tests and compiling the probe.'
        Invoke-Stage setup | Out-Null
        if ($EightGpuEnsemble) { Write-Status 'Smoke testing both probes on all eight GPUs.' }
        elseif ($CompareLegacy) { Write-Status 'Smoke testing the legacy probe and hardened all mode on both GPUs.' }
        else { Write-Status 'Smoke testing both modes on both GPUs.' }
        $smoke = Invoke-Stage smoke
        if ($smoke.Text -match '(?m)^est_collection_seconds=([0-9]{1,9})\r?$') {
            $seconds = [int]$Matches[1]
            Write-Status ('Rough estimate: {0:N1} min collection plus {1:N1} min planned gap, plus analysis and transfer.' -f ($seconds/60.0),($GapSeconds/60.0))
        }
        if ($EightGpuEnsemble) {
            Write-Status "The main capture has a $MaxCaptureMinutes-minute timeout. GPU billing continues until teardown."
        }
        $go = Read-Host 'Proceed with the main capture and planned gap? (yes/no)'
        if (([string]$go).Trim() -in @('yes','y')) {
            Invoke-Stage pilot | Out-Null
        } else { Write-Status 'Saving smoke results.' }
    } catch {
        $failure = $_
        Write-Warning "$($_.Exception.Message) Attempting to preserve partial results."
    }
    Save-Run
    if ($failure) { throw $failure }
} finally {
    if ($runCreated) { Write-Status "Remote run directory: $runPath" }
    Write-Status "Manage billing and teardown for instance $InstanceId in the Vast console."
}
