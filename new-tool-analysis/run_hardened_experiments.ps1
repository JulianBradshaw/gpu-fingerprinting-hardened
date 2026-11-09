<#
Run the single-H200 sanity protocol on an existing Vast instance.
Provision with provision_hardened_h200.ps1 first. Keep host_run.sh next to this script.

Example: .\run_hardened_experiments.ps1 -InstanceId 1234567
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)] [ValidatePattern('^[1-9][0-9]*$')] [string]$InstanceId,
    [string]$HostRunScript = (Join-Path $PSScriptRoot 'host_run.sh'),
    [string]$PrivateKeyPath = "$env:USERPROFILE\.ssh\id_ed25519",
    [string]$DownloadDir = (Join-Path $PSScriptRoot 'hardened_runs'),
    [ValidateRange(0,1440)] [int]$WarnIfMainMinutesOver = 45
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$clock = [Diagnostics.Stopwatch]::StartNew()
$runToken = [guid]::NewGuid().ToString('N')
$runPath = "/workspace/fp-sanity-$runToken"
$remoteScript = "$runPath/host_run.sh"
$dest = Join-Path $DownloadDir "inst${InstanceId}_$runToken"
$runCreated = $false
$destroyed = $false

function Write-Status([string]$Message) {
    Write-Host ('[{0:hh\:mm\:ss}] {1}' -f $clock.Elapsed,$Message)
}

function Invoke-External([string]$Command, [string[]]$Arguments, [switch]$Live) {
    # Windows PowerShell can wrap native stderr in error records. Keep it as output.
    $ErrorActionPreference = 'Continue'
    $LASTEXITCODE = 0
    $lines = @(& $Command @Arguments 2>&1 | ForEach-Object {
        $line = [string]$_
        if ($Live) { Write-Host $line }
        $line
    })
    return [pscustomobject]@{ Exit = $LASTEXITCODE; Text = $lines -join "`n" }
}

function Invoke-Ssh([string]$RemoteCommand, [switch]$Live) {
    Invoke-External -Command ssh -Arguments ($sshOptions + @('-p',$sshPort,$hostSpec,$RemoteCommand)) -Live:$Live
}

function Invoke-Stage([string]$Stage, [switch]$AllowTriage) {
    $reply = Invoke-Ssh "bash $remoteScript $Stage $runPath" -Live
    if ($AllowTriage -and $reply.Exit -eq 2 -and $reply.Text -cmatch '(?m)^HOSTRUN_TRIAGE analyze\r?$') { return $reply }
    if ($reply.Exit -ne 0 -or $reply.Text -cnotmatch "(?m)^HOSTRUN_OK $Stage\r?$") {
        throw "Host stage '$Stage' failed (SSH exit $($reply.Exit)). See the output above."
    }
    return $reply
}

function Save-Run {
    Write-Status '[6/6] Packaging run evidence and verifying the download.'
    $archive = Invoke-Stage archive
    if ($archive.Text -notmatch '(?m)^ARCHIVE_SHA256=([a-f0-9]{64})\r?$') {
        throw 'The host did not return an archive checksum.'
    }
    $expected = $Matches[1]
    $localArchive = Join-Path $dest 'run.tar.gz'
    $partial = "$localArchive.partial"
    for ($attempt=1; $attempt -le 3; $attempt++) {
        Write-Status "Download attempt $attempt/3."
        $transfer = Invoke-External -Command scp -Arguments ($sshOptions + @('-P',$sshPort,"${hostSpec}:$runPath.tar.gz",$partial)) -Live
        if ($transfer.Exit -eq 0 -and (Test-Path -LiteralPath $partial -PathType Leaf)) {
            $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $partial).Hash
            if ($actual -eq $expected) {
                Move-Item -LiteralPath $partial -Destination $localArchive
                Write-Status "Verified archive: $localArchive"
                return
            }
            Write-Status 'Archive checksum mismatch.'
        }
        if ($attempt -lt 3) { Start-Sleep -Seconds 5 }
    }
    throw "Download could not be verified. Remote archive: $runPath.tar.gz"
}

try {
    foreach ($path in @($HostRunScript,$PrivateKeyPath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "File not found: $path" }
    }
    foreach ($tool in @('vastai','ssh','scp')) { Get-Command $tool -ErrorAction Stop | Out-Null }
    $PrivateKeyPath = (Resolve-Path -LiteralPath $PrivateKeyPath).Path
    New-Item -ItemType Directory -Path $dest -Force | Out-Null
    $stagedScript = Join-Path $dest 'host_run.sh'
    $scriptText = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $HostRunScript).Path).Replace("`r`n","`n")
    [IO.File]::WriteAllText($stagedScript,$scriptText,[Text.UTF8Encoding]::new($false))

    $sshOptions = @('-i',$PrivateKeyPath,'-o','IdentitiesOnly=yes','-o','BatchMode=yes',
        '-o','StrictHostKeyChecking=accept-new','-o','ConnectTimeout=10',
        '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=8')
    Write-Status "Resolving SSH endpoint for instance $InstanceId."
    $endpoint = Invoke-External -Command vastai -Arguments @('ssh-url',$InstanceId)
    if ($endpoint.Exit -ne 0 -or $endpoint.Text.Trim() -cnotmatch '^ssh://([a-zA-Z0-9_][a-zA-Z0-9_.-]*)@([a-zA-Z0-9][a-zA-Z0-9.-]*):([0-9]{1,5})/?$') {
        throw "Could not resolve SSH endpoint (exit $($endpoint.Exit)): $($endpoint.Text)"
    }
    $hostSpec = "$($Matches[1])@$($Matches[2])"
    $sshPort = [int]$Matches[3]
    if ($sshPort -lt 1 -or $sshPort -gt 65535) { throw "Invalid SSH port: $sshPort" }

    $dph = $null
    try {
        $quote = Invoke-External -Command vastai -Arguments @('show','instance',$InstanceId,'--raw')
        if ($quote.Exit -eq 0) {
            $instance = $quote.Text | ConvertFrom-Json -ErrorAction Stop
            if ($null -ne $instance.dph_total -and [double]$instance.dph_total -gt 0) { $dph = [double]$instance.dph_total }
        }
    } catch { Write-Status 'Could not read the hourly rate from Vast.' }

    $ready = $false
    for ($attempt=1; $attempt -le 40; $attempt++) {
        Write-Status "Checking SSH ($attempt/40) at ${hostSpec}:$sshPort."
        $probe = Invoke-Ssh 'true'
        if ($probe.Exit -eq 0) { $ready = $true; break }
        if ($attempt -lt 40) { Start-Sleep -Seconds 8 }
    }
    if (-not $ready) { throw "SSH did not become available: $($probe.Text)" }
    Write-Status "[1/6] Uploading host script for $runPath."
    $mkdir = Invoke-Ssh "mkdir -- $runPath"
    if ($mkdir.Exit -ne 0) { throw "Could not create run directory: $($mkdir.Text)" }
    $runCreated = $true
    $upload = Invoke-External -Command scp -Arguments ($sshOptions + @('-P',$sshPort,$stagedScript,"${hostSpec}:$remoteScript")) -Live
    if ($upload.Exit -ne 0) { throw "Host script upload failed (exit $($upload.Exit))." }

    $failure = $null
    $analysisExit = 0
    try {
        Write-Status '[2/6] Setting up dependencies, running CPU tests and compiling CUDA.'
        Invoke-Stage setup | Out-Null
        Write-Status '[3/6] Collecting smoke bursts in both modes.'
        $smoke = Invoke-Stage smoke
        if ($smoke.Text -match '(?m)^est_main_seconds=([0-9]{1,9})\r?$') {
            $seconds = [int]$Matches[1]
            $minutes = $seconds / 60.0
            Write-Status ('Rough estimate for the 384 main bursts: {0:N1} minutes.' -f $minutes)
            if ($null -ne $dph) {
                Write-Status ('At USD {0:N3}/hr, collection alone is estimated at USD {1:N2}.' -f $dph,($seconds*$dph/3600))
            } else { Write-Status 'Check the hourly rate in the Vast console.' }
            if ($WarnIfMainMinutesOver -gt 0 -and $minutes -gt $WarnIfMainMinutesOver) {
                Write-Warning "Estimated collection exceeds $WarnIfMainMinutesOver minutes."
            }
        } else { Write-Warning 'The smoke stage returned no time estimate.' }
        $go = Read-Host 'Proceed with full main collection? (yes/no)'
        if (([string]$go).Trim() -in @('yes','y')) {
            Write-Status '[4/6] Collecting registration, calibration and held-out evaluation.'
            Invoke-Stage main | Out-Null
            Write-Status '[5/6] Analyzing both modes with frozen calibration.'
            $analysis = Invoke-Stage analyze -AllowTriage
            $analysisExit = $analysis.Exit
            if ($analysisExit -eq 2) { Write-Warning 'At least one mode missed the consistency screen. Both summaries will be saved.' }
        } else { Write-Status 'Main collection skipped. Saving smoke results.' }
    } catch {
        $failure = $_
        Write-Warning "$($_.Exception.Message) Attempting to preserve partial results."
    }

    Save-Run
    if ($failure) { throw $failure }
    $kill = Read-Host "Destroy instance $InstanceId now to stop billing? (yes/no)"
    if (([string]$kill).Trim() -in @('yes','y')) {
        $teardown = Invoke-External -Command vastai -Arguments @('destroy','instance',$InstanceId) -Live
        if ($teardown.Exit -ne 0) { throw "Destroy returned exit $($teardown.Exit). Confirm instance status in the Vast console." }
        if ($teardown.Text -notmatch "(?m)^destroying instance $InstanceId\.\r?$") {
            throw 'The CLI did not return the expected teardown acknowledgement. Check the instance in the Vast console.'
        }
        $destroyed = $true
        Write-Status "Vast accepted teardown of instance $InstanceId."
    }
    if ($analysisExit -eq 2) { exit 2 }
} finally {
    if (-not $destroyed) {
        Write-Status "Check instance $InstanceId in Vast. If still active, it may be billing. Teardown: vastai destroy instance $InstanceId"
        if ($runCreated) { Write-Status "Remote run directory: $runPath" }
    }
}
