# Provision an H200 rental. The single-GPU path also uploads the sanity package.
# Example: .\provision_hardened_h200.ps1 -MaxHourlyPrice 2.50
# The example rate is illustrative, not a price recommendation or total-spend cap.
[CmdletBinding(DefaultParameterSetName='Rent')]
param(
    [Parameter(Mandatory=$true,ParameterSetName='Rent')]
    [ValidateRange(0.01,1000)][double]$MaxHourlyPrice,
    [Parameter(Mandatory=$true,ParameterSetName='Resume')]
    [ValidatePattern('^[1-9][0-9]*$')][string]$InstanceId,
    [Parameter(ParameterSetName='Rent')][switch]$TwoGpuPilot,
    [Parameter(ParameterSetName='Rent')][switch]$EightGpuEnsemble,
    [switch]$WholeHostOnly,
    [ValidateRange(0,1)][double]$MinReliability = 0.99,
    [ValidateRange(0,1000000)][double]$MinDiskBandwidth = 3000,
    [ValidateRange(1,3600)][int]$SearchIntervalSeconds = 60,
    [ValidateRange(1,1000)][int]$MaxOfferSearchTries = 60,
    [ValidateRange(0,86400)][int]$SearchTimeoutSeconds = 0,
    [string]$PrivateKeyPath = "$env:USERPROFILE\.ssh\id_ed25519"
)

$ErrorActionPreference = 'Stop'
if ($TwoGpuPilot -and $EightGpuEnsemble) { throw 'Select one rental variant.' }
# Handle native exit codes ourselves consistently across Windows PowerShell and PS7.
$PSNativeCommandUseErrorActionPreference = $false
$image = 'pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel'
$diskSize = '30'
$gpuCountRequired = if ($EightGpuEnsemble) { 8 } elseif ($TwoGpuPilot) { 2 } else { 1 }
$gpuQuery = if ($EightGpuEnsemble) { 'gpu_name=H200' } elseif ($TwoGpuPilot) { 'gpu_name=H200_NVL' } else { 'gpu_name in [H200,H200_NVL]' }
$gpuPattern = if ($EightGpuEnsemble) { '^H200$' } elseif ($TwoGpuPilot) { '^H200[ _]NVL$' } else { '^H200' }
$gpuDescription = if ($EightGpuEnsemble) { 'eight plain H200s on one instance' } elseif ($TwoGpuPilot) { 'two H200 NVLs on one instance' } else { 'one H200 or H200 NVL' }
$pkg = Join-Path $PSScriptRoot 'gpu_fingerprinting_hardened_v1'
$helper = Join-Path $PSScriptRoot 'single_h200_sanity_analysis.py'
$plan = Join-Path $PSScriptRoot 'single_h200_sanity_plan.md'
$clock = [Diagnostics.Stopwatch]::StartNew()
$iid = $InstanceId
$creationAttempted = $false
$sessionFile = $null

function Write-Status([string]$Stage, [string]$Message) {
    Write-Host ('[{0:hh\:mm\:ss}] [{1}] {2}' -f $clock.Elapsed,$Stage,$Message)
}

function Invoke-External {
    param([string]$Command, [string[]]$Arguments, [switch]$AllowFailure)
    $errFile = [IO.Path]::GetTempFileName()
    try {
        # Windows PowerShell can turn native stderr into nonterminating errors.
        # Capture it, then judge the command by its actual exit code.
        $ErrorActionPreference = 'Continue'
        $LASTEXITCODE = 0
        $lines = @(& $Command @Arguments 2> $errFile)
        $exitCode = $LASTEXITCODE
        $stderr = [IO.File]::ReadAllText($errFile)
        $stdout = ($lines | ForEach-Object { [string]$_ }) -join "`n"
        if ($exitCode -ne 0 -and -not $AllowFailure) {
            throw "$Command failed (exit $exitCode): $stderr $stdout"
        }
        return [pscustomobject]@{ ExitCode=$exitCode; Stdout=$stdout; Stderr=$stderr }
    } finally {
        Remove-Item -LiteralPath $errFile -ErrorAction SilentlyContinue
    }
}

function Invoke-VastJson([string[]]$Arguments) {
    $reply = Invoke-External -Command vastai -Arguments ($Arguments + @('--raw'))
    try {
        if ([string]::IsNullOrWhiteSpace($reply.Stdout) -or $reply.Stdout.Trim() -eq 'null') {
            throw 'empty/null response'
        }
        # Windows PowerShell 5.1 emits a JSON array as one pipeline object.
        $decoded = $reply.Stdout | ConvertFrom-Json -ErrorAction Stop
        foreach ($item in $decoded) { $item }
    } catch {
        throw "Invalid JSON response from vastai $($Arguments -join ' '): $($_.Exception.Message)"
    }
}

function Get-Number($Object, [string]$Name) {
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property -or $null -eq $property.Value) { return $null }
    $number = 0.0
    if (-not [double]::TryParse([string]$property.Value,
        [Globalization.NumberStyles]::Float,[Globalization.CultureInfo]::InvariantCulture,[ref]$number)) { return $null }
    if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) { return $null }
    return $number
}

function Test-RegisteredKey($Keys, [string]$Blob) {
    # Accept list/object response envelopes; compare the key material, not comments.
    return (($Keys | ConvertTo-Json -Depth 30 -Compress) -like "*$Blob*")
}

try {
    Write-Status '1/6 Preflight' 'Checking local files, tools and SSH identity.'
    foreach ($command in @('vastai','ssh','scp')) { Get-Command $command -ErrorAction Stop | Out-Null }
    $requiredFiles = @($PrivateKeyPath)
    $packageFiles = @('probe_hardened.cu','chacha20_core.h','collect.py','fingerprint.py','test_hardening.py','requirements.txt')
    if ($TwoGpuPilot -or $EightGpuEnsemble) {
        $runnerName = if ($EightGpuEnsemble) { 'run_eight_h200_ensemble.ps1' } else { 'run_two_h200_pilot.ps1' }
        $pilotRunner = Join-Path $PSScriptRoot $runnerName
        $requiredFiles += @($pilotRunner,(Join-Path $PSScriptRoot 'two_h200_host.sh'),(Join-Path $PSScriptRoot 'two_h200_analysis.py'))
        $packageFiles += @('crypto_selftest.cpp','manage.py','evaluate.py')
        if ($EightGpuEnsemble) {
            $legacyDir = Join-Path (Split-Path $PSScriptRoot -Parent) 'existing-tool-analysis'
            $requiredFiles += @((Join-Path $PSScriptRoot 'run_two_h200_pilot.ps1'),(Join-Path $PSScriptRoot 'matched_h200.py'),
                (Join-Path $PSScriptRoot 'eight_h200_ensemble.py'),(Join-Path $legacyDir 'campaign_v4.py'),
                (Join-Path $legacyDir 'atomic_race_probe_batch_v2.cu'))
        }
    } else {
        $requiredFiles += @($helper,$plan)
    }
    foreach ($file in $requiredFiles) {
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file" }
    }
    foreach ($name in $packageFiles) {
        if (-not (Test-Path -LiteralPath (Join-Path $pkg $name) -PathType Leaf)) { throw "Missing package file: $name" }
    }
    $PrivateKeyPath = (Resolve-Path -LiteralPath $PrivateKeyPath).Path
    $sessionDir = Join-Path $PSScriptRoot ('provision-session-'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')+'-'+[guid]::NewGuid().ToString('N').Substring(0,8))
    New-Item -ItemType Directory -Path $sessionDir | Out-Null
    $sessionFile = Join-Path $sessionDir 'session.json'
    $knownHosts = Join-Path $env:TEMP 'vast_known_hosts'
    $opt = @('-i',$PrivateKeyPath,'-o','IdentitiesOnly=yes','-o','BatchMode=yes',
        '-o','StrictHostKeyChecking=no','-o',"UserKnownHostsFile=$knownHosts",'-o','LogLevel=ERROR',
        '-o','ConnectTimeout=10','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=8')

    if ($InstanceId) {
        $session = [ordered]@{ started_utc=[DateTime]::UtcNow.ToString('o'); status='resuming';
            instance_id=$iid; manual_teardown=$true }
        $session | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $sessionFile -Encoding UTF8
        Write-Status '3/6 Resume' "Continuing setup on existing instance $iid. Session record: $sessionFile"
    } else {
        $pub = (Get-Content -LiteralPath ($PrivateKeyPath+'.pub') -Raw).Trim()
        $parts = $pub -split '\s+'
        if ($parts.Count -lt 2 -or $pub.Contains("`n") -or $parts[0] -notmatch '^(ssh-|ecdsa-|sk-)') {
            throw 'Invalid SSH public-key file.'
        }
        Write-Status '1/6 Preflight' 'Checking account SSH-key registration.'
        $keys = Invoke-VastJson -Arguments @('show','ssh-keys')
        if (-not (Test-RegisteredKey $keys $parts[1])) {
            Invoke-External -Command vastai -Arguments @('create','ssh-key',$pub) | Out-Null
            $keys = Invoke-VastJson -Arguments @('show','ssh-keys')
            if (-not (Test-RegisteredKey $keys $parts[1])) { throw 'SSH key was not found after registration; no rental created.' }
        }
        Write-Status '1/6 Preflight' 'Account key present. The private key stays local; SSH uses this identity explicitly.'
        $rate = $MaxHourlyPrice.ToString('G',[Globalization.CultureInfo]::InvariantCulture)
        $query = "$gpuQuery num_gpus=$gpuCountRequired verified=true rentable=true cuda_vers>=12.4 direct_port_count>=1 disk_space>=$diskSize dph_total<=$rate"
        # gpu_frac measures the share of the host's GPUs in the offer. Host setup checks MIG.
        if ($WholeHostOnly) { $query += ' gpu_frac>=1' }
        Write-Status '2/6 Offers' "Searching for $gpuDescription. Maximum advertised instance rate USD $rate/hr, reliability >= $MinReliability, disk bandwidth >= $MinDiskBandwidth MB/s; whole-host-only=$WholeHostOnly."
        Write-Status '2/6 Offers' 'The hourly filter is not a total-bill cap. Full-device/exclusivity checks remain part of host preflight.'
        if ($SearchTimeoutSeconds -gt 0) {
            Write-Status '2/6 Offers' "Search window: $SearchTimeoutSeconds seconds, with up to $SearchIntervalSeconds seconds between attempts."
        }
        $chosen = $null
        $searchClock = [Diagnostics.Stopwatch]::StartNew()
        for ($attempt=1; $attempt -le $MaxOfferSearchTries; $attempt++) {
            if ($SearchTimeoutSeconds -gt 0 -and $searchClock.Elapsed.TotalSeconds -ge $SearchTimeoutSeconds) { break }
            $offers = @(Invoke-VastJson -Arguments @('search','offers',$query,'-o','dph+','--storage',$diskSize))
            if ($SearchTimeoutSeconds -gt 0 -and $searchClock.Elapsed.TotalSeconds -ge $SearchTimeoutSeconds) { break }
            $eligible = @(
                foreach ($candidate in $offers) {
                    if ($null -eq $candidate) { continue }
                    $price = Get-Number $candidate 'dph_total'
                    $reliability = Get-Number $candidate 'reliability2'
                    if ($null -eq $reliability) { $reliability = Get-Number $candidate 'reliability' }
                    $bandwidth = Get-Number $candidate 'disk_bw'
                    $fraction = Get-Number $candidate 'gpu_frac'
                    $gpuCount = Get-Number $candidate 'num_gpus'
                    if ($null -ne $price -and $price -ge 0 -and $price -le $MaxHourlyPrice -and
                        $null -ne $reliability -and $reliability -ge $MinReliability -and
                        $null -ne $bandwidth -and $bandwidth -ge $MinDiskBandwidth -and
                        $gpuCount -eq $gpuCountRequired -and $candidate.gpu_name -match $gpuPattern -and
                        (-not $WholeHostOnly -or ($null -ne $fraction -and $fraction -ge 1)) -and
                        ([string]$candidate.id -match '^[1-9][0-9]*$')) { $candidate }
                }
            )
            Write-Status '2/6 Offers' "Attempt $attempt/$MaxOfferSearchTries`: $($offers.Count) returned, $($eligible.Count) eligible."
            if ($eligible.Count -gt 0) { $chosen = $eligible | Sort-Object { [double]$_.dph_total } | Select-Object -First 1; break }
            if ($attempt -eq 1 -and $offers.Count -gt 0) {
                Write-Status '2/6 Offers' 'Returned offers failed local checks. First five returned offers:'
                $offers | Select-Object -First 5 id,gpu_name,num_gpus,dph_total,reliability2,reliability,disk_bw,gpu_frac |
                    Format-Table -AutoSize | Out-Host
            }
            if ($attempt -lt $MaxOfferSearchTries) {
                $waitMs = $SearchIntervalSeconds*1000
                if ($SearchTimeoutSeconds -gt 0) {
                    $remainingMs = $SearchTimeoutSeconds*1000 - $searchClock.ElapsedMilliseconds
                    if ($remainingMs -le 0) { break }
                    $waitMs = [math]::Min($waitMs, $remainingMs)
                }
                Write-Status '2/6 Offers' "No match; waiting $([math]::Round($waitMs/1000,1)) seconds."
                Start-Sleep -Milliseconds $waitMs
            }
        }
        if ($null -eq $chosen) { throw 'No qualifying offer within the configured search budget. No rental created.' }
        Write-Status '3/6 Create' "Selected offer $($chosen.id), advertised USD $($chosen.dph_total)/hr. Creating ONE instance."
        $chosen | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath (Join-Path $sessionDir 'offer.json') -Encoding UTF8
        $session = [ordered]@{ started_utc=[DateTime]::UtcNow.ToString('o'); status='creation_pending';
            offer_id=$chosen.id; advertised_hourly_rate=$chosen.dph_total; max_hourly_price=$MaxHourlyPrice;
            whole_host_only=[bool]$WholeHostOnly; gpu_count=$gpuCountRequired; gpu_name=$chosen.gpu_name;
            image=$image; instance_id=$null; manual_teardown=$true }
        $session | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $sessionFile -Encoding UTF8
        if ($SearchTimeoutSeconds -gt 0 -and $searchClock.Elapsed.TotalSeconds -ge $SearchTimeoutSeconds) {
            $session.status = 'search_timeout'
            $session | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $sessionFile -Encoding UTF8
            throw 'The search window expired before creation. No rental created.'
        }
        # Never automatically retry creation: a lost response can still mean a paid instance exists.
        $creationAttempted = $true
        $new = Invoke-VastJson -Arguments @('create','instance',[string]$chosen.id,'--image',$image,'--disk',$diskSize,'--ssh','--direct')
        if ([string]$new.new_contract -notmatch '^[1-9][0-9]*$') { throw 'Missing contract ID; creation status is uncertain.' }
        $iid = [string]$new.new_contract
        $session.instance_id = $iid
        $session.status = 'created'
        $session | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $sessionFile -Encoding UTF8
        if ($new.success -ne $true) {
            Write-Status '3/6 Create WARNING' "Creation returned contract ID $iid with a missing or unexpected success field. Continuing by polling this instance; creation will not be retried."
        }
        Write-Status '3/6 Create' "Instance $iid created. It may be billing now. Saved rental details: $sessionFile"
        Write-Status '3/6 Create' "Manual teardown: vastai destroy instance $iid"
    }

    $status = ''
    for ($attempt=1; $attempt -le 180; $attempt++) {
        try { $instance = Invoke-VastJson -Arguments @('show','instance',$iid) }
        catch {
            $status = ''
            Write-Status '4/6 Startup' 'Instance status unavailable. Checking SSH readiness directly.'
            break
        }
        if ($null -ne $instance.instances) {
            $instance = $instance.instances | Where-Object { [string]$_.id -eq $iid } | Select-Object -First 1
        }
        $status = [string]$instance.actual_status
        if ([string]::IsNullOrWhiteSpace($status)) {
            Write-Status '4/6 Startup' 'No actual_status reported. Checking SSH readiness directly.'
            break
        }
        Write-Status '4/6 Startup' "Instance $iid`: status=$status ($attempt/180)."
        if ($status -eq 'running') { break }
        if ($status -in @('error','failed','exited','destroyed','offline')) { throw "Instance entered $status; inspect console status details." }
        if ($attempt -lt 180) { Start-Sleep -Seconds 5 }
    }
    if ($status -and $status -ne 'running') { throw 'Instance did not reach running within the startup polling budget.' }
    $ready=$false
    for ($attempt=1; $attempt -le 36; $attempt++) {
        Write-Status '5/6 SSH' "Checking SSH access for instance $iid ($attempt/36)."
        $urlReply = Invoke-External -Command vastai -Arguments @('ssh-url',$iid) -AllowFailure
        if ($urlReply.ExitCode -eq 0 -and $urlReply.Stdout.Trim() -match '^ssh://([a-zA-Z0-9_][a-zA-Z0-9_.-]*)@([a-zA-Z0-9][a-zA-Z0-9.-]*):([0-9]{1,5})/?$' -and
            [int]$matches[3] -ge 1 -and [int]$matches[3] -le 65535) {
            $sshUser=$matches[1]; $sshHost=$matches[2]; $sshPort=$matches[3]
            $probe = Invoke-External -Command ssh -Arguments ($opt+@('-p',$sshPort,"$sshUser@$sshHost",'true')) -AllowFailure
            if ($probe.ExitCode -eq 0) { $ready=$true; break }
            Write-Status '5/6 SSH' "Not ready: $($probe.Stderr.Trim())"
        } else {
            Write-Status '5/6 SSH' 'Vast has not returned a usable SSH endpoint yet.'
        }
        if ($attempt -lt 36) { Start-Sleep -Seconds 5 }
    }
    if (-not $ready) { throw 'SSH access failed after 36 attempts. Check key/agent and console.' }
    $session.ssh_user=$sshUser; $session.ssh_host=$sshHost; $session.ssh_port=$sshPort
    if ($TwoGpuPilot -or $EightGpuEnsemble) {
        $session.status='ssh_ready'; $session.completed_utc=[DateTime]::UtcNow.ToString('o')
        $session | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $sessionFile -Encoding UTF8
        Write-Status '6/6 Ready' "Instance $iid is reachable. The experiment runner will upload the code and check the GPUs."
        $quotedRunner = $pilotRunner.Replace("'","''")
        $quotedKey = $PrivateKeyPath.Replace("'","''")
        Write-Host "& '$quotedRunner' -InstanceId $iid -PrivateKeyPath '$quotedKey'"
        Write-Host "Session record: $sessionFile"
        Write-Host "Manage teardown for instance $iid in the Vast console."
        return
    }
    $session | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $sessionFile -Encoding UTF8
    Write-Status '6/6 Upload' 'Uploading source package, analysis helper and Markdown plan to /workspace. This may take a moment.'
    $upload = Invoke-External -Command scp -Arguments ($opt+@('-r','-P',$sshPort,$pkg,$helper,$plan,"${sshUser}@${sshHost}:/workspace/"))
    if ($upload.Stdout) { Write-Host $upload.Stdout }
    if ($upload.Stderr) { Write-Host $upload.Stderr }
    $session.status='uploaded'; $session.completed_utc=[DateTime]::UtcNow.ToString('o')
    $session | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $sessionFile -Encoding UTF8
    Write-Status '6/6 Upload' 'Upload completed. No experiment has run.'
    $connectArguments = @($opt+@('-p',$sshPort,"$sshUser@$sshHost") | ForEach-Object { "'"+($_ -replace "'","''")+"'" })
    Write-Host "`nConnect from PowerShell:"
    Write-Host ('ssh '+($connectArguments -join ' '))
    Write-Host @'

On the host, stage ONE run directory:
set -euo pipefail
RUN="/workspace/fp-sanity-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$RUN"
mkdir "$RUN/code" "$RUN/data" "$RUN/logs" "$RUN/results"
cp -r /workspace/gpu_fingerprinting_hardened_v1/. "$RUN/code/"
cp /workspace/single_h200_sanity_analysis.py /workspace/single_h200_sanity_plan.md "$RUN/"
cd "$RUN"
cat single_h200_sanity_plan.md

Continue at section 3's "Record host metadata" commands.
Do NOT repeat section 3's initial RUN/mkdir commands: the files are already staged.
Then install dependencies, run CPU tests, build, and follow the smoke -> inspect -> main -> analysis gates.
'@
    Write-Host "`nDownload the complete run directory and verify hashes before teardown."
    Write-Host "Instance $iid remains running; manual teardown: vastai destroy instance $iid" -ForegroundColor Yellow
    Write-Host "Session record: $sessionFile"
} catch {
    Write-Status 'FAILED' $_.Exception.Message
    if ($iid) {
        Write-Host "Instance $iid has NOT been destroyed and may still be billing. Check the console. Manual teardown: vastai destroy instance $iid" -ForegroundColor Yellow
    } elseif ($creationAttempted) {
        Write-Host 'Creation status is UNCERTAIN. Check the Vast console before retrying; an instance may exist and be billing.' -ForegroundColor Yellow
    } else {
        Write-Host 'Stopped before instance creation.' -ForegroundColor Yellow
    }
    if ($sessionFile) { Write-Host "Session record: $sessionFile" }
    throw
}
