$ErrorActionPreference = "Stop"

$envPath = Join-Path (Split-Path -Parent $PSScriptRoot) ".env"
if (-not (Test-Path -LiteralPath $envPath -PathType Leaf)) {
    throw "Environment file is missing."
}

$obsoleteNames = [System.Collections.Generic.HashSet[string]]::new(
    [System.StringComparer]::Ordinal
)
@(
    "BENCH_CHAIN_RUNS",
    "BENCH_DONOR_ID",
    "BENCH_RUNS",
    "DEPLOYMENT_TX_HASH",
    "FOUNDRY_ARTIFACT_PATH",
    "LOGS_FROM_BLOCK",
    "LOGS_TO_BLOCK",
    "MATCH_RATIONALE_CID",
    "OPENAI_BASE_MODEL",
    "SEED_DONOR_CID",
    "SEED_RECIPIENT1_CID",
    "SEED_RECIPIENT2_CID",
    "SEED_RECIPIENT3_CID",
    "SEED_RECIPIENT4_CID",
    "SEED_RECIPIENT5_CID",
    "SEED_RECIPIENT6_CID",
    "SEED_RECIPIENT7_CID",
    "SEED_RECIPIENT8_CID",
    "SEED_RECIPIENT9_CID",
    "SEED_RECIPIENT10_CID",
    "TX_MAX_FEE_GWEI"
) | ForEach-Object { [void]$obsoleteNames.Add($_) }

$updatedLines = [System.Collections.Generic.List[string]]::new()
$modelIdWritten = $false
$removedCount = 0

foreach ($line in [System.IO.File]::ReadAllLines($envPath)) {
    if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=') {
        $name = $Matches[1]
        if ($obsoleteNames.Contains($name)) {
            $removedCount += 1
            continue
        }
        if ($name -eq "OPENAI_MODEL_ID") {
            if (-not $modelIdWritten) {
                $updatedLines.Add("OPENAI_MODEL_ID=gpt-4o-mini-2024-07-18")
                $modelIdWritten = $true
            }
            continue
        }
    }
    $updatedLines.Add($line)
}

if (-not $modelIdWritten) {
    $updatedLines.Add("OPENAI_MODEL_ID=gpt-4o-mini-2024-07-18")
}

$temporaryPath = "$envPath.codex-tmp"
$encoding = [System.Text.UTF8Encoding]::new($false)
try {
    [System.IO.File]::WriteAllLines($temporaryPath, $updatedLines, $encoding)
    Move-Item -LiteralPath $temporaryPath -Destination $envPath -Force
}
finally {
    if (Test-Path -LiteralPath $temporaryPath) {
        Remove-Item -LiteralPath $temporaryPath -Force
    }
}

Write-Output "Environment schema updated. Obsolete entries removed: $removedCount. Secret values were not printed."
