param(
    [string]$Source = "release\VFM-FP-open-source",
    [string]$Output = "release\VFM-FP-lean-proposed",
    [switch]$KeepOnly,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath ".").Path
$sourcePath = (Resolve-Path -LiteralPath $Source).Path
$inventoryPath = Join-Path $sourcePath "FILE_INVENTORY.md"

if (-not (Test-Path -LiteralPath $inventoryPath -PathType Leaf)) {
    $inventoryTool = Join-Path $repoRoot "tools\generate_file_inventory.ps1"
    if (-not (Test-Path -LiteralPath $inventoryTool -PathType Leaf)) {
        throw "Inventory not found and generator is missing: $inventoryTool"
    }
    & $inventoryTool -Root $sourcePath | Out-Null
}

if (-not (Test-Path -LiteralPath $inventoryPath -PathType Leaf)) {
    throw "Inventory not found after generation attempt: $inventoryPath"
}

$releaseRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "release"))
$outputPath = if ([System.IO.Path]::IsPathRooted($Output)) {
    [System.IO.Path]::GetFullPath($Output)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Output))
}

if ($outputPath -eq $repoRoot) {
    throw "Output path must not be the project root."
}
if (-not $outputPath.StartsWith($releaseRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Output must be inside the local release directory: $outputPath"
}
if ((Test-Path -LiteralPath $outputPath) -and -not $Force) {
    throw "Output already exists: $outputPath. Re-run with -Force to replace it."
}
if (Test-Path -LiteralPath $outputPath) {
    Remove-Item -LiteralPath $outputPath -Recurse -Force
}
New-Item -ItemType Directory -Path $outputPath | Out-Null

$allowedRecommendations = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
[void]$allowedRecommendations.Add("Keep")
if (-not $KeepOnly) {
    [void]$allowedRecommendations.Add("Review")
}

$selected = New-Object System.Collections.Generic.List[string]
foreach ($line in Get-Content -LiteralPath $inventoryPath) {
    if ($line -notmatch "^\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(Keep|Review|Optional)\s*\|\s*$") {
        continue
    }

    $relative = $Matches[1].Trim()
    $recommendation = $Matches[4].Trim()
    if ($allowedRecommendations.Contains($recommendation)) {
        $selected.Add($relative)
    }
}

$copied = 0
foreach ($relative in $selected) {
    $src = Join-Path $sourcePath ($relative -replace "/", "\")
    if (-not (Test-Path -LiteralPath $src -PathType Leaf)) {
        throw "Selected file not found in source release: $relative"
    }

    $dest = Join-Path $outputPath ($relative -replace "/", "\")
    $destDir = Split-Path -Parent $dest
    if (-not (Test-Path -LiteralPath $destDir)) {
        New-Item -ItemType Directory -Path $destDir | Out-Null
    }
    Copy-Item -LiteralPath $src -Destination $dest
    $copied += 1
}

$notes = @()
$notes += "# Lean Release Candidate"
$notes += ""
$notes += "Generated from: $Source"
$notes += "Generated at: $(Get-Date -Format s)"
$notes += "Mode: $(if ($KeepOnly) { 'Keep only' } else { 'Keep + Review' })"
$notes += "Copied files: $copied"
$notes += ""
$notes += "This folder is a proposed lean public release candidate."
$notes += "It excludes files marked Optional in FILE_INVENTORY.md, usually compatibility wrappers or historical helpers."
$notes += "Keep the broader release candidate until owner review confirms which Review files should remain."
$notes += ""
$notes += "After generation, run:"
$notes += ""
$notes += "    powershell -NoProfile -ExecutionPolicy Bypass -File tools/generate_file_inventory.ps1 -Root $Output"
$notes += "    powershell -NoProfile -ExecutionPolicy Bypass -File tools/audit_release.ps1 -Root $Output -Limit 100"
$notes += "    powershell -NoProfile -ExecutionPolicy Bypass -File tools/content_audit.ps1 -Root $Output -Limit 100"
$notes += "    python tools/syntax_check_release.py $Output"

Set-Content -LiteralPath (Join-Path $outputPath "LEAN_RELEASE_NOTES.md") -Value $notes -Encoding UTF8

Write-Output "Lean release created: $outputPath"
Write-Output "Mode: $(if ($KeepOnly) { 'Keep only' } else { 'Keep + Review' })"
Write-Output "Copied files: $copied"
