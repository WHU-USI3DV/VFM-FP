param(
    [string]$Root = ".",
    [string]$Output = "release\VFM-FP-open-source",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$rootPath = (Resolve-Path -LiteralPath $Root).Path
$outputPath = if ([System.IO.Path]::IsPathRooted($Output)) {
    $Output
} else {
    Join-Path $rootPath $Output
}
$outputFullPath = [System.IO.Path]::GetFullPath($outputPath)

$defaultReleaseRoot = [System.IO.Path]::GetFullPath((Join-Path $rootPath "release"))
if ($outputFullPath -eq $rootPath) {
    throw "Output path must not be the project root."
}

if ((Test-Path -LiteralPath $outputFullPath) -and -not $Force) {
    throw "Output already exists: $outputFullPath. Re-run with -Force to replace it."
}

if ((Test-Path -LiteralPath $outputFullPath) -and $Force) {
    if (-not $outputFullPath.StartsWith($defaultReleaseRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove output outside the local release directory: $outputFullPath"
    }
    Remove-Item -LiteralPath $outputFullPath -Recurse -Force
}

New-Item -ItemType Directory -Path $outputFullPath | Out-Null

$skipDirs = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
@(
    ".git",
    ".agents",
    ".codex",
    "release",
    "tmp",
    "__pycache__",
    ".ipynb_checkpoints",
    ".vscode",
    ".idea",
    ".venv",
    "venv",
    "env",
    "logs",
    "debug",
    "model_data",
    "VOCdevkit",
    "FacadeWHU_origin",
    "facadewhu",
    "facadewhu_extend",
    "wuhan_test",
    "JPEGImages",
    "SegmentationClass",
    "deeplab_features",
    "deeplab_encoder_vis",
    "DINO_feature",
    "pca_visualizations",
    "low_quality_pic",
    "check_quality",
    "Paper_save",
    "diversity",
    "need_facadewhu_ori_and_norway",
    "norway",
    "norway_facade",
    "norway_facadewhu",
    "ecp",
    "ECP",
    "ECP_0618",
    "ecp_0619_refine",
    "4_nyl_zip",
    "prepared",
    "dinov2_base"
) | ForEach-Object { [void]$skipDirs.Add($_) }

$blockedExtensions = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
@(
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
    ".mp4",
    ".avi",
    ".mov",
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".pth",
    ".pt",
    ".ckpt",
    ".onnx",
    ".bin",
    ".safetensors",
    ".npy",
    ".npz",
    ".pkl"
) | ForEach-Object { [void]$blockedExtensions.Add($_) }

$allowedExtensions = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
@(
    ".py",
    ".ps1",
    ".md",
    ".json",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".cfg",
    ".ini"
) | ForEach-Object { [void]$allowedExtensions.Add($_) }

$allowedNames = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
@(
    ".gitignore",
    ".gitattributes",
    "LICENSE",
    "LICENSE.txt"
) | ForEach-Object { [void]$allowedNames.Add($_) }

$blockedNames = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
@(
    "Seed_record.txt"
) | ForEach-Object { [void]$blockedNames.Add($_) }

function Test-AsciiName {
    param([string]$Name)
    for ($i = 0; $i -lt $Name.Length; $i++) {
        if ([int][char]$Name[$i] -gt 127) {
            return $false
        }
    }
    return $true
}
$copied = 0
$skippedFiles = 0
$skippedDirs = 0

$stack = New-Object 'System.Collections.Generic.Stack[System.IO.DirectoryInfo]'
$stack.Push([System.IO.DirectoryInfo]::new($rootPath))

while ($stack.Count -gt 0) {
    $dir = $stack.Pop()

    foreach ($childDir in $dir.EnumerateDirectories()) {
        if ($skipDirs.Contains($childDir.Name)) {
            $skippedDirs += 1
            continue
        }
        $stack.Push($childDir)
    }

    foreach ($file in $dir.EnumerateFiles()) {
        $relative = $file.FullName.Substring($rootPath.Length).TrimStart([char]92, [char]47)
        $name = $file.Name
        $ext = $file.Extension.ToLowerInvariant()

        if ((-not (Test-AsciiName $name)) -or $blockedNames.Contains($name) -or ($name -like "indexes_with_fewer_nums*.txt") -or $blockedExtensions.Contains($ext)) {
            $skippedFiles += 1
            continue
        }

        if ((-not $allowedExtensions.Contains($ext)) -and (-not $allowedNames.Contains($name))) {
            $skippedFiles += 1
            continue
        }

        $dest = Join-Path $outputFullPath $relative
        $destDir = Split-Path -Parent $dest
        if (-not (Test-Path -LiteralPath $destDir)) {
            New-Item -ItemType Directory -Path $destDir | Out-Null
        }
        Copy-Item -LiteralPath $file.FullName -Destination $dest
        $copied += 1
    }
}

$manifest = @()
$manifest += "# Local Release Copy"
$manifest += ""
$manifest += "Generated from project directory: $([System.IO.Path]::GetFileName($rootPath))"
$manifest += "Generated at: $(Get-Date -Format s)"
$manifest += "Copied files: $copied"
$manifest += "Skipped files: $skippedFiles"
$manifest += "Skipped directories: $skippedDirs"
$manifest += ""
$manifest += "This copy intentionally excludes datasets, generated images, model weights, archives, caches, and local run artifacts."
$manifest += "Run tools/audit_release.ps1 against this directory before publishing."
Set-Content -LiteralPath (Join-Path $outputFullPath "RELEASE_COPY_NOTES.md") -Value $manifest -Encoding UTF8

Write-Output "Release copy created: $outputFullPath"
Write-Output "Copied files: $copied"
Write-Output "Skipped files: $skippedFiles"
Write-Output "Skipped directories: $skippedDirs"


