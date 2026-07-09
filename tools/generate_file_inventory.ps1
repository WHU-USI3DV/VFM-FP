param(
    [string]$Root = "release\VFM-FP-essential",
    [string]$Output = "FILE_INVENTORY.md"
)

$ErrorActionPreference = "Stop"
$rootPath = (Resolve-Path -LiteralPath $Root).Path
$outputPath = if ([System.IO.Path]::IsPathRooted($Output)) {
    $Output
} else {
    Join-Path $rootPath $Output
}

function Get-FilePurpose {
    param([string]$RelativePath)

    $p = $RelativePath -replace "\\", "/"

    if ($p -eq "README.md") { return @("Core", "Project overview and entrypoint documentation.", "Keep") }
    if ($p -eq ".gitignore") { return @("Repo hygiene", "Git ignore rules for data, weights, logs, caches, and release folders.", "Keep") }
    if ($p -eq ".gitattributes") { return @("Repo hygiene", "Line-ending and text normalization rules.", "Keep") }
    if ($p -eq "RELEASE_COPY_NOTES.md") { return @("Release note", "Generated note describing how the clean release copy was created.", "Keep") }
    if ($p -eq "LEAN_RELEASE_NOTES.md") { return @("Release note", "Generated note describing how the lean release candidate was created.", "Keep") }

    if ($p -like "configs/facade_classes.json") { return @("Config", "Facade class names, ids, and colors used by documentation and release metadata.", "Keep") }
    if ($p -like "configs/paths.example.json") { return @("Config", "Example path conventions and accepted default locations.", "Keep") }

    if ($p -like "requirements/*") { return @("Environment", "Lightweight dependency list for segmentation or SDA workflows.", "Keep") }
    if ($p -like "docs/*") { return @("Documentation", "Release, data, licensing, citation, cleanup, or file-role documentation.", "Keep") }
    if ($p -like "tools/audit_release.*") { return @("Release tool", "Checks the release folder for data, weights, media, archives, and large files.", "Keep") }
    if ($p -like "tools/content_audit.ps1") { return @("Release tool", "Checks the release folder for local paths, placeholder markers, and sensitive text.", "Keep") }
    if ($p -like "tools/prepare_release.ps1") { return @("Release tool", "Creates the source-only clean release folder.", "Keep") }
    if ($p -like "tools/prepare_lean_release.ps1") { return @("Release tool", "Creates a proposed lean release from Keep and Review inventory items.", "Keep") }
    if ($p -like "tools/smoke_check.ps1") { return @("Release tool", "Validates required files and cache-free release hygiene.", "Keep") }
    if ($p -like "tools/generate_file_inventory.ps1") { return @("Release tool", "Generates this file-by-file inventory.", "Keep") }
    if ($p -like "tools/syntax_check_release.py") { return @("Release tool", "Syntax-checks release Python files without importing heavy dependencies.", "Keep") }

    if ($p -like "VCFS/LICENSE") { return @("License", "Upstream MIT license notice for the inherited DeepLabv3+ baseline code.", "Keep") }
    if ($p -like "VCFS/README_VFMFP.md") { return @("Documentation", "VFM-FP-specific notes for the segmentation module.", "Keep") }
    if ($p -like "VCFS/README.md") { return @("Documentation", "Upstream DeepLabv3+ README kept as baseline reference.", "Review") }
    if ($p -like "VCFS/requirements.txt") { return @("Environment", "Original lightweight requirements file for the segmentation baseline.", "Review") }
    if ($p -like "VCFS/environment.yml") { return @("Environment", "Original Conda environment export kept as an experiment reference.", "Optional") }
    if ($p -like "VCFS/.gitignore") { return @("Repo hygiene", "Original submodule ignore rules for datasets, logs, caches, and build artifacts.", "Review") }

    if ($p -like "VCFS/train.py") { return @("Segmentation core", "Training entrypoint for the DeepLabv3+ facade parser.", "Keep") }
    if ($p -like "VCFS/predict.py") { return @("Segmentation core", "Inference entrypoint for image, folder, video, FPS, and export workflows.", "Keep") }
    if ($p -like "VCFS/get_miou.py") { return @("Segmentation core", "mIoU evaluation entrypoint.", "Keep") }
    if ($p -like "VCFS/voc_annotation.py") { return @("Segmentation utility", "Generates VOC-style split files.", "Keep") }
    if ($p -like "VCFS/json_to_dataset.py") { return @("Segmentation utility", "Converts LabelMe-style annotations to segmentation masks.", "Keep") }
    if ($p -like "VCFS/deeplab.py") { return @("Segmentation core", "DeepLabv3+ inference class and preprocessing/postprocessing logic.", "Keep") }
    if ($p -like "VCFS/benchmark.py") { return @("Segmentation utility", "Benchmark/FPS utility for inference speed checks.", "Review") }
    if ($p -like "VCFS/benchimark.py") { return @("Compatibility", "Backward-compatible wrapper for benchmark.py typo.", "Optional") }

    if ($p -like "VCFS/nets/*") { return @("Segmentation model", "Network architecture, backbone, attention, or training-loss module.", "Keep") }
    if ($p -like "VCFS/utils/utils_fit.py") { return @("Segmentation core", "Training and validation loop helper functions.", "Keep") }
    if ($p -like "VCFS/utils/utils_metrics.py") { return @("Segmentation core", "Segmentation metrics and mIoU helpers.", "Keep") }
    if ($p -like "VCFS/utils/callbacks.py") { return @("Segmentation core", "Training callbacks, logging, checkpoints, and evaluation callbacks.", "Keep") }
    if ($p -like "VCFS/utils/utils.py") { return @("Segmentation core", "Shared preprocessing, resizing, configuration, and utility functions.", "Keep") }
    if ($p -like "VCFS/utils/dataloader_latest.py") { return @("Segmentation core", "Current training dataloader used by train.py.", "Keep") }
    if ($p -like "VCFS/utils/dataloader_latest_ecp.py") { return @("Segmentation data", "ECP-specific dataloader variant.", "Review") }
    if ($p -like "VCFS/utils/fea_upscale.py") { return @("Segmentation model", "Feature upscaling helper imported by some network variants.", "Keep") }
    if ($p -like "VCFS/utils/__init__.py") { return @("Package marker", "Marks the segmentation utils directory as a Python package.", "Keep") }
    if ($p -like "VCFS/utils/dataloader.py") { return @("Compatibility", "Wrapper exposing the legacy baseline dataloader under its original import path.", "Optional") }

    if ($p -like "VCFS/legacy/*" -or $p -like "VCFS/utils/legacy/*") { return @("Legacy", "Historical or one-off helper preserved for compatibility and reproducibility.", "Optional") }
    if ($p -like "VCFS/*_ori.py" -or $p -like "VCFS/utils/*_ori.py" -or $p -like "VCFS/delete.py" -or $p -like "VCFS/summary.py" -or $p -like "VCFS/turn.py" -or $p -like "VCFS/unsort.py" -or $p -like "VCFS/utils/delete.py" -or $p -like "VCFS/utils/downto.py" -or $p -like "VCFS/utils/Rename.py" -or $p -like "VCFS/utils/turn1.py" -or $p -like "VCFS/utils/dataloader_lastest*") { return @("Compatibility", "Wrapper preserving an old script name or import path.", "Optional") }

    if ($p -like "SDA/README.md") { return @("Documentation", "Overview of SDA data expansion and DINO ranking scripts.", "Keep") }
    if ($p -like "SDA/diffusion/Mul_Ab_norway.py") { return @("SDA core", "Configurable ControlNet generation script with accepted-paper defaults.", "Keep") }
    if ($p -like "SDA/diffusion/ImageNet_color.py") { return @("SDA core", "Semantic color palette used by diffusion segmentation conditioning.", "Keep") }
    if ($p -like "SDA/diffusion/voc_annotation.py") { return @("SDA utility", "VOC-style split generation helper for SDA outputs.", "Review") }
    if ($p -like "SDA/diffusion/legacy/*") { return @("SDA legacy", "Historical diffusion or maintenance helper preserved for reproducibility.", "Optional") }
    if ($p -like "SDA/diffusion/*.py") { return @("Compatibility", "Wrapper preserving an old SDA diffusion helper name.", "Optional") }

    if ($p -like "SDA/DINO_extract/dino_rank_generated.py") { return @("SDA core", "Configurable DINOv2 feature-distance ranking CLI.", "Keep") }
    if ($p -like "SDA/DINO_extract/Extract_Cul*.py") { return @("SDA core", "Compatibility entrypoint for a DINO ranking preset.", "Keep") }
    if ($p -like "SDA/DINO_extract/legacy/*") { return @("SDA legacy", "Historical DINO, ranking, visualization, or maintenance helper.", "Optional") }
    if ($p -like "SDA/DINO_extract/*.py") { return @("Compatibility", "Wrapper preserving an old SDA DINO helper or visualization name.", "Optional") }

    if ($p -like "*.txt") { return @("Text data", "Small text file preserved from the experiment tree; review whether it is needed.", "Review") }
    if ($p -like "*.json") { return @("Config", "JSON configuration or metadata file.", "Review") }
    return @("Other", "Unclassified lightweight file; review manually.", "Review")
}

$resolvedOutput = $null
if (Test-Path -LiteralPath $outputPath) {
    $resolvedOutput = (Resolve-Path -LiteralPath $outputPath).Path
}

$files = Get-ChildItem -LiteralPath $rootPath -Recurse -File -Force |
    Where-Object { $null -eq $resolvedOutput -or $_.FullName -ne $resolvedOutput } |
    Sort-Object FullName

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# File Inventory")
$lines.Add("")
$lines.Add("Generated for: $([System.IO.Path]::GetFileName($rootPath))")
$lines.Add("Generated at: $(Get-Date -Format s)")
$lines.Add("")
$lines.Add("Recommendation meanings:")
$lines.Add("")
$lines.Add("- Keep: necessary for the public source release.")
$lines.Add("- Review: useful but worth checking before final publication.")
$lines.Add("- Optional: compatibility, legacy, or historical helper; remove only after confirming it is not needed.")
$lines.Add("")
$lines.Add("| File | Category | Purpose | Recommendation |")
$lines.Add("| --- | --- | --- | --- |")

foreach ($file in $files) {
    $relative = $file.FullName.Substring($rootPath.Length).TrimStart([char]92, [char]47) -replace "\\", "/"
    $info = Get-FilePurpose $relative
    $lines.Add("| $relative | $($info[0]) | $($info[1]) | $($info[2]) |")
}

$encoding = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines($outputPath, [string[]]$lines, $encoding)
Write-Output "Inventory written: $outputPath"
Write-Output "Files listed: $($files.Count)"