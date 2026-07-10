param(
    [string]$Root = "."
)

$ErrorActionPreference = "Stop"
$rootPath = (Resolve-Path -LiteralPath $Root).Path
$failures = New-Object System.Collections.Generic.List[string]

function Add-Failure {
    param([string]$Message)
    $failures.Add($Message)
}

function Test-FileExists {
    param([string]$RelativePath)
    $path = Join-Path $rootPath $RelativePath
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        Add-Failure "Missing file: $RelativePath"
    }
}

function Test-JsonFile {
    param([string]$RelativePath)
    $path = Join-Path $rootPath $RelativePath
    try {
        Get-Content -Raw -LiteralPath $path | ConvertFrom-Json | Out-Null
    }
    catch {
        Add-Failure "Invalid JSON: $RelativePath"
    }
}

$requiredFiles = @(
    "README.md",
    ".gitignore",
    ".gitattributes",
    "configs/facade_classes.json",
    "configs/classes.facadewhu.json",
    "configs/classes.ecp.json",
    "configs/paths.example.json",
    "requirements/segmentation.txt",
    "requirements/sda.txt",
    "docs/citation_template.md",
    "docs/data_and_weights.md",
    "docs/licensing.md",
    "SDA/README.md",
    "SDA/prepare_vcfs_augmented_dataset.py",
    "SDA/DINO_extract/dino_rank_generated.py",
    "SDA/DINO_extract/semantic_consistency_filter.py",
    "SDA/diffusion/semantic_diffusion_augmentation.py",
    "SDA/diffusion/Mul_Ab_norway.py",
    "VCFS/README.md",
    "VCFS/README_VFMFP.md",
    "VCFS/train.py",
    "VCFS/predict.py",
    "VCFS/get_miou.py",
    "VCFS/deeplab.py",
    "VCFS/nets/deeplabv3_plus.py",
    "VCFS/utils/dataloader_latest.py",
    "VCFS/utils/split_utils.py",
    "VCFS/utils/class_config.py",
    "VCFS/utils/fea_upscale.py",
    "VCFS/voc_annotation.py"
)

foreach ($file in $requiredFiles) {
    Test-FileExists $file
}

Test-JsonFile "configs/facade_classes.json"
Test-JsonFile "configs/classes.facadewhu.json"
Test-JsonFile "configs/classes.ecp.json"
Test-JsonFile "configs/paths.example.json"

$forbiddenDirectoryNames = @(
    "__pycache__",
    ".ipynb_checkpoints",
    "logs",
    "debug",
    "VOCdevkit",
    "FacadeWHU_origin",
    "facadewhu",
    "facadewhu_extend",
    "wuhan_test",
    "4_nyl_zip"
)

$gitPath = Join-Path $rootPath ".git"

$forbiddenFileExtensions = @(
    ".pyc",
    ".pyo",
    ".pth",
    ".pt",
    ".ckpt",
    ".onnx",
    ".bin",
    ".safetensors",
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".npy",
    ".npz"
)

Get-ChildItem -LiteralPath $rootPath -Recurse -Directory -Force | ForEach-Object {
    if ($_.FullName.StartsWith($gitPath, [System.StringComparison]::OrdinalIgnoreCase)) { return }
    if ($forbiddenDirectoryNames -contains $_.Name) {
        Add-Failure "Found forbidden directory: $($_.FullName)"
    }
}

Get-ChildItem -LiteralPath $rootPath -Recurse -File -Force | ForEach-Object {
    if ($_.FullName.StartsWith($gitPath, [System.StringComparison]::OrdinalIgnoreCase)) { return }
    if ($forbiddenFileExtensions -contains $_.Extension.ToLowerInvariant()) {
        Add-Failure "Found forbidden file: $($_.FullName)"
    }
}

if ($failures.Count -gt 0) {
    Write-Output "Smoke check failed:"
    foreach ($failure in $failures) {
        Write-Output "- $failure"
    }
    exit 1
}

Write-Output "Smoke check passed."
exit 0
