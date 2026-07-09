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

function Test-AsciiFile {
    param([string]$RelativePath)
    $path = Join-Path $rootPath $RelativePath
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        Add-Failure "Missing ASCII file: $RelativePath"
        return
    }

    $text = Get-Content -Raw -LiteralPath $path
    for ($i = 0; $i -lt $text.Length; $i++) {
        if ([int][char]$text[$i] -gt 127) {
            Add-Failure "Non-ASCII character in: $RelativePath"
            return
        }
    }
}

$requiredFiles = @(
    "README.md",
    ".gitignore",
    ".gitattributes",
    "configs/facade_classes.json",
    "configs/paths.example.json",
    "requirements/segmentation.txt",
    "requirements/sda.txt",
    "docs/open_source_cleanup.md",
    "docs/data_and_weights.md",
    "docs/script_inventory.md",
    "docs/release_checklist.md",
    "docs/release_status.md",
    "docs/citation_template.md",
    "docs/file_review_guide.md",
    "docs/owner_decisions.md",
    "docs/licensing.md",
    "SDA/README.md",
    "VCFS/README_VFMFP.md",
    "VCFS/benchmark.py",
    "VCFS/benchimark.py",
    "VCFS/summary.py",
    "VCFS/turn.py",
    "VCFS/delete.py",
    "VCFS/unsort.py",
    "VCFS/legacy/__init__.py",
    "VCFS/legacy/model_summary.py",
    "VCFS/legacy/remap_segmentation_colors.py",
    "VCFS/legacy/filter_split_file.py",
    "VCFS/legacy/shuffle_split_file.py",
    "VCFS/utils/dataloader_latest.py",
    "VCFS/utils/dataloader_lastest.py",
    "VCFS/utils/dataloader_latest_ecp.py",
    "VCFS/utils/dataloader_lastest_ecp.py",
    "VCFS/utils/delete.py",
    "VCFS/utils/downto.py",
    "VCFS/utils/Rename.py",
    "VCFS/utils/turn1.py",
    "VCFS/utils/legacy/__init__.py",
    "VCFS/utils/legacy/prune_image_triplets.py",
    "VCFS/utils/legacy/random_downsample_folder.py",
    "VCFS/utils/legacy/add_syn_prefix.py",
    "VCFS/utils/legacy/remap_segmentation_colors_ecp.py",
    "SDA/diffusion/check_quality.py",
    "SDA/diffusion/delete.py",
    "SDA/diffusion/downto.py",
    "SDA/diffusion/Rename.py",
    "SDA/diffusion/small.py",
    "SDA/diffusion/small2.py",
    "SDA/diffusion/sort.py",
    "SDA/diffusion/unsort.py",
    "SDA/diffusion/legacy/__init__.py",
    "SDA/diffusion/legacy/filter_zero_quality_scores.py",
    "SDA/diffusion/legacy/keep_first_generated_per_index.py",
    "SDA/diffusion/legacy/random_downsample_folder.py",
    "SDA/diffusion/legacy/copy_trainval_images_with_syn_prefix.py",
    "SDA/diffusion/legacy/count_facade_detail_pixels.py",
    "SDA/diffusion/legacy/count_facade_detail_pixels_with_header.py",
    "SDA/diffusion/legacy/sort_score_file_desc.py",
    "SDA/diffusion/legacy/shuffle_score_file.py",
    "SDA/DINO_extract/Check_quality.py",
    "SDA/DINO_extract/chouqu.py",
    "SDA/DINO_extract/Com_Cmean.py",
    "SDA/DINO_extract/delete.py",
    "SDA/DINO_extract/Extract_Cul2.py",
    "SDA/DINO_extract/random_get.py",
    "SDA/DINO_extract/Rename.py",
    "SDA/DINO_extract/small.py",
    "SDA/DINO_extract/SORT_ascend.py",
    "SDA/DINO_extract/legacy/__init__.py",
    "SDA/DINO_extract/legacy/make_quality_comparison_images.py",
    "SDA/DINO_extract/legacy/copy_ranked_generated_images.py",
    "SDA/DINO_extract/legacy/average_scores_by_source_id.py",
    "SDA/DINO_extract/legacy/keep_first_syn_image_per_index.py",
    "SDA/DINO_extract/legacy/extract_cul_norway_variant.py",
    "SDA/DINO_extract/legacy/random_downsample_folder.py",
    "SDA/DINO_extract/legacy/add_syn_prefix_to_images.py",
    "SDA/DINO_extract/legacy/count_roof_pixels.py",
    "SDA/DINO_extract/legacy/sort_score_file_asc.py",
    "SDA/DINO_extract/dino_rank_generated.py",
    "SDA/DINO_extract/Extract_Cul.py",
    "SDA/DINO_extract/Extract_Cul_facadewhu.py",
    "SDA/DINO_extract/Extract_Cul_ecp.py",
    "SDA/DINO_extract/legacy/extract_cul_default_original.py",
    "SDA/DINO_extract/legacy/extract_cul_facadewhu_original.py",
    "SDA/DINO_extract/legacy/extract_cul_ecp_original.py",
    "SDA/diffusion/Mul_Ab_norway.py",
    "SDA/diffusion/legacy/mul_ab_norway_original.py",
    "SDA/diffusion/Mul_Ab_2.py",
    "SDA/diffusion/Mul_Ab_2_diversity.py",
    "SDA/diffusion/legacy/mul_ab_2_original.py",
    "SDA/diffusion/legacy/mul_ab_2_diversity_original.py",
    "SDA/DINO_extract/deeplabv3+.py",
    "SDA/DINO_extract/dino_extract.py",
    "SDA/DINO_extract/get_feature.py",
    "SDA/DINO_extract/legacy/deeplabv3_plus_visualization.py",
    "SDA/DINO_extract/legacy/dino_pca_single_image.py",
    "SDA/DINO_extract/legacy/deeplab_encoder_feature_visualization.py",
    "VCFS/deeplab_ori.py",
    "VCFS/get_miou_ori.py",
    "VCFS/legacy/deeplab_original.py",
    "VCFS/legacy/get_miou_original.py",
    "VCFS/utils/dataloader.py",
    "VCFS/utils/dataloader_ori.py",
    "VCFS/utils/callbacks_ori.py",
    "VCFS/utils/utils_ori.py",
    "VCFS/utils/legacy/dataloader_baseline.py",
    "VCFS/utils/legacy/dataloader_original.py",
    "VCFS/utils/legacy/callbacks_original.py",
    "VCFS/utils/legacy/utils_original.py",
    "tools/audit_release.py",
    "tools/audit_release.ps1",
    "tools/prepare_release.ps1",
    "tools/prepare_lean_release.ps1",
    "tools/content_audit.ps1",
    "tools/generate_file_inventory.ps1",
    "tools/syntax_check_release.py"
)

foreach ($file in $requiredFiles) {
    Test-FileExists $file
}

Test-JsonFile "configs/facade_classes.json"
Test-JsonFile "configs/paths.example.json"

$asciiFiles = @(
    "README.md",
    ".gitignore",
    ".gitattributes",
    "configs/facade_classes.json",
    "configs/paths.example.json",
    "requirements/segmentation.txt",
    "requirements/sda.txt",
    "docs/open_source_cleanup.md",
    "docs/data_and_weights.md",
    "docs/script_inventory.md",
    "docs/release_status.md",
    "docs/release_checklist.md",
    "docs/citation_template.md",
    "docs/file_review_guide.md",
    "docs/owner_decisions.md",
    "docs/licensing.md",
    "SDA/README.md",
    "VCFS/README_VFMFP.md",
    "VCFS/benchimark.py",
    "VCFS/summary.py",
    "VCFS/turn.py",
    "VCFS/delete.py",
    "VCFS/unsort.py",
    "VCFS/legacy/__init__.py",
    "VCFS/legacy/model_summary.py",
    "VCFS/legacy/remap_segmentation_colors.py",
    "VCFS/legacy/filter_split_file.py",
    "VCFS/legacy/shuffle_split_file.py",
    "VCFS/utils/dataloader_lastest.py",
    "VCFS/utils/dataloader_lastest_ecp.py",
    "VCFS/utils/delete.py",
    "VCFS/utils/downto.py",
    "VCFS/utils/Rename.py",
    "VCFS/utils/turn1.py",
    "VCFS/utils/legacy/__init__.py",
    "VCFS/utils/legacy/prune_image_triplets.py",
    "VCFS/utils/legacy/random_downsample_folder.py",
    "VCFS/utils/legacy/add_syn_prefix.py",
    "VCFS/utils/legacy/remap_segmentation_colors_ecp.py",
    "SDA/diffusion/check_quality.py",
    "SDA/diffusion/delete.py",
    "SDA/diffusion/downto.py",
    "SDA/diffusion/Rename.py",
    "SDA/diffusion/small.py",
    "SDA/diffusion/small2.py",
    "SDA/diffusion/sort.py",
    "SDA/diffusion/unsort.py",
    "SDA/diffusion/legacy/__init__.py",
    "SDA/diffusion/legacy/filter_zero_quality_scores.py",
    "SDA/diffusion/legacy/keep_first_generated_per_index.py",
    "SDA/diffusion/legacy/random_downsample_folder.py",
    "SDA/diffusion/legacy/copy_trainval_images_with_syn_prefix.py",
    "SDA/diffusion/legacy/count_facade_detail_pixels.py",
    "SDA/diffusion/legacy/count_facade_detail_pixels_with_header.py",
    "SDA/diffusion/legacy/sort_score_file_desc.py",
    "SDA/diffusion/legacy/shuffle_score_file.py",
    "SDA/DINO_extract/Check_quality.py",
    "SDA/DINO_extract/Com_Cmean.py",
    "SDA/DINO_extract/delete.py",
    "SDA/DINO_extract/random_get.py",
    "SDA/DINO_extract/Rename.py",
    "SDA/DINO_extract/small.py",
    "SDA/DINO_extract/SORT_ascend.py",
    "SDA/DINO_extract/legacy/__init__.py",
    "SDA/DINO_extract/legacy/make_quality_comparison_images.py",
    "SDA/DINO_extract/legacy/average_scores_by_source_id.py",
    "SDA/DINO_extract/legacy/keep_first_syn_image_per_index.py",
    "SDA/DINO_extract/legacy/random_downsample_folder.py",
    "SDA/DINO_extract/legacy/add_syn_prefix_to_images.py",
    "SDA/DINO_extract/legacy/count_roof_pixels.py",
    "SDA/DINO_extract/legacy/sort_score_file_asc.py",
    "SDA/DINO_extract/dino_rank_generated.py",
    "SDA/DINO_extract/Extract_Cul.py",
    "SDA/DINO_extract/Extract_Cul_facadewhu.py",
    "SDA/DINO_extract/Extract_Cul_ecp.py",
    "SDA/diffusion/Mul_Ab_norway.py",
    "SDA/diffusion/Mul_Ab_2.py",
    "SDA/diffusion/Mul_Ab_2_diversity.py",
    "SDA/DINO_extract/deeplabv3+.py",
    "SDA/DINO_extract/dino_extract.py",
    "SDA/DINO_extract/get_feature.py",
    "VCFS/deeplab_ori.py",
    "VCFS/get_miou_ori.py",
    "VCFS/utils/dataloader.py",
    "VCFS/utils/dataloader_ori.py",
    "VCFS/utils/callbacks_ori.py",
    "VCFS/utils/utils_ori.py",
    "tools/audit_release.py",
    "tools/audit_release.ps1",
    "tools/content_audit.ps1",
    "tools/prepare_release.ps1"
)

foreach ($file in $asciiFiles) {
    Test-AsciiFile $file
}

$cacheDirs = Get-ChildItem -LiteralPath $rootPath -Recurse -Directory -Force -Filter "__pycache__"
foreach ($dir in $cacheDirs) {
    Add-Failure "Found __pycache__: $($dir.FullName)"
}

$checkpointDirs = Get-ChildItem -LiteralPath $rootPath -Recurse -Directory -Force -Filter ".ipynb_checkpoints"
foreach ($dir in $checkpointDirs) {
    Add-Failure "Found .ipynb_checkpoints: $($dir.FullName)"
}

$pycFiles = Get-ChildItem -LiteralPath $rootPath -Recurse -File -Force | Where-Object { $_.Extension -eq ".pyc" }
foreach ($file in $pycFiles) {
    Add-Failure "Found pyc file: $($file.FullName)"
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



