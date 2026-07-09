param(
    [string]$Root = ".",
    [int]$LargeMB = 10,
    [int]$Limit = 200
)

$ErrorActionPreference = "Stop"

$rootPath = (Resolve-Path -LiteralPath $Root).Path
$largeBytes = $LargeMB * 1024 * 1024

$skipDirs = @(".git", ".venv", "venv", "env")
$artifactDirs = @(
    "__pycache__",
    ".ipynb_checkpoints",
    ".vscode",
    "logs",
    "debug",
    "model_data",
    "VOCdevkit",
    "FacadeWHU_origin",
    "facadewhu",
    "facadewhu_extend",
    "wuhan_test",
    "deeplab_features",
    "DINO_feature",
    "pca_visualizations",
    "low_quality_pic",
    "check_quality",
    "Paper_save"
)

$binaryExtensions = @(
    ".pth",
    ".pt",
    ".ckpt",
    ".bin",
    ".tar",
    ".zip",
    ".rar",
    ".7z",
    ".safetensors"
)

$mediaExtensions = @(
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)

$findings = New-Object System.Collections.Generic.List[object]

Get-ChildItem -LiteralPath $rootPath -Recurse -File -Force | ForEach-Object {
    $file = $_
        $relative = $file.FullName.Substring($rootPath.Length).TrimStart([char]92, [char]47)
    $parts = $relative -split '[\\/]'

    foreach ($skip in $skipDirs) {
        if ($parts -contains $skip) {
            return
        }
    }

    foreach ($dir in $artifactDirs) {
        if ($parts -contains $dir) {
            $findings.Add([PSCustomObject]@{
                Reason = "artifact-dir:$dir"
                Size = $file.Length
                Path = $relative
            })
            return
        }
    }

    $ext = $file.Extension.ToLowerInvariant()
    if ($binaryExtensions -contains $ext) {
        $findings.Add([PSCustomObject]@{
            Reason = "large-binary:$ext"
            Size = $file.Length
            Path = $relative
        })
        return
    }

    if ($mediaExtensions -contains $ext) {
        $findings.Add([PSCustomObject]@{
            Reason = "media:$ext"
            Size = $file.Length
            Path = $relative
        })
        return
    }

    if ($file.Length -ge $largeBytes) {
        $findings.Add([PSCustomObject]@{
            Reason = "large-file:>=${LargeMB}MB"
            Size = $file.Length
            Path = $relative
        })
    }
}

Write-Output "Scanned: $rootPath"
Write-Output "Findings: $($findings.Count)"

$findings |
    Select-Object -First $Limit |
    ForEach-Object {
        Write-Output "$($_.Reason) $($_.Size) $($_.Path)"
    }

if ($findings.Count -gt $Limit) {
    Write-Output "... $($findings.Count - $Limit) more findings omitted"
}

if ($findings.Count -gt 0) {
    exit 1
}

exit 0