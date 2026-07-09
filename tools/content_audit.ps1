param(
    [string]$Root = "release\VFM-FP-open-source",
    [int]$Limit = 100
)

$ErrorActionPreference = "Stop"

$rootPath = (Resolve-Path -LiteralPath $Root).Path
$skipDirs = @(".git", "__pycache__", ".ipynb_checkpoints")
$skipFiles = @("tools\content_audit.ps1")
$caseInsensitivePatterns = @(
    "C:\\Users\\",
    "E:\\",
    "D:\\",
    "/home/",
    "\\home\\",
    "87317",
    "password",
    "secret",
    "api[_-]?key",
    "access[_-]?token",
    "bearer[ ]+"
)
$caseSensitivePatterns = @(
    "TODO",
    "PLACEHOLDER",
    "TBD"
)

$findings = New-Object System.Collections.Generic.List[object]

function Add-Matches {
    param(
        [string]$FilePath,
        [string]$RelativePath,
        [array]$Patterns,
        [bool]$CaseSensitive
    )

    try {
        $matches = Select-String -LiteralPath $FilePath -Pattern $Patterns -CaseSensitive:$CaseSensitive
        foreach ($match in $matches) {
            $script:findings.Add([PSCustomObject]@{
                Path = $RelativePath
                Line = $match.LineNumber
                Text = $match.Line.Trim()
            })
        }
    }
    catch {
        return
    }
}

Get-ChildItem -LiteralPath $rootPath -Recurse -File -Force | ForEach-Object {
    $file = $_
    $relative = $file.FullName.Substring($rootPath.Length).TrimStart([char]92, [char]47)
    $parts = $relative -split "[\/]"

    foreach ($skip in $skipDirs) {
        if ($parts -contains $skip) {
            return
        }
    }

    foreach ($skipFile in $skipFiles) {
        if ($relative -eq $skipFile) {
            return
        }
    }

    Add-Matches -FilePath $file.FullName -RelativePath $relative -Patterns $caseInsensitivePatterns -CaseSensitive $false
    Add-Matches -FilePath $file.FullName -RelativePath $relative -Patterns $caseSensitivePatterns -CaseSensitive $true
}

Write-Output "Scanned: $rootPath"
Write-Output "Findings: $($findings.Count)"

$findings |
    Select-Object -First $Limit |
    ForEach-Object {
        Write-Output "$($_.Path):$($_.Line): $($_.Text)"
    }

if ($findings.Count -gt $Limit) {
    Write-Output "... $($findings.Count - $Limit) more findings omitted"
}

if ($findings.Count -gt 0) {
    exit 1
}

exit 0