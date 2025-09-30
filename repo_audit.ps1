param(
  [string]$Repo = "moizk12/framed-ai",
  [string]$Branch = "main",
  [int64]$LargeFileMB = 50
)

function Info($msg){ Write-Host "[i] $msg" -ForegroundColor Cyan }
function Ok($msg){ Write-Host "[OK] $msg" -ForegroundColor Green }
function Warn($msg){ Write-Host "[!] $msg" -ForegroundColor Yellow }
function Err($msg){ Write-Host "[x] $msg" -ForegroundColor Red }

# Pre-checks
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Err "git not found"; exit 1 }
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { Err "GitHub CLI (gh) not found"; exit 1 }

# Create temp workspace
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$root = Join-Path $env:TEMP "framed-audit-$stamp"
New-Item -ItemType Directory -Path $root | Out-Null

# Pull repo fresh
Info "Cloning $Repo -> $root"
$clone = gh repo clone $Repo $root 2>&1
if ($LASTEXITCODE -ne 0) { Err "Clone failed: $clone"; exit 1 }
Set-Location $root

# Report file
$report = Join-Path $root "AuditReport.md"
"## Repo Audit - $Repo ($stamp)`n" | Out-File -Encoding UTF8 $report

# === Repo basics via API ===
$repoInfoJson = gh api "/repos/$Repo" 2>$null
if (-not $?) {
  Err "Failed to fetch repo info via API"
} else {
  $repoInfo = $repoInfoJson | ConvertFrom-Json
  "### Basics`n- Default branch: **$($repoInfo.default_branch)**`n- Private: **$($repoInfo.private)**`n- Topics: $([string]::Join(', ', $repoInfo.topics))`n" | Add-Content $report
  if ($repoInfo.default_branch -ne $Branch) { Warn "Default branch is $($repoInfo.default_branch), expected $Branch" } else { Ok "Default branch = $Branch" }
}

# === Branch protection ===
$protJson = gh api "/repos/$Repo/branches/$Branch/protection" 2>$null
if (-not $?) {
  Warn "No branch protection found on $Branch"
  "### Branch Protection`n- Not found for $Branch`n" | Add-Content $report
} else {
  $prot = $protJson | ConvertFrom-Json
  $enfAdmins = $prot.enforce_admins.enabled
  $prReq = $prot.required_pull_request_reviews.required_approving_review_count
  $strict = $prot.required_status_checks.strict
  $contexts = $prot.required_status_checks.contexts
  "### Branch Protection`n- enforce_admins: **$enfAdmins**`n- required_reviews: **$prReq**`n- strict_checks: **$strict**`n- contexts: $([string]::Join(', ', $contexts))`n" | Add-Content $report
  if ($enfAdmins -and $prReq -ge 1 -and $strict) { Ok "Branch protection looks good" } else { Warn "Branch protection missing one or more requirements" }
}

# === Release v0.1.0 ===
$relList = gh release list --limit 100 2>$null
$hasRel = $relList -match "^v0\.1\.0\s"
"### Releases`n$relList`n" | Add-Content $report
if ($hasRel) { Ok "Release v0.1.0 exists" } else { Warn "Release v0.1.0 not found" }

# === Actions / CI ===
$runs = gh run list --limit 5 2>$null
"### Recent CI runs`n$runs`n" | Add-Content $report
if ($runs -match "success") { Ok "At least one recent CI run succeeded" } else { Warn "No recent successful CI runs" }

# === Files: existence checks ===
$expected = @(
  "README.md",
  "LICENSE",
  ".gitignore",
  ".env.example",
  ".github\workflows\ci.yml",
  ".github\ISSUE_TEMPLATE\bug.yml",
  ".github\ISSUE_TEMPLATE\feature.yml",
  ".github\pull_request_template.md",
  "docs\arch.md",
  "docs\roadmap.md",
  "docs\contributing.md",
  "docs\security.md",
  "docs\echo.md",
  "docs\remix.md"
)

$missing = @()
$expected | ForEach-Object {
  if (-not (Test-Path $_)) { $missing += $_ }
}
"### Files - Missing`n$(if($missing.Count){($missing -join "`n")}else{"None"})`n" | Add-Content $report
if ($missing.Count -eq 0) { Ok "All expected files present" } else { Warn "Missing files: $($missing -join ', ')" }

# === README badge & CI contents ===
$readmeOk = $false
if (Test-Path "README.md") {
  $readme = Get-Content -Raw README.md
  if ($readme -match "/actions/workflows/ci.yml") { $readmeOk = $true }
}
if ($readmeOk) { Ok "README contains CI badge reference" } else { Warn "README missing CI badge link (/actions/workflows/ci.yml)" }

$ciOk = $false
if (Test-Path ".github\workflows\ci.yml") {
  $ci = Get-Content -Raw ".github\workflows\ci.yml"
  if ($ci -match "ruff check" -and $ci -match "pytest" -and $ci -match "bandit") { $ciOk = $true }
}
if ($ciOk) { Ok "CI workflow has ruff/pytest/bandit steps" } else { Warn "CI workflow missing expected steps" }

# === Secrets scan (very simple heuristics) ===
$secretHits = @()
$secretHits += (git grep -n "sk-" 2>$null)
$secretHits += (git grep -n "OPENAI_API_KEY" 2>$null)
"### Secret scan (heuristic)`n$(if($secretHits){$secretHits -join "`n"}else{"No obvious secrets found"})`n" | Add-Content $report
if ($secretHits){ Warn "Potential secrets found in history/tree (review report)" } else { Ok "No obvious secrets found" }

# === Ensure .env not tracked ===
$trackedEnv = (git ls-files ".env")
if ($trackedEnv) { Warn ".env appears tracked by git!" } else { Ok ".env is not tracked" }

# === Large files scan ===
$threshold = $LargeFileMB * 1MB
$big = Get-ChildItem -Recurse -File | Where-Object { $_.Length -gt $threshold }
"### Large files (>${LargeFileMB}MB)`n$(if($big){ ($big | ForEach-Object { "$($_.FullName)  $([math]::Round($_.Length/1MB,2)) MB" }) -join "`n" } else {"None"})`n" | Add-Content $report
if ($big) { Warn "Large files present (>$LargeFileMB MB)" } else { Ok "No large files over ${LargeFileMB}MB" }

# === Topics ===
$topicsJson = gh api "/repos/$Repo/topics" -H "Accept: application/vnd.github+json" 2>$null
if ($?) {
  $topics = ($topicsJson | ConvertFrom-Json).names
  "### Topics`n- $([string]::Join(', ', $topics))`n" | Add-Content $report
}

# === Branch / HEAD check ===
$current = (git branch --show-current)
"### Branch/HEAD`n- Local HEAD: **$current**`n" | Add-Content $report
if ($current -eq $Branch) { Ok "Local HEAD on $Branch" } else { Warn "Local HEAD not on $Branch" }

# Finish
"---`nAudit complete. Open this file for details:`n$report" | Add-Content $report
Ok "Audit complete -> $report"
$resp = Read-Host "Open report? (y/n)"
if ($resp -match 'y') { Start-Process -FilePath notepad.exe -ArgumentList $report }
