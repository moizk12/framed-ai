# Push to Hugging Face Space Script
# This script helps push the latest commit to HF Space

Write-Host "=== FRAMED - Push to Hugging Face Space ===" -ForegroundColor Cyan
Write-Host ""

# Check current commit
$currentCommit = git rev-parse HEAD
Write-Host "Current commit: $currentCommit" -ForegroundColor Green
Write-Host ""

# Check HF Space status
Write-Host "Checking HF Space status..." -ForegroundColor Yellow
$hfCommit = git ls-remote hf main | Select-String -Pattern "refs/heads/main" | ForEach-Object { $_.ToString().Split("`t")[0] }
Write-Host "HF Space commit: $hfCommit" -ForegroundColor $(if ($hfCommit -eq $currentCommit) { "Green" } else { "Yellow" })
Write-Host ""

if ($hfCommit -eq $currentCommit) {
    Write-Host "✅ HF Space is already up to date!" -ForegroundColor Green
    exit 0
}

Write-Host "⚠️  HF Space needs update. Pushing..." -ForegroundColor Yellow
Write-Host ""

# Method 1: Try with credential helper (if token is stored)
Write-Host "Attempting push with stored credentials..." -ForegroundColor Cyan
$pushResult = git push hf main 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Push successful!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Verifying..." -ForegroundColor Yellow
    $newHfCommit = git ls-remote hf main | Select-String -Pattern "refs/heads/main" | ForEach-Object { $_.ToString().Split("`t")[0] }
    if ($newHfCommit -eq $currentCommit) {
        Write-Host "✅ Verified: HF Space is now at commit $newHfCommit" -ForegroundColor Green
    }
} else {
    Write-Host "❌ Push failed. Authentication required." -ForegroundColor Red
    Write-Host ""
    Write-Host "To push manually, you need a Hugging Face access token:" -ForegroundColor Yellow
    Write-Host "1. Get token from: https://huggingface.co/settings/tokens" -ForegroundColor Cyan
    Write-Host "2. Run one of these commands:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "   Option A (with token in URL):" -ForegroundColor Green
    Write-Host "   git push https://moizk12:YOUR_TOKEN@huggingface.co/spaces/moizk12/framed-ai.git main" -ForegroundColor White
    Write-Host ""
    Write-Host "   Option B (interactive - will prompt):" -ForegroundColor Green
    Write-Host "   git push hf main" -ForegroundColor White
    Write-Host "   (Username: moizk12, Password: YOUR_TOKEN)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "See HF_PUSH_INSTRUCTIONS.md for detailed instructions." -ForegroundColor Cyan
    exit 1
}
