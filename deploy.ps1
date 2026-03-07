# ═══════════════════════════════════════════════════════════
#  health-mcp-rag · One-click deploy to GitHub
#  Run from project root: .\deploy.ps1
# ═══════════════════════════════════════════════════════════

Set-Location "C:\Users\Zhi\Desktop\health-mcp-rag"

Write-Host "`n🔍 Checking status..." -ForegroundColor Cyan
git status

Write-Host "`n📦 Staging all changes..." -ForegroundColor Cyan
git add .

# Auto-generate commit message with timestamp
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
$message = "feat: LLM synthesis on all agent tools, nav tabs, UI improvements [$timestamp]"

Write-Host "`n✍️  Committing: $message" -ForegroundColor Cyan
git commit -m $message

Write-Host "`n🚀 Pushing to GitHub..." -ForegroundColor Cyan
git push origin master

Write-Host "`n✅ Done! Changes pushed to GitHub." -ForegroundColor Green
Write-Host "   Railway will auto-deploy in ~1-2 minutes." -ForegroundColor Green
Write-Host "   Check: https://railway.app/dashboard`n" -ForegroundColor Yellow
