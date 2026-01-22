# Hugging Face Space Push Instructions

## Current Status
- ✅ GitHub push: **SUCCESSFUL** (commit `3f8ce8b`)
- ❌ HF Space push: **FAILED** (requires authentication token)
- HF Space is at commit `bf3b226` (needs `3f8ce8b`)

## Solution: Push with Hugging Face Access Token

### Option 1: Push with Token in URL (One-time)

Replace `YOUR_HF_TOKEN` with your actual Hugging Face access token:

```bash
git push https://moizk12:YOUR_HF_TOKEN@huggingface.co/spaces/moizk12/framed-ai.git main
```

### Option 2: Configure Git Credential Helper (Recommended)

1. **Get your Hugging Face access token:**
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with "write" permissions
   - Copy the token

2. **Configure git to use the token:**
   ```bash
   git config --global credential.helper store
   ```

3. **Push (will prompt for credentials once):**
   ```bash
   git push hf main
   ```
   - Username: `moizk12`
   - Password: `YOUR_HF_TOKEN` (paste your token here)

4. **After first push, credentials are stored and future pushes won't prompt**

### Option 3: Use Environment Variable

Set the token as an environment variable and update remote URL:

```powershell
# PowerShell
$env:HF_TOKEN = "YOUR_HF_TOKEN"
git remote set-url hf https://moizk12:$env:HF_TOKEN@huggingface.co/spaces/moizk12/framed-ai.git
git push hf main
```

### Option 4: Use Hugging Face CLI (Alternative)

If you have `huggingface-cli` installed:

```bash
huggingface-cli login
git push hf main
```

## Verify Push Success

After pushing, verify with:

```bash
git ls-remote hf main
```

Should show: `3f8ce8b7a043f91bb3a6f309777189aec91642ca`

## Important Notes

- **Never commit tokens to git** - they should only be in credential storage or environment variables
- The HF Space will **auto-deploy** after the push completes
- If the Space is connected to GitHub, it may auto-sync, but manual push ensures immediate deployment
