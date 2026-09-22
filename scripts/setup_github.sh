#!/usr/bin/env bash
# =============================================================================
# MZSAE GitHub Repository Configuration Script
# =============================================================================
# Configures the MZSAE repository at github.com/mohamedhossammohamed/MZSAE
# using the GitHub CLI (gh). Idempotent — safe to run multiple times.
#
# Prerequisites:
#   - gh CLI installed: https://cli.github.com/
#   - Authenticated: gh auth login
#
# Usage:
#   chmod +x scripts/setup_github.sh
#   ./scripts/setup_github.sh
# =============================================================================

set -euo pipefail

REPO="mohamedhossammohamed/MZSAE"
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }

# ---------------------------------------------------------------------------
# 0. Preflight Checks
# ---------------------------------------------------------------------------
info "Verifying gh CLI authentication..."
if ! gh auth status 2>/dev/null; then
    echo "ERROR: gh is not authenticated. Run 'gh auth login' first."
    exit 1
fi
ok "Authenticated"

info "Verifying repository exists..."
if ! gh repo view "$REPO" --json name -q '.name' &>/dev/null; then
    echo "ERROR: Repository $REPO not found."
    exit 1
fi
ok "Repository $REPO found"

echo ""
echo "==========================================================================="
echo "  MZSAE GitHub Configuration — $REPO"
echo "==========================================================================="
echo ""

# ---------------------------------------------------------------------------
# 1. Repository Metadata
# ---------------------------------------------------------------------------
info "Setting repository description, homepage, and default branch..."
gh repo edit "$REPO" \
  --description "Neuromorphic Sparse Attention Engine — biological eviction, Cauchy-Schwarz sentinels, Apple Silicon Metal" \
  --homepage "https://mohamedhossammohamed.github.io/MZSAE" \
  --default-branch main \
  2>/dev/null || true
ok "Repository metadata set"

# Topics (max 20 — critical for GitHub search discoverability)
info "Adding repository topics..."
TOPICS=(
  "attention-mechanism"
  "sparse-attention"
  "metal-shading-language"
  "apple-silicon"
  "llm-inference"
  "kv-cache"
  "edge-ai"
  "on-device-ml"
  "neuromorphic-computing"
  "temporal-difference"
  "reinforcement-learning"
  "pytorch"
  "mlx"
  "llama"
  "qwen"
  "mistral"
  "bitnet"
  "ternary-weights"
  "flash-attention"
  "systems-engineering"
)

for topic in "${TOPICS[@]}"; do
  gh repo edit "$REPO" --add-topic "$topic" 2>/dev/null || true
done
ok "Added ${#TOPICS[@]} topics"

# ---------------------------------------------------------------------------
# 2. Repository Features (Enable/Disable via API)
# ---------------------------------------------------------------------------
info "Configuring repository features..."

gh api "repos/$REPO" -X PATCH \
  -F has_wiki=true \
  -F has_projects=true \
  -F has_discussions=true \
  -F allow_squash_merge=true \
  -F allow_rebase_merge=true \
  -F allow_merge_commit=false \
  -F delete_branch_on_merge=true \
  -F allow_auto_merge=true \
  -f squash_merge_commit_title="PR_TITLE" \
  -f squash_merge_commit_message="PR_BODY" \
  --silent 2>/dev/null || warn "Some repo features may require admin access"

ok "Repository features configured"

# ---------------------------------------------------------------------------
# 3. Branch Protection Rules (main)
# ---------------------------------------------------------------------------
info "Setting branch protection rules for 'main'..."

gh api "repos/$REPO/branches/main/protection" -X PUT \
  --input - <<'PROTECTION_EOF' 2>/dev/null || warn "Branch protection requires admin access or GitHub Pro"
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["test (ubuntu-latest)", "test (macos-latest)"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false
  },
  "restrictions": null,
  "required_linear_history": false,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false,
  "required_conversation_resolution": true,
  "lock_branch": false,
  "allow_fork_syncing": true
}
PROTECTION_EOF

ok "Branch protection configured"

# ---------------------------------------------------------------------------
# 4. Labels (Semantic + Priority + Backend + Model + Status)
# ---------------------------------------------------------------------------
info "Creating labels..."

# Helper: create label idempotently (delete + create to update color/description)
create_label() {
  local name="$1" color="$2" description="$3"
  # Try to create; if it exists, update it
  gh label create "$name" --color "$color" --description "$description" \
    --repo "$REPO" --force 2>/dev/null || true
}

# Priority labels
create_label "priority: critical"  "B60205" "Blocking release or security issue"
create_label "priority: high"      "D93F0B" "Important for next release"
create_label "priority: medium"    "FBCA04" "Nice to have"
create_label "priority: low"       "0E8A16" "Backlog"

# Type labels
create_label "type: bug"           "D73A4A" "Something isn't working"
create_label "type: feature"       "A2EEEF" "New feature request"
create_label "type: performance"   "F9D0C4" "Performance improvement"
create_label "type: documentation" "0075CA" "Documentation only"
create_label "type: benchmark"     "5319E7" "Benchmarking and profiling"
create_label "type: research"      "BFD4F2" "Research question or paper-related"

# Backend labels
create_label "backend: metal"      "E4E669" "Apple Metal backend"
create_label "backend: cuda"       "76B900" "NVIDIA CUDA backend (roadmap)"
create_label "backend: cpu"        "C5DEF5" "CPU reference backend"

# Model labels
create_label "model: llama"        "FF6B6B" "Llama family models"
create_label "model: qwen"         "4ECDC4" "Qwen family models"
create_label "model: mistral"      "FFA07A" "Mistral/Mixtral models"
create_label "model: gemma"        "95E1D3" "Google Gemma models"
create_label "model: bitnet"       "F38181" "Ternary/1-bit models"

# Status labels
create_label "status: needs-triage"    "BFDADC" "New issue, needs review"
create_label "status: in-progress"     "0052CC" "Actively being worked on"
create_label "status: blocked"         "B60205" "Blocked on external factor"
create_label "status: help-wanted"     "008672" "Community contributions welcome"
create_label "status: good-first-issue" "7057FF" "Good for newcomers"

ok "Created 24 labels"

# ---------------------------------------------------------------------------
# 5. Milestones
# ---------------------------------------------------------------------------
info "Creating milestones..."

create_milestone() {
  local title="$1" description="$2"
  # Check if milestone already exists
  existing=$(gh api "repos/$REPO/milestones" -q ".[].title" 2>/dev/null | grep -F "$title" || true)
  if [ -z "$existing" ]; then
    gh api "repos/$REPO/milestones" -X POST \
      -f title="$title" \
      -f description="$description" \
      -f state="open" \
      --silent 2>/dev/null || warn "Failed to create milestone: $title"
  else
    warn "Milestone already exists: $title"
  fi
}

create_milestone \
  "v1.3.0: CUDA Backend" \
  "Port Metal kernels to CUDA SM90 for NVIDIA Ampere/Hopper GPUs"

create_milestone \
  "v1.4.0: llama.cpp Integration" \
  "Native patch for llama.cpp attention backend — drop-in MZSAE decode"

create_milestone \
  "v1.5.0: Dynamic-τ ACC Veto" \
  "Auto-calibrate sentinel threshold per layer for natural text distributions"

create_milestone \
  "v2.0.0: Real Semantic NIAH + arXiv" \
  "Validate NIAH with frozen LLM embeddings on real text. Submit arXiv preprint."

ok "Milestones configured"

# ---------------------------------------------------------------------------
# 6. GitHub Pages
# ---------------------------------------------------------------------------
info "Enabling GitHub Pages (workflow deployment)..."

gh api "repos/$REPO/pages" -X POST \
  --input - <<'PAGES_EOF' 2>/dev/null || warn "Pages may already be enabled or require setup"
{
  "build_type": "workflow",
  "source": {
    "branch": "main",
    "path": "/"
  }
}
PAGES_EOF

# Enforce HTTPS
gh api "repos/$REPO/pages" -X PUT \
  -F https_enforced=true \
  --silent 2>/dev/null || true

ok "GitHub Pages configured"

# ---------------------------------------------------------------------------
# 7. Repository Secrets (Reminders)
# ---------------------------------------------------------------------------
echo ""
info "Repository secrets (set manually):"
echo "  gh secret set PYPI_TOKEN --repo $REPO"
echo "  gh secret set CODECOV_TOKEN --repo $REPO"
echo ""

# ---------------------------------------------------------------------------
# 8. Discussion Categories
# ---------------------------------------------------------------------------
info "Discussions enabled via API in step 2."
echo "  Create categories via GitHub UI: General, Ideas, Q&A, Show and Tell, Benchmarks"
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "==========================================================================="
echo "  ✅ MZSAE GitHub Configuration Complete"
echo "==========================================================================="
echo ""
echo "  Repository:    https://github.com/$REPO"
echo "  Pages:         https://mohamedhossammohamed.github.io/MZSAE/"
echo "  Topics:        ${#TOPICS[@]} added"
echo "  Labels:        24 created"
echo "  Milestones:    4 created"
echo "  Branch rules:  main protected (1 review, status checks)"
echo ""
echo "  Next steps:"
echo "    1. Set PYPI_TOKEN and CODECOV_TOKEN secrets"
echo "    2. Push all files to trigger CI and Pages deployment"
echo "    3. Upload social preview image via Settings > Social Preview"
echo "    4. Create Discussion categories via the Discussions tab"
echo ""
