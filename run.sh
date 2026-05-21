#!/bin/bash
set -euo pipefail

# ─── AutoReg Splat: Full Training Pipeline ───
# Runs all phases end-to-end on a single GPU.
# Expected hardware: NVIDIA A6000 96GB
# Expected total time: ~4-5 hours with 5000 scenes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUM_SCENES="${1:-5000}"
CONFIG="configs/default.yaml"
DATA_DIR="data/synthetic"
TOKENIZER_CKPT_DIR="checkpoints/tokenizer"
MODEL_CKPT_DIR="checkpoints/model"
OUTPUT_DIR="outputs"

echo "════════════════════════════════════════════════════════"
echo "  AutoReg Splat — Full Training Pipeline"
echo "  Scenes: ${NUM_SCENES}"
echo "  Config: ${CONFIG}"
echo "════════════════════════════════════════════════════════"

# ── check GPU ──
if command -v nvidia-smi &>/dev/null; then
    echo ""
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found, proceeding anyway"
fi

START_TIME=$(date +%s)

# ─────────────────────────────────────────────
# Phase 0: Generate synthetic data (~2-3 min)
# ─────────────────────────────────────────────
echo ""
echo "▶ Phase 0: Generating ${NUM_SCENES} synthetic scenes..."
PHASE0_START=$(date +%s)

python3 scripts/generate_synthetic.py \
    --output_dir "$DATA_DIR" \
    --num_scenes "$NUM_SCENES" \
    --seed 42

PHASE0_END=$(date +%s)
echo "✓ Phase 0 complete — $((PHASE0_END - PHASE0_START))s"

# ─────────────────────────────────────────────
# Phase 1: Train RVQ-VAE tokenizer (~10 min)
# ─────────────────────────────────────────────
echo ""
echo "▶ Phase 1: Training Gaussian tokenizer (RVQ-VAE)..."
PHASE1_START=$(date +%s)

python3 scripts/train_tokenizer.py \
    --config "$CONFIG" \
    --data_dir "$DATA_DIR" \
    --output_dir "$TOKENIZER_CKPT_DIR"

PHASE1_END=$(date +%s)
echo "✓ Phase 1 complete — $((PHASE1_END - PHASE1_START))s"

# ── Phase 1.5: Diagnose tokenizer ──
echo ""
echo "▶ Phase 1.5: Diagnosing tokenizer quality..."
python3 scripts/diagnose_tokenizer.py \
    --tokenizer_ckpt "${TOKENIZER_CKPT_DIR}/tokenizer_best.pt" \
    --data_dir "$DATA_DIR" \
    --config "$CONFIG"

# ─────────────────────────────────────────────
# Phase 2: Train AR transformer (~3-4 hours)
# ─────────────────────────────────────────────
echo ""
echo "▶ Phase 2: Training autoregressive transformer..."
PHASE2_START=$(date +%s)

python3 scripts/train_model.py \
    --config "$CONFIG" \
    --data_dir "$DATA_DIR" \
    --tokenizer_ckpt "${TOKENIZER_CKPT_DIR}/tokenizer_best.pt" \
    --output_dir "$MODEL_CKPT_DIR"

PHASE2_END=$(date +%s)
echo "✓ Phase 2 complete — $((PHASE2_END - PHASE2_START))s"

# ─────────────────────────────────────────────
# Phase 3: Quick evaluation
# ─────────────────────────────────────────────
echo ""
echo "▶ Phase 3: Generating sample outputs..."
PHASE3_START=$(date +%s)

# use a random scene's top-down image as test input
TEST_IMAGE=$(find "$DATA_DIR" -name "top_down.png" | shuf -n 1)
echo "  Test input: ${TEST_IMAGE}"

python3 scripts/evaluate.py \
    --config "$CONFIG" \
    --model_ckpt "${MODEL_CKPT_DIR}/model_best.pt" \
    --tokenizer_ckpt "${TOKENIZER_CKPT_DIR}/tokenizer_best.pt" \
    --input_image "$TEST_IMAGE" \
    --output_dir "$OUTPUT_DIR" \
    --max_gaussians 512 \
    --num_samples 4

PHASE3_END=$(date +%s)
echo "✓ Phase 3 complete — $((PHASE3_END - PHASE3_START))s"

# ─────────────────────────────────────────────
# Phase 4: Full model diagnostics
# ─────────────────────────────────────────────
echo ""
echo "▶ Phase 4: Running model diagnostics..."

python3 scripts/diagnose_model.py \
    --config "$CONFIG" \
    --model_ckpt "${MODEL_CKPT_DIR}/model_best.pt" \
    --tokenizer_ckpt "${TOKENIZER_CKPT_DIR}/tokenizer_best.pt" \
    --data_dir "$DATA_DIR"

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
END_TIME=$(date +%s)
TOTAL=$((END_TIME - START_TIME))
HOURS=$((TOTAL / 3600))
MINS=$(( (TOTAL % 3600) / 60 ))

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Pipeline complete!"
echo ""
echo "  Phase 0 (data gen):   $((PHASE0_END - PHASE0_START))s"
echo "  Phase 1 (tokenizer):  $((PHASE1_END - PHASE1_START))s"
echo "  Phase 2 (transformer): $((PHASE2_END - PHASE2_START))s"
echo "  Phase 3 (eval):       $((PHASE3_END - PHASE3_START))s"
echo "  ──────────────────────"
echo "  Total:                ${HOURS}h ${MINS}m"
echo ""
echo "  Checkpoints: ${TOKENIZER_CKPT_DIR}/, ${MODEL_CKPT_DIR}/"
echo "  Outputs:     ${OUTPUT_DIR}/"
echo "════════════════════════════════════════════════════════"
