# CODEX_INSTRUCTIONS.md — Bio-PM Pipeline V3 Updates

## What You Are Doing

You are implementing fixes to the Bio-PM IRB feature extraction pipeline based on a full
forensic analysis of the codebase. Three bugs were confirmed through empirical testing.
You will fix them in the project code only. You will not touch anything inside `CS690TR/`.

Read all instructions before writing any code.

---

## Hard Constraints

- **Do not modify any file inside `CS690TR/`**. That directory is read-only.
- **Do not modify any existing pipeline scripts** (`irb_preprocess.py`, `irb_preprocess_alt.py`,
  `irb_extract.py`). Create new v3 versions.
- All new files output to `_v3` paths (`preprocessed_v3/`, `features/biopm_features_v3.npz`, etc.)
- New scripts go in `extraction pipeline/` alongside the existing ones.
- New shell script goes in the project root alongside `run_extraction_pipeline.sh`.
- **The existing `preprocessed/` and `features/biopm_features.npz` must not be touched.**

---

## Background: The Three Confirmed Bugs

### Bug 1: `masked_mean_std` pools all tokens including NaN padding

**Location:** `CS690TR/src/models/biopm.py` (read-only — fix externally)

**What happens:** The function takes a mask argument but ignores it. It runs
`x.mean(dim=1)` and `x.std(dim=1)` over all `pad_size` token slots, including the
NaN-padded ones. With only 18 real tokens and pad_size=57, 67% of slots are padding.
With pad_size=192, 91% are padding. Padding tokens all produce similar near-zero
transformer outputs, so including them crushes the std-pool to near-zero.

**Fix:** Implement a corrected pooling function in `irb_extract_v3.py` that detects
real vs padding tokens by magnitude and pools only over real ones.

### Bug 2: Fill rate is only 33% even with pad_size=57

**What happens:** A 3-second window at 30 Hz = 90 samples produces ~18 movement elements
on average. With pad_size=57, fill is 18/57 = 32%. The alt pipeline tried to fix this by
grouping 3 windows (9s) but scaled pad_size to 171, keeping the ratio identical.

**Fix:** Group 3 consecutive windows into a 9-second block AND keep pad_size=57 (not 171).
In a 9-second block, ~54 movement elements are detected. Fill becomes 54/57 = 94%.
Excess elements beyond 57 are truncated (this is fine — 94% fill means almost none are
truncated in practice).

### Bug 3: Gravity signal is near-zero because DC was removed from data

**What happens:** The IRB accelerometer windows have the gravity DC component removed
before delivery (mean magnitude ~0.026g vs expected ~1.0g). The lowpass filter at 0.5 Hz
extracts almost nothing. The 900-d gravity features are near-noise.

**Fix:** Use raw acc (no filter) for gravity signal. From permutation testing, raw acc
gives 0.107 mean_abs vs 0.066 from lowpass — a real but modest improvement.

**Note on `encoder_gravity`:** The model has a gravity CNN (`encoder_gravity`) but the
checkpoint has ZERO gravity encoder weights. Calling it would use random initialization.
Do not call `encoder_gravity`. Continue using the flat900 approach (interpolate + flatten).

---

## Files to Create

### File 1: `extraction pipeline/irb_preprocess_v3.py`

This is the same logic as `irb_preprocess_alt.py` with exactly two changes:
1. `pad_size = 57` (not `group_size * 57`)
2. Gravity signal uses raw acc (no lowpass filter)

Copy `irb_preprocess_alt.py` as the base. Apply the following modifications:

**Change 1 — pad_size:**
```python
# Find this block in irb_preprocess_alt.py:
pad_size = group_size * 57   # scale with group size

# Replace with:
pad_size = 57   # DO NOT scale — keep fixed to get ~100% fill from 9s blocks
```

**Change 2 — gravity signal in `process_block` function:**
```python
# Find this line (in the process_block function):
acc_grav = lowpass_filter(
    raw_acc, config["LowF1"],
    config["target_FS"], order=config["Order1"]
)

# Replace with:
acc_grav = raw_acc.astype(np.float32)   # raw acc: best gravity proxy when DC is removed
```

**Change 3 — update all output references to use v3 naming:**
The script should write HDF5 files to `preprocessed_v3/` by default when called without
`--output`. Update the default in `argparse`:
```python
p.add_argument("--output", default="preprocessed_v3", ...)
```

**Change 4 — update the print/log messages to say v3:**
In the header print block, change the description to say:
```
Bio-PM IRB Preprocessing V3 (9s grouped, pad_size=57, raw gravity)
```

**Change 5 — update the final "Next step" print to reference v3:**
```python
print(f"Next: python irb_extract_v3.py --preprocessed {args.output} \\")
print(f"        --output features/biopm_features_v3.npz")
```

---

### File 2: `extraction pipeline/irb_extract_v3.py`

Copy `irb_extract.py` as the base. Apply the following modifications:

**Change 1 — add the corrected pooling function:**

Add this function right after the imports, before `load_all_h5`:

```python
def masked_mean_std_valid(tokens):
    """
    Correct implementation of masked mean+std pooling.

    The original masked_mean_std in biopm.py ignores the mask argument and
    pools over all pad_size token slots, including NaN-padded ones. This
    function detects real vs padding tokens by their output magnitude and
    pools only over the real ones.

    Padding detection: the transformer outputs near-zero vectors (magnitude < 1e-4)
    for input rows that were NaN-padded in x_acc_filt. Real ME tokens produce
    outputs with substantially higher magnitude.

    tokens: (B, L, D) — transformer output, D=64
    returns: (B, 128) — [mean_over_real | std_over_real]
    """
    mag   = tokens.abs().mean(dim=-1)     # (B, L) — per-token magnitude
    valid = (mag > 1e-4)                   # (B, L) — True = real token

    B, L, D = tokens.shape
    out_mean = torch.zeros(B, D, device=tokens.device, dtype=tokens.dtype)
    out_std  = torch.zeros(B, D, device=tokens.device, dtype=tokens.dtype)

    for b in range(B):
        real_tokens = tokens[b, valid[b]]    # (n_real, D)
        n_real = real_tokens.shape[0]
        if n_real == 0:
            continue
        out_mean[b] = real_tokens.mean(dim=0)
        if n_real > 1:
            out_std[b] = real_tokens.std(dim=0)
        # if n_real == 1: std stays zero (single token has no variance)

    return torch.cat([out_mean, out_std], dim=-1)   # (B, 128)
```

**Change 2 — use `masked_mean_std_valid` instead of `masked_mean_std` in the extraction loop:**

```python
# Find this line in run_extraction:
pooled = masked_mean_std(tokens)

# Replace with:
pooled = masked_mean_std_valid(tokens)
```

**Change 3 — update default output path:**
```python
p.add_argument("--output", default="features/biopm_features_v3.npz", ...)
```

**Change 4 — update the final print to show v3:**
```python
print("=" * 64)
print("Bio-PM IRB Feature Extraction V3")
print("  Pooling: valid-token-only (masked_mean_std_valid)")
print("  Gravity: flat900 with raw acc input")
print(f"  Saved: {args.output}")
print(f"  features : {features.shape}  (N, 1028)")
print("=" * 64)
```

---

### File 3: `extraction pipeline/export_legacy_schema_v3.py`

Copy `export_legacy_schema.py` as the base. The only change is the default paths:

```python
p.add_argument("--source", default="features/biopm_features_v3.npz")
p.add_argument("--output", default="features/biopm_features_legacy_schema_v3.npz")
```

And update the print header:
```python
print("Legacy Schema Export V3")
```

---

### File 4: `run_extraction_pipeline_v3.sh`

Copy `run_extraction_pipeline.sh` as the base. Update these values:

```bash
PREPROCESSED_DIR="preprocessed_v3"
FEATURES="features/biopm_features_v3.npz"
LEGACY_FEATURES="features/biopm_features_legacy_schema_v3.npz"
```

Update the pipeline steps to use v3 scripts:

```bash
echo "[1/4] V3 preprocessing: 9s grouped blocks, pad_size=57, raw gravity ..."
python "$PIPELINE_DIR/irb_preprocess_v3.py" \
    --data_dir "$DATA_DIR" \
    --output   "$PREPROCESSED_DIR"

echo "[2/4] Extract Bio-PM embeddings with valid-token pooling ..."
python "$PIPELINE_DIR/irb_extract_v3.py" \
    --preprocessed "$PREPROCESSED_DIR" \
    --checkpoint   "$CHECKPOINT" \
    --output       "$FEATURES"

echo "[3/4] Verify v3 features ..."
python "$PIPELINE_DIR/verify_embeddings.py" --features "$FEATURES"

echo "[4/4] Export legacy-compatible visit schema ..."
python "$PIPELINE_DIR/export_legacy_schema_v3.py" \
    --source "$FEATURES" \
    --output "$LEGACY_FEATURES"
```

Update the header print and completion message to say V3.

---

## Verification

After creating all files, run these checks:

```bash
# 1. Syntax check all new Python files
python -m py_compile "extraction pipeline/irb_preprocess_v3.py"
python -m py_compile "extraction pipeline/irb_extract_v3.py"
python -m py_compile "extraction pipeline/export_legacy_schema_v3.py"

# 2. Help check
BIOPM_ROOT=CS690TR python "extraction pipeline/irb_preprocess_v3.py" --help
BIOPM_ROOT=CS690TR python "extraction pipeline/irb_extract_v3.py" --help

# 3. Confirm CS690TR is untouched
diff -rq CS690TR/src CS690TR/CS690TR/src --exclude='*.pyc' --exclude='__pycache__'
# Should output only pyc differences, no .py file differences
```

---

## What the V3 Pipeline Produces

```
preprocessed_v3/           HDF5 files: 9s blocks, pad_size=57, raw acc gravity
features/
  biopm_features_v3.npz            shape (N_windows, 1028)
  biopm_features_legacy_schema_v3.npz
    features      (198, 1028)     visit-level mean-pooled
    features_even (198, 1028)     even blocks (for ICC)
    features_odd  (198, 1028)     odd blocks (for ICC)
    labels, pids, arat, fma, subjects, weeks
```

Expected improvements vs standard pipeline:
- Fill rate: 33% -> 94%
- std_pool mean_abs: ~0.42 -> ~0.50 (based on autopsy permutation data)
- Gravity mean_abs: ~0.066 -> ~0.107 (raw acc vs lowpass)

---

## Do Not Do

- Do not call `model.encoder_gravity(...)`. The checkpoint has no gravity encoder weights.
  Using it would produce random-initialized features.
- Do not scale pad_size with group_size. It must stay at 57.
- Do not modify any file in CS690TR/.
- Do not modify any existing pipeline files (irb_preprocess.py, irb_extract.py, etc).
