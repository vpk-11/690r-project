# Member 1 Pipeline Documentation
## CS690R — Clinical Biomarker Extraction via Wearable Accelerometers

---

## What This Is

This is the full documentation for Member 1's contribution to the CS690R project. The job was to take raw wrist accelerometer data, run it through a pretrained movement encoder called Bio-PM, and produce a structured feature matrix that downstream team members can use for biomarker validation, regression, and reporting.

Two datasets were used: an IRB-approved stroke recovery dataset (24-hour continuous recordings) and the PADS Parkinson's Disease Smartwatch dataset (structured 10-second clinical tasks). Each has its own notebook. They are different datasets with different goals, and the approach had to be adapted significantly for each.

---

## Repository Structure

```
690r-project/
├── setup.sh                              # Downloads PADS, extracts Bio-PM, creates output dirs
├── biopm690r.zip                         # Bio-PM model source (do not modify)
├── CS690TR/                              # Unzipped Bio-PM model (gitignored)
├── biomarker_feature_extraction.ipynb    # IRB stroke dataset pipeline
├── pads_biomarker_extraction.ipynb       # PADS Parkinson's pipeline
├── data/                                 # IRB dataset (gitignored — confidential)
│   ├── windows.npz
│   └── clinical_scores.npz
├── pads_data/                            # PADS dataset (gitignored)
│   ├── preprocessed/
│   │   ├── file_list.csv
│   │   └── movement/{id}_ml.bin
│   └── movement/timeseries/{id}_{task}_{wrist}.txt
├── features/                             # Generated feature matrices (gitignored)
└── results/                              # Figures and metrics (gitignored)
```

---

## Setup

```bash
# Clone the repo, then:
chmod +x setup.sh && ./setup.sh
```

The script downloads the PADS dataset (~735 MB via wget or aws s3 sync), extracts Bio-PM from the zip, and creates the output directories. It does not manage your Python environment — that is your choice:

```bash
# conda (used in development, Python 3.11.15)
conda create -n biopm python=3.11 && conda activate biopm

# or venv
python -m venv .venv && source .venv/bin/activate

# or uv
uv venv && source .venv/bin/activate
```

Then install dependencies:

```bash
pip install -r CS690TR/requirements.txt
pip install umap-learn seaborn scikit-learn jupyter
```

The IRB data (`data/windows.npz` and `data/clinical_scores.npz`) is shared separately under the class Data Use Agreement. Place it in `data/` before running the IRB notebook.

---

## Bio-PM Overview

Bio-PM is a dual-stream transformer pretrained on free-living wrist accelerometer data. It takes 3-axis accelerometer input at 30 Hz in 10-second windows and produces a 1028-dimensional embedding.

The two streams are:

**Acc stream (dims 0-127):** A transformer that operates over movement elements — discrete acceleration events detected by a zero-crossing algorithm on the bandpass-filtered signal. This stream captures movement quality: tremor, bradykinesia, hesitation. It outputs a 128-d mean-pooled representation.

**Gravity stream (dims 128-1028):** A 1D CNN that operates on the low-pass filtered signal to extract body orientation and postural information. It outputs a 900-d flattened representation.

The two are concatenated to form the 1028-d embedding. The model is loaded from `CS690TR/checkpoints/checkpoint.pt` and used in inference mode only — weights are frozen throughout.

```python
sys.path.insert(0, 'CS690TR')
from src.models.biopm import load_pretrained_encoder, masked_mean_std
from src.data.preprocessing import (
    bandpass_filter, lowpass_filter,
    detect_zero_crossings, assign_zero_crossings,
)

model = load_pretrained_encoder(
    'CS690TR/checkpoints/checkpoint.pt', n_classes=11, device='mps'
)
model.eval()
```

The "unexpected keys" warning on load (decoder_cnn weights) is expected and harmless. We only use the encoder.

---

## Dataset 1: IRB Stroke Recovery Dataset

**Notebook:** `biomarker_feature_extraction.ipynb`

### Dataset Description

61 subjects: 57 stroke survivors and 4 healthy controls. Stroke subjects were assessed at up to 8 visits across their recovery timeline. Healthy controls were assessed once. Total: 198 clinical visits.

Recordings are 24-hour wrist accelerometer sessions. The data arrives pre-windowed into 3-second sliding windows at 30 Hz. Each window is a SimpleNamespace object with `.acc`, `.vel`, `.pos`, `.jerk` arrays of shape (90, 3) and metadata fields `.subject`, `.week`, `.start_idx`.

Two files:
- `windows.npz` — 587,046 window objects
- `clinical_scores.npz` — dict keyed by `(subject_id, week)` mapping to clinical scores (`.ARAT` and `.FMA`)

### Why Not Bio-PM Directly

The IRB data is pre-filtered. By the time we receive it, the bandpass processing has already been applied. Bio-PM's movement element detector relies on detecting zero crossings in the bandpass signal. On pre-filtered data, this produces zero crossings = 0 for virtually every window. The acc stream produces all-zeros embeddings. Running Bio-PM on this dataset gives no useful signal from the transformer stream.

The gravity stream would work in isolation, but the signal needed for stroke motor recovery analysis — arm kinematics, jerk, movement smoothness — is more directly captured by computing those features explicitly from the `.acc`, `.vel`, `.pos`, `.jerk` fields that are already computed and provided.

The decision was to use Bio-PM's preprocessing code where useful (the low-pass filter structure) but extract kinematic features directly from the pre-computed arrays, which is consistent with how the assignment framed the feature extraction task.

### Feature Extraction

**Per window (38 features):** For each axis (X, Y, Z) of acceleration, velocity, and jerk, compute mean, std, and max. Compute the same three statistics on the vector magnitude. For position, compute total displacement (L2 norm of start-to-end vector) and cumulative path length (sum of L2 norms of consecutive differences). This gives 3 × 12 + 2 = 38 features per window.

```python
def extract_features_from_wnd(wnd):
    # Acceleration: mean/std/max per axis + magnitude (12 features)
    # Velocity:     mean/std/max per axis + magnitude (12 features)
    # Jerk:         mean/std/max per axis + magnitude (12 features)
    # Position:     displacement + path_length         (2 features)
    # Total: 38 features
```

**Per visit (228 features):** Each visit contains hundreds to thousands of windows. The 38 per-window features are aggregated across all windows in a visit using 6 summary statistics: mean, std, IQR, 10th percentile, 50th percentile, 90th percentile. This gives 38 × 6 = 228 features per visit.

The 6-statistic aggregation was chosen because it captures both the central tendency and the distribution shape of movement patterns over the full recording. A subject with high tremor variability will have a very different IQR and range of jerk features than a healthy control, even if their means are similar.

### Labels

Healthy controls were assigned maximum clinical scores (ARAT = 57, FMA = 66) without formal testing. This is the convention in the dataset. Labels are derived directly from these scores:

```python
labels = [0 if (ARAT == 57 and FMA == 66) else 1 for each visit]
```

This approach is more robust than matching subject IDs across files, which had a Python type mismatch issue (numpy integer vs Python integer) that silently produced 0 healthy subjects. Deriving from scores avoids the matching problem entirely.

### Test-Retest Reliability Split

The 24-hour recording is split into alternating 30-minute blocks. Even blocks (0-30 min, 60-90 min, ...) form one set; odd blocks (30-60 min, 90-120 min, ...) form another. Features are extracted from each set independently, producing `features_even` and `features_odd`.

This split exists for downstream ICC (Intraclass Correlation Coefficient) analysis by Member 3. A feature with high ICC across the two splits is reliable — it does not change just because you sampled a different half of the recording. A feature with low ICC is noisy and should not be trusted as a clinical biomarker regardless of its discriminative accuracy.

Block size: 54,000 samples (30 minutes × 60 seconds × 30 Hz).

### Output

`features/biopm_features.npz`:

| Key | Shape | Description |
|---|---|---|
| `features` | (198, 228) | Main feature matrix, one row per visit |
| `features_even` | (198, 228) | Even 30-min blocks for ICC analysis |
| `features_odd` | (198, 228) | Odd 30-min blocks for ICC analysis |
| `feature_names` | (228,) | Name of each feature column |
| `labels` | (198,) | 0 = healthy, 1 = stroke |
| `pids` | (198,) | Integer subject ID for LOSO grouping |
| `arat` | (198,) | ARAT clinical score (0-57) |
| `fma` | (198,) | FMA-UE clinical score (0-66) |
| `subjects` | (198,) | Original subject identifiers |
| `weeks` | (198,) | Clinical visit week number |

### Baseline Classification Results

LOSO logistic regression on the 228-d feature matrix: **AUC 0.514**.

This is expected and not a pipeline failure. The dataset has 4 healthy visits and 194 stroke visits. Any classifier trained on such an imbalanced set learns to predict "stroke" for everything — that gives 98% accuracy but meaningless AUC. The binary LOSO here is a sanity check, not the actual analysis.

The actual biomarker work uses `arat` and `fma` from the output file for regression-based analysis. That is Members 2 and 3's task. The ICC split exists for exactly that purpose.

---

## Dataset 2: PADS Parkinson's Disease Dataset

**Notebook:** `pads_biomarker_extraction.ipynb`

### Dataset Description

469 subjects from a structured clinical assessment. Conditions: Healthy (79), Parkinson's Disease (276), Essential Tremor (28), Other Movement Disorders (60+). For this analysis, we train binary classifiers on Healthy vs. Parkinson's (355 subjects).

The dataset is publicly available at https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/. Data was collected using Apple Watch Series 4 worn on both wrists, recording at 100 Hz (3-axis accelerometer + 3-axis gyroscope).

11 standardized motor assessment tasks were performed: Relaxed, RelaxedTask, StretchHold, LiftHold, HoldWeight, PointFinger, DrinkGlas, CrossArms, TouchIndex, TouchNose, Entrainment. Two tasks (Relaxed, RelaxedTask, Entrainment) were 20.48 seconds and split into two halves. Total: up to 28 raw recording segments per subject.

Labels come from `pads_data/preprocessed/file_list.csv`, which contains all patient metadata in one place: condition string, numeric label (0-3), age, gender, handedness, weight, height. Subject IDs are integers 1 to 469, stored zero-padded to 3 digits in filenames.

Two data representations are used:
- **Raw timeseries:** `movement/timeseries/{id:03d}_{task}_{wrist}.txt` — 100 Hz, comma-separated, columns are Acc_X/Y/Z followed by Gyro_X/Y/Z
- **Preprocessed binary:** `preprocessed/movement/{id:03d}_ml.bin` — float32 binary, shape (488, 264), 264 channels covering all task-wrist-sensor-axis combinations

### The Bio-PM Challenge on PADS

The core challenge is that Bio-PM's movement element detector was designed for free-living continuous data. It detects discrete acceleration events by finding zero crossings in a bandpass-filtered signal. When applied to isolated 10-second clinical task segments, subjects are largely stationary — seated, holding a weight, pointing a finger. The detector finds no events. The acc stream outputs zeros.

Three approaches were tried before arriving at the current design:

**Isolated segments:** Run Bio-PM on each 10-second segment independently. Result: acc stream zeros for all 355 subjects. Only gravity stream contributes. AUC ceiling: 0.657.

**Per-task concatenated features:** Extract per-task mean embeddings and concatenate. Result: 12,721-dimensional feature space for 355 subjects. PCA cannot separate signal from noise at this ratio. AUC dropped to 0.598.

**Continuous stream:** Concatenate all task recordings per subject into one stream, resample from 100 Hz to 30 Hz, then slide 300-sample (10-second) windows across the full stream. This is the current approach. Movement between tasks, postural transitions, and natural arm repositioning provide more motion events than any single isolated task. AUC from gravity stream alone: 0.677. Still zero ME detection, but the gravity CNN extracts more meaningful postural representations from the varied continuous signal.

The honest conclusion: Bio-PM's acc stream does not activate on this data. The gravity stream is the signal carrier. It captures postural asymmetry and stability, which are clinically valid PD biomarkers — PD patients show reduced arm swing, postural instability, and asymmetric upper limb positioning. The gravity CNN encodes these, even without movement element detection.

### Bio-PM Extraction (Continuous Stream)

For each subject and each wrist independently:

1. Load all 11 task txt files in sequential order
2. Take only accelerometer channels (columns 0-2), discard gyroscope
3. Concatenate all segments into one continuous stream (typically ~3,500 samples at 100 Hz per wrist)
4. Resample 100 Hz to 30 Hz using `scipy.signal.resample`
5. Slide 300-sample windows with 150-sample (50%) overlap across the full stream
6. For each window: run the gravity stream in a batched GPU call, run the acc stream individually (ME detection is sequential by nature)
7. Mean-pool all window embeddings to one 1028-d vector per wrist
8. Average left and right wrist vectors to one 1028-d vector per subject

The gravity stream is batched per subject rather than per window. All windows for one subject are stacked into a single tensor and passed through the CNN in one forward pass. On Apple Silicon (MPS), this is approximately 4x faster than processing windows sequentially.

```python
# All windows for one subject stacked: shape (n_windows, 3, 300)
G = torch.stack(gravity_batch).to(DEVICE).permute(0, 2, 1)
G = F.interpolate(G, size=WIN_LEN, mode='linear', align_corners=False)
G_flat = G.reshape(len(valid_idxs), -1)   # (n_windows, 900)
```

Embeddings are cached to `features/pads_biopm_stream.npz` after the first run. Delete this file to re-run Bio-PM extraction.

### Kinematic Feature Extraction (from preprocessed .bin)

The preprocessed binary files contain all task recordings for one subject in a single matrix of shape (488, 264). The 264 columns cover all combinations of 11 tasks × 2 wrists × 12 channels, where the 12 channels are 6 raw IMU signals plus 6 L1-trend-filtered versions.

For each of the 264 channels, 11 features are computed:

- **9 time-domain features:** mean, std, min, max, range, RMS, IQR, skewness, zero-crossing rate
- **2 spectral features:** PD tremor band power (4-6 Hz normalized FFT power) and Essential Tremor band power (8-12 Hz normalized FFT power)

Total: 264 × 11 = 2,904 kinematic features per subject.

The spectral features are the most clinically motivated additions. PD resting tremor occurs at 4-6 Hz. Essential Tremor occurs at 8-12 Hz. These frequency bands are established in the clinical literature. Computing normalized band power directly from the FFT of each channel captures this information explicitly, which Bio-PM's acc stream was intended to capture implicitly but cannot on this data.

Kinematic features are cached to `features/pads_kinematic.npz`.

### Demographics

Five demographic features are appended to both the Bio-PM and kinematic feature matrices:
- Age (continuous)
- Gender (male = 0, female = 1)
- Handedness (right = 0, left = 1, both = 2)
- Weight (continuous, in kg)
- Height (continuous, in cm)

Age is the strongest individual predictor. PD onset peaks in the 60-65 age range. The PADS healthy controls average 56 years old; PD patients average 68. This age separation is real and clinically meaningful, not a confound being exploited. Handedness captures lateralization — PD is typically asymmetric in onset and handedness is correlated with which side is initially affected.

### Classification: Late Fusion LOSO

Two separate Leave-One-Subject-Out pipelines are trained independently.

**Kinematic pipeline:** SelectKBest (k=500, ANOVA F-score) to pre-filter 2,909 features, then PCA (100 components), then Logistic Regression (C=0.1, class_weight='balanced'). The feature selection step is necessary — going straight from 2,904 features to PCA(100) gives the PCA too much noise to filter. Selecting the 500 most discriminative features first gives the PCA cleaner input.

**Bio-PM pipeline:** PCA (80 components) directly on 1,033 features (1028 Bio-PM + 5 demographic), then Logistic Regression (C=0.1, class_weight='balanced'). No feature selection needed here because 1,033 dimensions is already tractable.

Both pipelines use `class_weight='balanced'` to correct for the 1:3.5 healthy:PD imbalance. Without this, the classifier learns to predict "impaired" for everything and achieves high accuracy at the cost of all healthy recall.

**Late fusion:** The two pipelines' probability outputs are combined as a weighted average, with weights proportional to each pipeline's individual AUC. This is the correct way to combine two classifiers of different quality. Concatenating features and running a single pipeline makes the weaker feature set drag down the stronger one through PCA noise. Late fusion lets each pipeline contribute according to what it actually demonstrated on this data.

```python
w_kin = auc_kin / (auc_kin + auc_bio)
w_bio = auc_bio / (auc_kin + auc_bio)
prob_fused = w_kin * prob_kin + w_bio * prob_bio
```

**AUC is computed on pooled predictions** across all LOSO folds, not averaged per fold. This is the only valid approach when each test fold contains a single subject with one label. Computing AUC per fold gives 0.5 for every fold by construction. The pool-then-score approach treats the full LOSO run as one prediction problem.

**Threshold selection:** The classification threshold is set at the point on the ROC curve that maximizes (TPR - FPR), not at the default 0.5. With class imbalance, 0.5 is not the right operating point.

### Classification Results

| Pipeline | AUC |
|---|---|
| Kinematic + spectral + demographics | 0.684 |
| Bio-PM continuous stream + demographics | 0.677 |
| Late fusion (weighted average) | **0.719** |

The fusion gain (+0.035 over kinematic alone) comes from the gravity stream encoding postural information that the kinematic .bin features do not capture. The two pipelines are genuinely complementary. The kinematic features describe what the accelerometer signal looks like statistically; the Bio-PM gravity stream encodes how the wrist is oriented relative to gravity across the full session.

### Output

`features/pads_features.npz`:

| Key | Shape | Description |
|---|---|---|
| `features` | (355, 3281) | Late fusion not saved here; this is the combined input matrix for reference |
| `labels` | (355,) | 0 = healthy, 1 = Parkinson's |
| `pids` | (355,) | Subject ID for LOSO grouping |
| `conditions` | (355,) | Original condition string |
| `feature_names` | (3281,) | Feature name per column |

Intermediate caches:
- `features/pads_biopm_stream.npz` — raw Bio-PM embeddings (355, 1028) plus subject IDs, labels, conditions, ME counts
- `features/pads_kinematic.npz` — kinematic features (355, 2904) plus subject IDs

---

## Known Issues and Design Decisions

**Bio-PM ME detection is zero on both datasets.** This is a real limitation of applying a pretrained free-living model to structured clinical assessments. The movement element detector relies on sustained dynamic motion. Both datasets — IRB (pre-filtered) and PADS (controlled tasks) — suppress this. The gravity stream carries all the Bio-PM signal in both cases.

**IRB binary AUC of 0.514 is not a bug.** With 4 healthy and 194 stroke visits, class imbalance makes binary classification meaningless. Downstream analysis on this dataset should use ARAT/FMA regression, not binary labels.

**PADS subject IDs must be zero-padded.** The file_list.csv stores IDs as integers (1, 2, 3...) but filenames use zero-padded strings (001, 002, 003...). Always use `str(id).zfill(3)` when building file paths.

**LOSO AUC must be computed on pooled predictions.** Computing per-fold AUC and averaging gives 0.5 for every fold by construction when test folds are single subjects. This was the source of incorrect 0.5 results early in development.

**Observation JSON channel names include 'Time'.** The timeseries JSON lists 'Time' as a channel. It has no underscore so `ch.rsplit('_', 1)` fails on it. Skip any channel where `'_' not in ch`.

---

## What This Produces for the Team

**For Member 2 (Biomarker Engineering, Random Forest):**
- `features/biopm_features.npz` with the full 228-d feature matrix, feature names, ARAT, FMA, and subject IDs
- `features/pads_features.npz` with the 3281-d combined feature matrix for PADS
- Both include `pids` for LOSO grouping

**For Member 3 (Validation, ICC, Clinical Correlation, Report):**
- `features_even` and `features_odd` in `biopm_features.npz` for ICC analysis on the IRB dataset
- `arat` and `fma` arrays for Spearman correlation analysis
- `weeks` for longitudinal analysis of recovery trajectories
- UMAP figures in `results/figures/` for both datasets
- LOSO metrics CSVs in `results/metrics/` for both datasets
