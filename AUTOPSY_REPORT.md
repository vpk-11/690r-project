# Bio-PM IRB Pipeline: Full Autopsy Report
## 0. TL;DR (Read This First)
- MPS availability check: `False` (executed on 2026-05-04). Analysis ran on `cpu`.
- Selected test subjects: `[539, 533, 551]` from existing HDF5 windows (30 windows/subject).
- `masked_mean_std` is not masked: it computes `x.mean(dim=1)` and `x.std(dim=1)` across all tokens, including NaN-padding replacement tokens.
- Fill-rate issue is structural: typical ME count stays near constant while `pad_size` grows, so token dilution reduces transformer variance signal.
- Gravity stream in extractor is not `encoder_gravity`; it is flattened/interpolated raw `x_gravity` (900-d), so quality depends entirely on preprocessing signal.
- Best tested fix direction: keep `pad_size=57`, group 3 windows (9s), and use gravity input variant with non-vanishing low-frequency content.

## 1. Repository and File Map
- Model source read: `CS690TR/src/models/biopm.py`
- Preprocessing source read: `CS690TR/src/data/preprocessing.py`

### Pipeline scripts
- `debug_pipeline.py`: debug_pipeline.py -- Step-by-step diagnostic to find why embeddings are all zeros.. Suspicious constants: none
- `export_legacy_schema.py`: export_legacy_schema.py -- Build old-style visit-level schema from BIOPM window-level output.. Suspicious constants: none
- `inspect_dataset.py`: inspect_dataset.py -- Characterize clinical_scores.npz and windows.npz.. Suspicious constants: pad_size  = {int(win_sec * 192 / 10)}   (int({win_sec:.0f} * 192 / 10))"), WS        = {win_sec:.0f}   (window size in seconds)")
- `irb_analyze.py`: irb_analyze.py -- Even/odd split, UMAP visualization, LOSO LR baseline.. Suspicious constants: none
- `irb_extract.py`: irb_extract.py -- Extract Bio-PM embeddings from preprocessed IRB HDF5 files.. Suspicious constants: TARGET_GRAV_LEN = 300
- `irb_preprocess.py`: irb_preprocess.py -- Preprocess IRB stroke .npz dataset for Bio-PM.. Suspicious constants: pad_size={CONFIG['pad_size']}  30 Hz"), WS={CONFIG['WS']}s  pad_size={CONFIG['pad_size']}  30 Hz")
- `irb_preprocess_alt.py`: irb_preprocess_alt.py -- Alt preprocessing: groups 3 consecutive windows into. Suspicious constants: pad_size = group_size * 57, TARGET_GRAV_LEN=300 regardless of T., GROUP_SIZE = 3
- `verify_embeddings.py`: verify_embeddings.py -- Sanity check Bio-PM features.. Suspicious constants: none

## 2. Bio-PM Architecture: How It Actually Works
- `BioPMModel` contains `encoder_acc` (transformer), `encoder_gravity` (CNN 64-d), and `classifier` head.
- Acc transformer input: patches `(B,L,32)`, positions `(B,L)`, metadata `(B,L,>=2)`; output tokens `(B,L,64)` through 5 relative-position encoder layers.
- `masked_mean_std` in current code does **not** use mask arg; it concatenates unmasked sequence mean+std.
- Gravity CNN is defined as `Conv1d(3->16)->GN->GELU->Dropout2d->Conv1d(16->32,stride2)->...->Conv1d(32->64,stride2)->AvgMaxPool1d(K=12)->Linear(1536->64)`.
- `load_pretrained_encoder` loads checkpoint only into `encoder_acc`; missing/unexpected keys are reported. Current run: missing=0 unexpected=26.
- Model does not call `lowpass_filter` or `bandpass_filter` internally; filtering is external in preprocessing scripts.

## 3. The Two Streams: Signal Flow
### 3a. Acc Transformer Stream [dims 0-127]
- `bandpass_filter`: Butterworth bandpass via `butter` + `filtfilt`.
- `detect_zero_crossings` returns: `(resampled_vel,time_index,me_list,me_normalize_list,me_normalizeInfo_list,me_normalize_padding,me_normalizeInfo_padding,pos_info,zero_crossings_list,zero_crossings_time_list)`.
- `me_info` columns are `axis,start_point,end_point,len,min,max,dirct,peaks`.
- `assign_zero_crossings` reuses crossing boundaries on another signal and offsets axis by `+2` in metadata; returns no `dirct` column.
- Packed `x_acc_filt` in `irb_preprocess.py` is `[0:32]=me_norm,[32]=pos_info,[33:38]=axis,len,min,max,dirct`, matching your spec.
### 3b. Gravity CNN Stream [dims 128-1027]
- Preprocessing writes `x_gravity = lowpass_filter(raw_acc, 0.5 Hz)` by default.
- In `irb_extract.py`, gravity feature is **not** the model CNN output; it is `x_gravity -> transpose -> interpolate(T->300) -> flatten` => 900 dims.

## 4. The Gravity Stream Problem

### Is gravity present in the data?
|subj|group|win_count|ARAT|FMA|sample_rate|shape|mean_gravity_mag_g|
|---|---:|---:|---:|---:|---:|---|---:|
|539|stroke|83179|53|64|30|(90, 3)|0.0271|
|533|stroke|47607|51|57|30|(90, 3)|0.0261|
|551|healthy|43886|57|66|30|(90, 3)|0.0258|

### Lowpass filter behavior
- Subject 539: raw_abs=0.15620; 0.1Hz mean_abs=0.42512 max_abs=0.53402; 0.5Hz mean_abs=0.07052 max_abs=0.40474; 1.0Hz mean_abs=0.12987 max_abs=0.45218; 2.0Hz mean_abs=0.14634 max_abs=0.49480; 5.0Hz mean_abs=0.15614 max_abs=0.51434
- Subject 533: raw_abs=0.14729; 0.1Hz mean_abs=0.51315 max_abs=1.00131; 0.5Hz mean_abs=0.08002 max_abs=0.30793; 1.0Hz mean_abs=0.10548 max_abs=0.38892; 2.0Hz mean_abs=0.14072 max_abs=0.88914; 5.0Hz mean_abs=0.14731 max_abs=0.92139
- Subject 551: raw_abs=0.14287; 0.1Hz mean_abs=0.07472 max_abs=0.12505; 0.5Hz mean_abs=0.10257 max_abs=0.26657; 1.0Hz mean_abs=0.13071 max_abs=0.52693; 2.0Hz mean_abs=0.13966 max_abs=0.96096; 5.0Hz mean_abs=0.14288 max_abs=1.09095
- If these lowpass outputs are tiny, gravity was likely removed upstream or centered per window.

## 5. The pad_size Problem

### Does pad_size affect ME detection?
- Detection logic uses `pad_size` only for truncation/padding after MEs are extracted; it does not change zero-crossing discovery itself.

### Fill rates by pad_size
|subj|pad_size|avg_MEs|fill_rate_pct|zero_windows_pct|
|---:|---:|---:|---:|---:|
|539|57|15.63|27.43|0.00|
|539|96|15.63|16.28|0.00|
|539|171|15.63|9.14|0.00|
|539|192|15.63|8.14|0.00|
|533|57|17.37|30.47|0.00|
|533|96|17.37|18.09|0.00|
|533|171|17.37|10.16|0.00|
|533|192|17.37|9.05|0.00|
|551|57|18.50|32.46|0.00|
|551|96|18.50|19.27|0.00|
|551|171|18.50|10.82|0.00|
|551|192|18.50|9.64|0.00|

### Transformer pooling by pad_size
|pad_size|mean_pool_abs|std_pool_abs|
|---:|---:|---:|
|57|0.70649|0.41789|
|96|0.73480|0.34335|
|171|0.75922|0.26680|
|192|0.76303|0.25302|
- Since `masked_mean_std` ignores masks, larger `pad_size` injects more padded-token embeddings into pooled stats.

## 6. Filter Permutation Results

### Acc stream filters
|filter_config|avg_MEs|fill_57_pct|zero_windows_pct|mean_pool_abs|std_pool_abs|
|---|---:|---:|---:|---:|---:|
|std|17.17|30.12|0.00|0.70595|0.42122|
|wide|0.00|0.00|100.00|nan|nan|
|pd|3.87|6.78|0.00|0.78120|0.13464|
|high|16.98|29.79|0.00|0.70683|0.41820|
|raw|11.58|20.31|0.00|0.72205|0.36897|
|std_rms|18.38|32.24|0.00|0.63922|0.51218|

### Gravity stream inputs
|gravity_input|grav_stream_abs|embedding_norm|
|---|---:|---:|
|lowpass05|0.06632|2.55902|
|raw|0.10675|4.41357|
|mean_const|0.01350|0.49044|
|demean_lowpass|0.06599|2.53245|
|highpass_removed|0.08133|3.46283|

## 7. The 100% Fill Rate Fix
|subj|avg_MEs_grouped_9s|zero_windows_pct|fill_pct_trunc57|mean_pool_abs|std_pool_abs|
|---:|---:|---:|---:|---:|---:|
|539|51.40|0.00|90.18|0.67041|0.50516|
|533|53.70|0.00|94.21|0.68325|0.47995|
|551|54.30|0.00|95.26|0.67728|0.47615|

## 8. Existing File Quality Check
- `preprocessed/`: files=198, sampled_fill=32.34%, x_gravity_range=[-1.76851, 1.56228]
- `preprocessed_alt/`: files=198, sampled_fill=35.51%, x_gravity_range=[-1.56269, 1.24300]
- `features/biopm_features.npz` shape=(587046, 1028), mean_abs=0.70913, std_abs=0.41857, grav_abs=0.05521, zero_rows=0, grav_per_subject=[0.02586,0.07098]
- `features/biopm_features_alt.npz` shape=(195610, 1028), mean_abs=0.69872, std_abs=0.43599, grav_abs=0.05581, zero_rows=0, grav_per_subject=[0.02456,0.07243]

## 9. What the TA Means by "Movements Should Be Detected"
- TA statement is correct: movement elements are being detected at non-trivial rates. Primary failures are padding dilution, unmasked pooling, and weak gravity preprocessing/extraction coupling, not total ME absence.

## Appendix: Step 4a return-value audit
- resampled_vel: ndarray shape=(90, 3) dtype=float64 range=[-0.45666,0.29548]
- time_index: ndarray shape=(90,) dtype=float64 range=[0.00000,2.96667]
- me_list: type=list len=21
- me_norm: ndarray shape=(21, 32) dtype=float64 range=[-0.45717,0.29817]
- me_info: type=DataFrame shape=(21, 8)
- me_norm_pad: ndarray shape=(57, 32) dtype=float64 range=[-100.00000,0.29817]
- me_info_pad: type=DataFrame shape=(57, 8)
- pos_info: ndarray shape=(21,) dtype=float64 range=[0.03333,0.91111]
- zc_list: type=list len=3
- zc_time_list: type=list len=3
- assign_zero_crossings pos_info len=21, me_info columns=['axis', 'start_point', 'end_point', 'len', 'min', 'max', 'peaks']
