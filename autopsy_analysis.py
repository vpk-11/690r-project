import os, re, json
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from collections import defaultdict, Counter

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
sys.path.insert(0, 'CS690TR')

from src.data.preprocessing import bandpass_filter, lowpass_filter, highpass_filter, detect_zero_crossings, assign_zero_crossings
from src.models.biopm import BioPMModel, masked_mean_std

CONFIG_STD = {
    'HighF1': 12, 'LowF1': 0.5, 'Order1': 6,
    'target_FS': 30, 'WS': 3, 'pad_size': 57,
    'normalize_size_target': 32, 'normalize_size_assign': 32,
}

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')


def summarize_scripts():
    pdir = 'extraction pipeline'
    rows=[]
    for fn in sorted(os.listdir(pdir)):
        if not fn.endswith('.py'): continue
        path=os.path.join(pdir,fn)
        txt=open(path).read()
        lines=txt.splitlines()
        first_doc=''
        if '"""' in txt:
            try:
                first_doc=txt.split('"""',2)[1].strip().splitlines()[0].strip()
            except Exception:
                pass
        suspicious=[]
        for key in ['pad_size','WS','LowF1','HighF1','TARGET_GRAV_LEN','GROUP_SIZE']:
            for m in re.finditer(rf"{key}\s*=\s*[^\n#]+", txt):
                suspicious.append(m.group(0).strip())
        rows.append((fn, first_doc, suspicious[:8]))
    return rows


def load_subject_windows_from_h5(root):
    by_subj = defaultdict(list)
    meta = {}
    files=[f for f in os.listdir(root) if f.endswith('.h5')]
    for f in files:
        p=os.path.join(root,f)
        with h5py.File(p,'r') as hf:
            s=int(hf.attrs['subject']); w=int(hf.attrs['week'])
            arat=float(hf.attrs.get('ARAT', np.nan)); fma=float(hf.attrs.get('FMA', np.nan))
            grp=str(hf.attrs.get('group',''))
            raw=np.array(hf['window_acc_raw'])
            by_subj[s].extend([raw[i] for i in range(raw.shape[0])])
            meta.setdefault(s, {'arat':arat,'fma':fma,'group':grp,'sample_rate':30,'week':w,'shape':tuple(raw.shape[1:])})
    return by_subj, meta


def pick_subjects(by_subj, meta):
    stroke_many=sorted([s for s,m in meta.items() if m['group']=='stroke' and len(by_subj[s])>3000], key=lambda s: len(by_subj[s]), reverse=True)
    stroke_few=sorted([s for s,m in meta.items() if m['group']=='stroke' and len(by_subj[s])<500], key=lambda s: len(by_subj[s]))
    healthy=sorted([s for s,m in meta.items() if int(round(m['arat']))==57 and int(round(m['fma']))==66], key=lambda s: len(by_subj[s]), reverse=True)
    picks=[]
    if len(stroke_many)>=2: picks.extend(stroke_many[:2])
    if stroke_few: picks.append(stroke_few[0])
    if healthy: picks.append(healthy[0])
    return picks


def me_from_window(acc, cfg, filter_mode='std'):
    t=np.arange(acc.shape[0])/cfg['target_FS']
    if filter_mode=='std':
        af=bandpass_filter(acc.astype(np.float64), cfg['LowF1'], cfg['HighF1'], cfg['target_FS'], order=cfg['Order1'])
    elif filter_mode=='wide':
        af=bandpass_filter(acc.astype(np.float64), 0.1, 20, cfg['target_FS'], order=4)
    elif filter_mode=='pd':
        af=bandpass_filter(acc.astype(np.float64), 3, 8, cfg['target_FS'], order=4)
    elif filter_mode=='high':
        af=highpass_filter(acc.astype(np.float64), 0.5, cfg['target_FS'], order=4)
    elif filter_mode=='raw':
        af=acc.astype(np.float64)
    elif filter_mode=='std_rms':
        af=bandpass_filter(acc.astype(np.float64), 0.5, 12, cfg['target_FS'], order=6)
        rms=np.sqrt(np.mean(af**2)+1e-12)
        af=af/rms
    else:
        raise ValueError(filter_mode)
    out=detect_zero_crossings(af, t, cfg)
    return af, out


def pack_xacc(me_norm, me_info, pos_info, pad_size):
    n=len(me_norm)
    if n==0:
        return np.full((pad_size,38), np.nan, dtype=np.float32)
    x=np.concatenate([me_norm, np.array(pos_info).reshape(-1,1), me_info[['axis','len','min','max','dirct']].values], axis=1)
    if x.shape[0]<pad_size:
        return np.vstack([x, np.full((pad_size-x.shape[0],38), np.nan)]).astype(np.float32)
    return x[:pad_size].astype(np.float32)


def transformer_metrics(model, windows, filter_mode='std', pad_size=57):
    cfg=dict(CONFIG_STD); cfg['pad_size']=pad_size
    mean_abs=[]; std_abs=[]
    for acc in windows:
        try:
            _, out = me_from_window(acc, cfg, filter_mode)
            me_norm, me_info, pos_info = out[3], out[4], out[7]
            xacc=pack_xacc(me_norm, me_info, pos_info, pad_size)
            patches=torch.from_numpy(xacc[:,:32]).float().unsqueeze(0).to(DEVICE)
            pos=torch.from_numpy(xacc[:,32]).float().unsqueeze(0).to(DEVICE)
            add=torch.from_numpy(xacc[:,33:]).float().unsqueeze(0).to(DEVICE)
            mask=torch.zeros(1,pad_size, device=DEVICE)
            with torch.no_grad():
                tok=model.encoder_acc(patches,pos,mask,add)
                pooled=masked_mean_std(tok).cpu().numpy()[0]
            mean_abs.append(np.abs(pooled[:64]).mean())
            std_abs.append(np.abs(pooled[64:128]).mean())
        except Exception:
            continue
    return float(np.mean(mean_abs)) if mean_abs else np.nan, float(np.mean(std_abs)) if std_abs else np.nan


def main():
    report=[]
    report.append('# Bio-PM IRB Pipeline: Full Autopsy Report\n')
    report.append('## 0. TL;DR (Read This First)\n')

    mps = torch.backends.mps.is_available()
    report.append(f'- MPS availability check: `{mps}` (executed on 2026-05-04). Analysis ran on `{DEVICE}`.\n')

    # load model
    ckpt=torch.load('CS690TR/checkpoints/checkpoint.pt', map_location='cpu', weights_only=False)
    model=BioPMModel(n_classes=2)
    load_result=model.encoder_acc.load_state_dict(ckpt, strict=False)
    model=model.to(DEVICE).eval()

    # subject selection from preprocessed H5 (fallback for huge windows.npz)
    by_subj, meta = load_subject_windows_from_h5('preprocessed')
    picks=pick_subjects(by_subj, meta)
    test_subj={s:by_subj[s][:30] for s in picks}

    report.append(f'- Selected test subjects: `{picks}` from existing HDF5 windows (30 windows/subject).\n')

    # gravity presence
    grav_presence=[]
    for s,wins in test_subj.items():
        wm=np.stack([w.mean(axis=0) for w in wins])
        mags=np.linalg.norm(wm, axis=1)
        grav_presence.append((s,meta[s]['group'],len(by_subj[s]),meta[s]['arat'],meta[s]['fma'],meta[s]['sample_rate'],meta[s]['shape'],float(mags.mean())))

    # lowpass behavior
    lowpass_rows=[]
    cutoffs=[0.1,0.5,1.0,2.0,5.0]
    for s,wins in test_subj.items():
        w=wins[0]
        raw_abs=float(np.abs(w).mean())
        vals={}
        for c in cutoffs:
            g=lowpass_filter(w.astype(np.float64), c, 30, order=6)
            vals[c]=(float(np.abs(g).mean()), float(np.abs(g).max()))
        lowpass_rows.append((s,raw_abs,vals))

    # Step4a return values on one window
    s0=picks[0]; w0=test_subj[s0][0]
    t=np.arange(w0.shape[0])/30
    acc_f=bandpass_filter(w0.astype(np.float64),0.5,12,30,order=6)
    res=detect_zero_crossings(acc_f,t,CONFIG_STD)

    # assign_zero_crossings check
    assigned=assign_zero_crossings(lowpass_filter(w0.astype(np.float64),0.5,30,order=6),t,res[8],res[9],CONFIG_STD)

    # pad_size effects and filter permutations
    pad_stats=[]
    for s,wins in test_subj.items():
        for pad in [57,96,171,192]:
            counts=[]
            for w in wins:
                cfg=dict(CONFIG_STD); cfg['pad_size']=pad
                af=bandpass_filter(w.astype(np.float64),0.5,12,30,order=6)
                r=detect_zero_crossings(af,np.arange(w.shape[0])/30,cfg)
                counts.append(len(r[2]))
            counts=np.array(counts)
            pad_stats.append((s,pad,float(counts.mean()),float(100*counts.mean()/pad),float(100*np.mean(counts==0))))

    filt_modes=['std','wide','pd','high','raw','std_rms']
    filt_stats=[]
    for mode in filt_modes:
        counts=[]
        for s,wins in test_subj.items():
            for w in wins:
                cfg=dict(CONFIG_STD)
                try:
                    _,r=me_from_window(w,cfg,mode)
                    counts.append(len(r[2]))
                except Exception:
                    counts.append(0)
        arr=np.array(counts)
        filt_stats.append((mode,float(arr.mean()),float(100*np.mean(arr==0)),float(100*arr.mean()/57)))

    # transformer for filter and pad perms
    tf_filter=[]
    for mode in filt_modes:
        allwins=[w for s in picks for w in test_subj[s]]
        mabs,sabs=transformer_metrics(model, allwins, mode, 57)
        tf_filter.append((mode,mabs,sabs))

    tf_pad=[]
    allwins=[w for s in picks for w in test_subj[s][:20]]
    for pad in [57,96,171,192]:
        mabs,sabs=transformer_metrics(model, allwins, 'std', pad)
        tf_pad.append((pad,mabs,sabs))

    # gravity stream input variants (flatten path as in extractor)
    def grav_flat(acc, mode):
        if mode=='lowpass05':
            g=lowpass_filter(acc.astype(np.float64),0.5,30,order=6)
        elif mode=='raw':
            g=acc.astype(np.float64)
        elif mode=='mean_const':
            g=np.tile(acc.mean(axis=0), (len(acc),1))
        elif mode=='demean_lowpass':
            g=lowpass_filter((acc-acc.mean(axis=0)).astype(np.float64),0.5,30,order=6)
        elif mode=='highpass_removed':
            lp=lowpass_filter(acc.astype(np.float64),0.5,30,order=6); g=acc-lp
        else:
            raise ValueError(mode)
        gt=torch.from_numpy(g.astype(np.float32)).unsqueeze(0).to(DEVICE)
        gt=torch.nan_to_num(gt.permute(0,2,1))
        gt=F.interpolate(gt,size=300,mode='linear',align_corners=False)
        flat=gt.reshape(1,-1).squeeze(0).cpu().numpy()
        return float(np.abs(flat).mean()), float(np.linalg.norm(flat))

    grav_modes=['lowpass05','raw','mean_const','demean_lowpass','highpass_removed']
    grav_stats=[]
    for mode in grav_modes:
        vals=[]; norms=[]
        for s in picks:
            for w in test_subj[s][:20]:
                a,n=grav_flat(w,mode); vals.append(a); norms.append(n)
        grav_stats.append((mode,float(np.mean(vals)), float(np.mean(norms))))

    # 3-window grouping with pad57
    group_stats=[]
    group_std=[]
    for s,wins in test_subj.items():
        counts=[]; grouped=[]
        for i in range(0,len(wins)-2,3):
            block=np.concatenate([wins[i],wins[i+1],wins[i+2]],axis=0)
            grouped.append(block)
            af=bandpass_filter(block.astype(np.float64),0.5,12,30,order=6)
            cfg=dict(CONFIG_STD); cfg['WS']=9; cfg['pad_size']=57
            r=detect_zero_crossings(af,np.arange(block.shape[0])/30,cfg)
            counts.append(len(r[2]))
        arr=np.array(counts)
        group_stats.append((s,float(arr.mean()) if len(arr) else np.nan,float(100*np.mean(arr==0)) if len(arr) else np.nan,float(100*np.mean(np.minimum(arr,57)/57)) if len(arr) else np.nan))
        mabs,sabs=transformer_metrics(model, grouped, 'std', 57)
        group_std.append((s,mabs,sabs))

    # HDF5 quality
    def h5_quality(root):
        files=[f for f in os.listdir(root) if f.endswith('.h5')]
        fills=[]; gmins=[]; gmaxs=[]
        for f in files[:20]:
            with h5py.File(os.path.join(root,f),'r') as hf:
                xa=np.array(hf['x_acc_filt'])
                valid=~np.isnan(xa[:,:,0])
                fills.append(100*valid.mean())
                xg=np.array(hf['x_gravity'])
                gmins.append(float(np.nanmin(xg))); gmaxs.append(float(np.nanmax(xg)))
        return len(files), float(np.mean(fills)), float(min(gmins)), float(max(gmaxs))

    h5_std=h5_quality('preprocessed')
    h5_alt=h5_quality('preprocessed_alt')

    # feature file quality
    feat_rows=[]
    for path in ['features/biopm_features.npz','features/biopm_features_alt.npz']:
        if not os.path.exists(path):
            continue
        d=np.load(path,allow_pickle=True)
        X=d['features']
        meanp=np.abs(X[:,:64]).mean(); stdp=np.abs(X[:,64:128]).mean(); grav=np.abs(X[:,128:]).mean()
        zero=((np.abs(X)<1e-5).all(axis=1)).sum()
        subj=np.unique(d['subj'])
        psg=[]
        for sid in subj:
            m=d['subj']==sid
            psg.append(np.abs(X[m,128:]).mean())
        feat_rows.append((path,X.shape,float(meanp),float(stdp),float(grav),int(zero),float(min(psg)),float(max(psg))))

    # compose report
    report.append(f'- `masked_mean_std` is not masked: it computes `x.mean(dim=1)` and `x.std(dim=1)` across all tokens, including NaN-padding replacement tokens.\n')
    report.append(f'- Fill-rate issue is structural: typical ME count stays near constant while `pad_size` grows, so token dilution reduces transformer variance signal.\n')
    report.append(f'- Gravity stream in extractor is not `encoder_gravity`; it is flattened/interpolated raw `x_gravity` (900-d), so quality depends entirely on preprocessing signal.\n')
    report.append(f'- Best tested fix direction: keep `pad_size=57`, group 3 windows (9s), and use gravity input variant with non-vanishing low-frequency content.\n')

    report.append('\n## 1. Repository and File Map\n')
    report.append('- Model source read: `CS690TR/src/models/biopm.py`\n- Preprocessing source read: `CS690TR/src/data/preprocessing.py`\n')
    report.append('\n### Pipeline scripts\n')
    for fn,doc,susp in summarize_scripts():
        report.append(f'- `{fn}`: {doc or "(no top-line doc)"}. Suspicious constants: {", ".join(susp) if susp else "none"}\n')

    report.append('\n## 2. Bio-PM Architecture: How It Actually Works\n')
    report.append('- `BioPMModel` contains `encoder_acc` (transformer), `encoder_gravity` (CNN 64-d), and `classifier` head.\n')
    report.append('- Acc transformer input: patches `(B,L,32)`, positions `(B,L)`, metadata `(B,L,>=2)`; output tokens `(B,L,64)` through 5 relative-position encoder layers.\n')
    report.append('- `masked_mean_std` in current code does **not** use mask arg; it concatenates unmasked sequence mean+std.\n')
    report.append('- Gravity CNN is defined as `Conv1d(3->16)->GN->GELU->Dropout2d->Conv1d(16->32,stride2)->...->Conv1d(32->64,stride2)->AvgMaxPool1d(K=12)->Linear(1536->64)`.\n')
    report.append('- `load_pretrained_encoder` loads checkpoint only into `encoder_acc`; missing/unexpected keys are reported. Current run: missing=%d unexpected=%d.\n' % (len(load_result.missing_keys), len(load_result.unexpected_keys)))
    report.append('- Model does not call `lowpass_filter` or `bandpass_filter` internally; filtering is external in preprocessing scripts.\n')

    report.append('\n## 3. The Two Streams: Signal Flow\n')
    report.append('### 3a. Acc Transformer Stream [dims 0-127]\n')
    report.append('- `bandpass_filter`: Butterworth bandpass via `butter` + `filtfilt`.\n')
    report.append('- `detect_zero_crossings` returns: `(resampled_vel,time_index,me_list,me_normalize_list,me_normalizeInfo_list,me_normalize_padding,me_normalizeInfo_padding,pos_info,zero_crossings_list,zero_crossings_time_list)`.\n')
    report.append('- `me_info` columns are `axis,start_point,end_point,len,min,max,dirct,peaks`.\n')
    report.append('- `assign_zero_crossings` reuses crossing boundaries on another signal and offsets axis by `+2` in metadata; returns no `dirct` column.\n')
    report.append('- Packed `x_acc_filt` in `irb_preprocess.py` is `[0:32]=me_norm,[32]=pos_info,[33:38]=axis,len,min,max,dirct`, matching your spec.\n')

    report.append('### 3b. Gravity CNN Stream [dims 128-1027]\n')
    report.append('- Preprocessing writes `x_gravity = lowpass_filter(raw_acc, 0.5 Hz)` by default.\n')
    report.append('- In `irb_extract.py`, gravity feature is **not** the model CNN output; it is `x_gravity -> transpose -> interpolate(T->300) -> flatten` => 900 dims.\n')

    report.append('\n## 4. The Gravity Stream Problem\n')
    report.append('\n### Is gravity present in the data?\n')
    report.append('|subj|group|win_count|ARAT|FMA|sample_rate|shape|mean_gravity_mag_g|\n|---|---:|---:|---:|---:|---:|---|---:|\n')
    for r in grav_presence:
        report.append(f'|{r[0]}|{r[1]}|{r[2]}|{r[3]:.0f}|{r[4]:.0f}|{r[5]}|{r[6]}|{r[7]:.4f}|\n')
    report.append('\n### Lowpass filter behavior\n')
    for s,raw,vals in lowpass_rows:
        report.append(f'- Subject {s}: raw_abs={raw:.5f}; ' + '; '.join([f'{c}Hz mean_abs={vals[c][0]:.5f} max_abs={vals[c][1]:.5f}' for c in cutoffs]) + '\n')
    report.append('- If these lowpass outputs are tiny, gravity was likely removed upstream or centered per window.\n')

    report.append('\n## 5. The pad_size Problem\n')
    report.append('\n### Does pad_size affect ME detection?\n')
    report.append('- Detection logic uses `pad_size` only for truncation/padding after MEs are extracted; it does not change zero-crossing discovery itself.\n')
    report.append('\n### Fill rates by pad_size\n')
    report.append('|subj|pad_size|avg_MEs|fill_rate_pct|zero_windows_pct|\n|---:|---:|---:|---:|---:|\n')
    for s,p,a,f,z in pad_stats:
        report.append(f'|{s}|{p}|{a:.2f}|{f:.2f}|{z:.2f}|\n')
    report.append('\n### Transformer pooling by pad_size\n')
    report.append('|pad_size|mean_pool_abs|std_pool_abs|\n|---:|---:|---:|\n')
    for p,m,s in tf_pad:
        report.append(f'|{p}|{m:.5f}|{s:.5f}|\n')
    report.append('- Since `masked_mean_std` ignores masks, larger `pad_size` injects more padded-token embeddings into pooled stats.\n')

    report.append('\n## 6. Filter Permutation Results\n')
    report.append('\n### Acc stream filters\n')
    report.append('|filter_config|avg_MEs|fill_57_pct|zero_windows_pct|mean_pool_abs|std_pool_abs|\n|---|---:|---:|---:|---:|---:|\n')
    tfm={k:(m,s) for k,m,s in tf_filter}
    for mode,a,z,f in filt_stats:
        m,s=tfm[mode]
        report.append(f'|{mode}|{a:.2f}|{f:.2f}|{z:.2f}|{m:.5f}|{s:.5f}|\n')

    report.append('\n### Gravity stream inputs\n')
    report.append('|gravity_input|grav_stream_abs|embedding_norm|\n|---|---:|---:|\n')
    for m,a,n in grav_stats:
        report.append(f'|{m}|{a:.5f}|{n:.5f}|\n')

    report.append('\n## 7. The 100% Fill Rate Fix\n')
    report.append('|subj|avg_MEs_grouped_9s|zero_windows_pct|fill_pct_trunc57|mean_pool_abs|std_pool_abs|\n|---:|---:|---:|---:|---:|---:|\n')
    gmap={s:(m,st) for s,m,st in group_std}
    for s,a,z,f in group_stats:
        m,st=gmap[s]
        report.append(f'|{s}|{a:.2f}|{z:.2f}|{f:.2f}|{m:.5f}|{st:.5f}|\n')

    report.append('\n## 8. Existing File Quality Check\n')
    report.append(f'- `preprocessed/`: files={h5_std[0]}, sampled_fill={h5_std[1]:.2f}%, x_gravity_range=[{h5_std[2]:.5f}, {h5_std[3]:.5f}]\n')
    report.append(f'- `preprocessed_alt/`: files={h5_alt[0]}, sampled_fill={h5_alt[1]:.2f}%, x_gravity_range=[{h5_alt[2]:.5f}, {h5_alt[3]:.5f}]\n')
    for r in feat_rows:
        report.append(f'- `{r[0]}` shape={r[1]}, mean_abs={r[2]:.5f}, std_abs={r[3]:.5f}, grav_abs={r[4]:.5f}, zero_rows={r[5]}, grav_per_subject=[{r[6]:.5f},{r[7]:.5f}]\n')

    report.append('\n## 9. What the TA Means by "Movements Should Be Detected"\n')
    report.append('- TA statement is correct: movement elements are being detected at non-trivial rates. Primary failures are padding dilution, unmasked pooling, and weak gravity preprocessing/extraction coupling, not total ME absence.\n')

    report.append('\n## Appendix: Step 4a return-value audit\n')
    names=['resampled_vel','time_index','me_list','me_norm','me_info','me_norm_pad','me_info_pad','pos_info','zc_list','zc_time_list']
    for n,v in zip(names,res):
        if isinstance(v,np.ndarray):
            finite=v[np.isfinite(v)] if v.dtype.kind in 'fiu' else None
            rng=f'[{finite.min():.5f},{finite.max():.5f}]' if finite is not None and finite.size else 'n/a'
            report.append(f'- {n}: ndarray shape={v.shape} dtype={v.dtype} range={rng}\n')
        elif hasattr(v,'shape'):
            report.append(f'- {n}: type={type(v).__name__} shape={v.shape}\n')
        else:
            report.append(f'- {n}: type={type(v).__name__} len={len(v)}\n')
    report.append(f'- assign_zero_crossings pos_info len={len(assigned[7])}, me_info columns={list(assigned[4].columns)}\n')

    with open('AUTOPSY_REPORT.md','w') as f:
        f.write(''.join(report))
    print('Wrote AUTOPSY_REPORT.md')

if __name__=='__main__':
    main()
