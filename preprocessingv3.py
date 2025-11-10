#!/usr/bin/env python3
"""
Full preprocessing pipeline with all visual diagnostics:
- Raw vs LPF (X/Y/Z)
- Gravity separation (X/Y/Z)
- Zoomed 5 s LPF (X/Y/Z)
- Dynamic XYZ
- Combined LPF XYZ
- Overlapping window segmentation
- Frequency-domain comparison (2.5s vs 5s)
- FFT spectrum per axis
- Dynamic acceleration histogram
- Axis correlation matrix
- Feature trends over time

Example usage:
& "E:/Program Files/Python/python.exe" "E:/documents/Thesis code/preprocessingv3.py" `
  --in "E:\documents\Thesis code\clean_&_filtered\back\LOG011_clean.csv" `
  --out "E:/documents/Thesis code/features/backstroke_features4.csv" `
  --scaler "E:/documents/Thesis code/scalers/backstroke_scaler4.json" `
  --plots "E:/documents/Thesis code/plots/backstroke4" `
  --fs 100 --units g --win_s 5 --hop_s 2.5 --label backstroke
"""

import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt, windows
import seaborn as sns

# ------------------------ column handling ------------------------
def normalize_and_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
    df.columns = [re.sub(r'\s+', '_', str(c).strip().lower()) for c in df.columns]
    def pick(cands): return next((c for c in cands if c in df.columns), None)
    m_time = pick(["timestamp_ms","time_ms","timestamp","time"])
    m_ax   = pick(["ax_g","x_acc","acc_x","ax","x"])
    m_ay   = pick(["ay_g","y_acc","acc_y","ay","y"])
    m_az   = pick(["az_g","z_acc","acc_z","az","z"])
    for need,var in zip(["timestamp_ms","ax_g","ay_g","az_g"],[m_time,m_ax,m_ay,m_az]):
        if var is None: raise ValueError(f"Missing {need}; got {list(df.columns)}")
    df = df.rename(columns={m_time:"timestamp_ms",m_ax:"ax_g",m_ay:"ay_g",m_az:"az_g"})
    return df

# ---------------------------- filters -----------------------------
def butter_lpf(x, fc, fs, order=4):
    b,a = butter(order, fc/(0.5*fs), btype="low")
    return filtfilt(b,a,x)

def kalman_1d(z,q=1e-5,r=1e-3):
    xhat=np.zeros_like(z,dtype=float);P=1;x=float(z[0])
    for i,zi in enumerate(z):
        xpred=x;P=P+q
        K=P/(P+r);x=xpred+K*(zi-xpred);P=(1-K)*P;xhat[i]=x
    return xhat

# ---------------------- feature calculations ----------------------
def time_features(x):
    x=np.asarray(x);m=np.mean(x);s=np.std(x)
    return dict(mean=m,std=s,var=np.var(x),
                skew=((x-m)**3).mean()/(s**3+1e-12),
                kurt=((x-m)**4).mean()/(s**4+1e-12),
                p2p=np.ptp(x),sma=np.mean(np.abs(x)),
                energy=np.mean(x**2),
                zcr=((x[:-1]*x[1:])<0).mean())

def spectral_features(x,fs):
    N=len(x)
    if N<8: return {k:0.0 for k in["dom_freq","dom_power","spec_centroid","spec_entropy","band05_1","band1_3"]}
    X=np.abs(rfft(x*np.hanning(N)))+1e-12;f=rfftfreq(N,1/fs);P=X**2;S=np.sum(P)
    dom=int(np.argmax(P));pnorm=P/(S+1e-12)
    return dict(dom_freq=float(f[dom]),
                dom_power=float(P[dom]/(S+1e-12)),
                spec_centroid=float(np.sum(f*P)/(S+1e-12)),
                spec_entropy=float(-np.sum(pnorm*np.log(pnorm))),
                band05_1=float(np.mean(P[(f>=0.5)&(f<1.0)])),
                band1_3=float(np.mean(P[(f>=1.0)&(f<3.0)])))

# --------------------------- pipeline -----------------------------
def preprocess_and_features(df,fs=100.0,y_axis_boost=1.0,pre_lpf_hz=10.0,grav_lpf_hz=0.3,win_s=5.0,hop_s=2.5):
    n0=int(fs*2)
    ax=df["ax_g"].values-np.mean(df["ax_g"].values[:n0])
    ay=df["ay_g"].values-np.mean(df["ay_g"].values[:n0])
    az=df["az_g"].values-np.mean(df["az_g"].values[:n0])
    ax_raw,ay_raw,az_raw=ax.copy(),ay.copy(),az.copy()

    ax=kalman_1d(ax); ay=kalman_1d(ay); az=kalman_1d(az)
    ax_lpf=butter_lpf(ax,pre_lpf_hz,fs)
    ay_lpf=butter_lpf(ay,pre_lpf_hz,fs)
    az_lpf=butter_lpf(az,pre_lpf_hz,fs)
    gax=butter_lpf(ax_lpf,grav_lpf_hz,fs)
    gay=butter_lpf(ay_lpf,grav_lpf_hz,fs)
    gaz=butter_lpf(az_lpf,grav_lpf_hz,fs)
    dax,day,daz=ax_lpf-gax,(ay_lpf-gay)*y_axis_boost,az_lpf-gaz
    amag=np.sqrt(dax**2+day**2+daz**2)

    win=int(win_s*fs); hop=int(hop_s*fs)
    total_samples=len(df)
    rows=[]; starts=list(range(0, total_samples - win + 1, hop))
    if starts[0]!=0: starts.insert(0,0)

    for i in starts:
        i1=i+win
        if i1>total_samples: break
        t0=int(df["timestamp_ms"].iloc[i])
        t1=int(df["timestamp_ms"].iloc[i1-1])
        feats={"t_start_ms":t0,"t_end_ms":t1}
        segs={"x":dax[i:i1],"y":day[i:i1],"z":daz[i:i1],"mag":amag[i:i1]}
        for nm,s in segs.items(): feats.update({f"{nm}_{k}":v for k,v in time_features(s).items()})
        feats.update({f"y_{k}":v for k,v in spectral_features(segs["y"],fs).items()})
        feats.update({f"mag_{k}":v for k,v in spectral_features(segs["mag"],fs).items()})
        rows.append(feats)

    print(f"🪄 Extracted {len(rows)} windows "
          f"(window={win_s}s, hop={hop_s}s, overlap≈{100*(1-hop_s/win_s):.0f}%) "
          f"from {total_samples/fs:.1f}s of data.")
    arrays=(ax_raw,ay_raw,az_raw,ax_lpf,ay_lpf,az_lpf,gax,gay,gaz,dax,day,daz,amag,starts,win)
    return pd.DataFrame(rows),arrays

# -------------------------- scaling utils --------------------------
def fit_scaler(df,cols): 
    mu=df[cols].mean(); sd=df[cols].std(ddof=0)
    return {"means":mu.to_dict(),"stds":sd.to_dict()}

def apply_scaler(df,scaler,cols):
    for c in cols: df[c]=(df[c]-scaler["means"][c])/(scaler["stds"][c]+1e-12)
    return df

# ---------------------------- plotting -----------------------------
def plot_all(df, fs, feats,
             ax_raw, ay_raw, az_raw,
             ax_lpf, ay_lpf, az_lpf,
             gax, gay, gaz,
             dax, day, daz, amag,
             starts, win, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    t = df["timestamp_ms"].values / 1000

    # === 1️⃣ Raw vs LPF (each axis) ===
    for name, raw, lpf in [("X",ax_raw,ax_lpf),("Y",ay_raw,ay_lpf),("Z",az_raw,az_lpf)]:
        plt.figure(figsize=(10,5))
        plt.plot(t, raw, label=f"{name} raw", alpha=0.6)
        plt.plot(t, lpf, label=f"{name} LPF10", lw=1.5)
        plt.title(f"{name}-Axis: Raw vs LPF10"); plt.xlabel("Time (s)"); plt.ylabel("g")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(outdir/f"{name.lower()}_raw_vs_lpf.png", dpi=150); plt.close()

    # === 2️⃣ LPF vs Gravity (each axis) ===
    for name, lpf, grav in [("X",ax_lpf,gax),("Y",ay_lpf,gay),("Z",az_lpf,gaz)]:
        plt.figure(figsize=(10,5))
        plt.plot(t, lpf, label=f"{name} LPF10")
        plt.plot(t, grav, label=f"{name} Gravity (0.3 Hz)")
        plt.title(f"{name}-Axis: LPF10 vs Gravity"); plt.xlabel("Time (s)"); plt.ylabel("g")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(outdir/f"{name.lower()}_lpf_vs_gravity.png", dpi=150); plt.close()

    # === 3️⃣ Dynamic Components ===
    plt.figure(figsize=(10,5))
    plt.plot(t,dax,label="dX"); plt.plot(t,day,label="dY"); plt.plot(t,daz,label="dZ")
    plt.title("Dynamic Acceleration (Gravity Removed)")
    plt.xlabel("Time (s)"); plt.ylabel("g"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(outdir/"dynamic_xyz.png", dpi=150); plt.close()

    # === 4️⃣ Zoomed Raw vs LPF (first 5 s) ===
    zoom_dur=int(5*fs); t_zoom=t[:zoom_dur]
    for name, raw, lpf in [("X",ax_raw,ax_lpf),("Y",ay_raw,ay_lpf),("Z",az_raw,az_lpf)]:
        plt.figure(figsize=(10,5))
        plt.plot(t_zoom, raw[:zoom_dur], label=f"{name} raw", alpha=0.7)
        plt.plot(t_zoom, lpf[:zoom_dur], label=f"{name} LPF10")
        plt.title(f"{name}-Axis Zoom (First 5 s)"); plt.xlabel("Time (s)"); plt.ylabel("g")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(outdir/f"{name.lower()}_raw_vs_lpf_zoom.png", dpi=150); plt.close()

    # === 5️⃣ Combined LPF10 XYZ ===
    plt.figure(figsize=(10,5))
    plt.plot(t, ax_lpf, label="X", color='tab:red')
    plt.plot(t, ay_lpf, label="Y", color='tab:green')
    plt.plot(t, az_lpf, label="Z", color='tab:blue')
    plt.title("Combined LPF10 Filtered Axes (X, Y, Z)")
    plt.xlabel("Time (s)"); plt.ylabel("Acceleration (g)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(outdir/"combined_xyz_lpf.png", dpi=150); plt.close()

    # === 6️⃣ Window Segmentation ===
    plt.figure(figsize=(12,5))
    plt.plot(t, amag, color='tab:blue', lw=1, label="|a_dyn|")
    for idx,i in enumerate(starts):
        t0=t[i]; t1=t[i+win-1] if i+win-1<len(t) else t[-1]
        plt.axvspan(t0,t1,color='tab:orange',alpha=0.15 if idx%2==0 else 0.05)
    plt.title("Overlapping Window Segmentation (5s window, 2.5s hop)")
    plt.xlabel("Time (s)"); plt.ylabel("Dynamic Magnitude (g)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(outdir/"window_segmentation.png", dpi=150); plt.close()

    # === 7️⃣ Frequency Comparison 2.5s vs 5s ===
    x5=amag[:int(5*fs)]; x25=amag[:int(2.5*fs)]
    f5=rfftfreq(len(x5),1/fs); f25=rfftfreq(len(x25),1/fs)
    P5=np.abs(rfft(x5*windows.hann(len(x5)))); P25=np.abs(rfft(x25*windows.hann(len(x25))))
    plt.figure(figsize=(10,5))
    plt.plot(f25,P25/np.max(P25),label="2.5s window",color='tab:orange')
    plt.plot(f5,P5/np.max(P5),label="5s window",color='tab:blue')
    plt.title("Frequency-Domain Stability (2.5s vs 5s)")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Normalized Amplitude")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(outdir/"frequency_comparison.png", dpi=150); plt.close()

    # === 8️⃣ FFT XYZ Comparison ===
    plt.figure(figsize=(10,5))
    for sig,name,color in zip([ax_lpf,ay_lpf,az_lpf],["X","Y","Z"],["tab:red","tab:green","tab:blue"]):
        f=rfftfreq(len(sig),1/fs); P=np.abs(rfft(sig*np.hanning(len(sig))))
        plt.plot(f,P/np.max(P),label=f"{name}-axis",color=color)
    plt.title("Frequency Spectrum by Axis (LPF10)")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Normalized Amplitude")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(outdir/"fft_xyz_comparison.png", dpi=150); plt.close()

    # === 9️⃣ Histogram of Dynamic Magnitude ===
    plt.figure(figsize=(8,5))
    plt.hist(amag,bins=60,color='skyblue',edgecolor='black')
    plt.title("Distribution of Dynamic Acceleration Magnitude")
    plt.xlabel("|a_dyn| (g)"); plt.ylabel("Frequency")
    plt.grid(True); plt.tight_layout()
    plt.savefig(outdir/"amag_histogram.png", dpi=150); plt.close()

    # === 🔟 Axis Correlation Matrix ===
    corr=pd.DataFrame({"X":ax_lpf,"Y":ay_lpf,"Z":az_lpf}).corr()
    plt.figure(figsize=(5,4))
    sns.heatmap(corr,annot=True,cmap="coolwarm",fmt=".2f",vmin=-1,vmax=1)
    plt.title("Axis Correlation Matrix (LPF10)")
    plt.tight_layout(); plt.savefig(outdir/"axis_correlation.png", dpi=150); plt.close()

    # === 1️⃣1️⃣ Feature Trends ===
    if not feats.empty and "mag_mean" in feats.columns:
        plt.figure(figsize=(10,5))
        plt.plot(feats["t_start_ms"]/1000,feats["mag_mean"],label="Mean(|a|)")
        plt.plot(feats["t_start_ms"]/1000,feats["mag_std"],label="Std(|a|)")
        plt.title("Feature Trends Over Time")
        plt.xlabel("Time (s)"); plt.ylabel("Value")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(outdir/"feature_trends.png", dpi=150); plt.close()

# ------------------------------- main -------------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in",dest="inp",required=True)
    ap.add_argument("--out",dest="out",required=True)
    ap.add_argument("--scaler",required=True)
    ap.add_argument("--plots",default=None)
    ap.add_argument("--fs",type=float,default=100.0)
    ap.add_argument("--units",choices=["g","mg"],default="g")
    ap.add_argument("--y_boost",type=float,default=1.0)
    ap.add_argument("--win_s",type=float,default=5.0)
    ap.add_argument("--hop_s",type=float,default=2.5)
    ap.add_argument("--label",default=None)
    args=ap.parse_args()

    df=pd.read_csv(args.inp); df=normalize_and_map_columns(df)
    if args.units=="mg": df[["ax_g","ay_g","az_g"]] /= 1000.0

    feats,arr=preprocess_and_features(df,fs=args.fs,y_axis_boost=args.y_boost,win_s=args.win_s,hop_s=args.hop_s)
    if feats.empty: 
        print("⚠️ No valid windows."); return

    if args.label: feats["label"]=args.label
    feature_cols=[c for c in feats.columns if c not in("t_start_ms","t_end_ms","label")]
    scaler=fit_scaler(feats,feature_cols)
    feats_scaled=apply_scaler(feats.copy(),scaler,feature_cols)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.scaler).parent.mkdir(parents=True,exist_ok=True)
    feats_scaled.to_csv(args.out,index=False)
    with open(args.scaler,"w") as f: json.dump(scaler,f,indent=2)

    plots_dir=Path(args.plots) if args.plots else Path(args.out).parent/"plots"
    (ax_raw,ay_raw,az_raw,ax,ay,az,gax,gay,gaz,dax,day,daz,amag,starts,win)=arr
    plot_all(df,args.fs,feats,ax_raw,ay_raw,az_raw,ax,ay,az,gax,gay,gaz,dax,day,daz,amag,starts,win,plots_dir)

    print(f"✅ Features → {args.out}")
    print(f"✅ Scaler   → {args.scaler}")
    print(f"🖼️ Plots saved to: {plots_dir}")

if __name__=="__main__":
    main()
