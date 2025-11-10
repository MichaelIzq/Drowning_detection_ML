#!/usr/bin/env python3
"""
Preprocessing v5: Complete filtering, overlap windows, diagnostic plots,
and automatic summary text report.

Usage example:
& "E:\documents\Thesis code\preprocessing_for_singlestroke.py" `
  --in "E:\documents\Thesis code\LOG002_capturestroke_clean" `
  --out "E:/documents/Thesis code/single_Stroke_analysis/freestyle_nobias.csv" `
  --scaler "E:/documents/Thesis code/scalers/freestyle_nobias.json" `
  --plots "E:/documents/Thesis code/single_stoke_analysis/freestyle_nobias" `
  --fs 100 --units g --win_s 5 --hop_s 2.5 --label freestyle --disable_bias
"""

import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt, windows

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
def preprocess_and_features(df,fs=100.0,y_axis_boost=1.0,pre_lpf_hz=10.0,grav_lpf_hz=0.3,
                            win_s=5.0,hop_s=2.5,disable_bias=False):
    n0=int(fs*2)
    ax, ay, az = df["ax_g"].values, df["ay_g"].values, df["az_g"].values

    if not disable_bias:
        ax=ax-np.mean(ax[:n0]); ay=ay-np.mean(ay[:n0]); az=az-np.mean(az[:n0])

    ax_raw,ay_raw,az_raw=ax.copy(),ay.copy(),az.copy()

    # Filters
    ax=kalman_1d(ax); ay=kalman_1d(ay); az=kalman_1d(az)
    ax_lpf=butter_lpf(ax,pre_lpf_hz,fs)
    ay_lpf=butter_lpf(ay,pre_lpf_hz,fs)
    az_lpf=butter_lpf(az,pre_lpf_hz,fs)
    gax=butter_lpf(ax_lpf,grav_lpf_hz,fs)
    gay=butter_lpf(ay_lpf,grav_lpf_hz,fs)
    gaz=butter_lpf(az_lpf,grav_lpf_hz,fs)
    dax,day,daz=ax_lpf-gax,(ay_lpf-gay)*y_axis_boost,az_lpf-gaz
    amag=np.sqrt(dax**2+day**2+daz**2)

    # Windows
    win=int(win_s*fs); hop=int(hop_s*fs)
    total_samples=len(df)
    start_indices = list(range(0, total_samples - win + 1, hop))
    if start_indices[0] != 0: start_indices.insert(0, 0)

    rows=[]
    for i in start_indices:
        i1=i+win
        if i1>total_samples: break
        t0=int(df["timestamp_ms"].iloc[i]); t1=int(df["timestamp_ms"].iloc[i1-1])
        feats={"t_start_ms":t0,"t_end_ms":t1}
        segs={"x":dax[i:i1],"y":day[i:i1],"z":daz[i:i1],"mag":amag[i:i1]}
        for nm,s in segs.items(): feats.update({f"{nm}_{k}":v for k,v in time_features(s).items()})
        feats.update({f"y_{k}":v for k,v in spectral_features(segs["y"],fs).items()})
        feats.update({f"mag_{k}":v for k,v in spectral_features(segs["mag"],fs).items()})
        rows.append(feats)

    arrays=(ax_raw,ay_raw,az_raw,ax,ay,az,ax_lpf,ay_lpf,az_lpf,gax,gay,gaz,dax,day,daz,amag,start_indices,win)
    return pd.DataFrame(rows),arrays

# -------------------------- scaling utils --------------------------
def fit_scaler(df,cols): mu=df[cols].mean();sd=df[cols].std(ddof=0);return{"means":mu.to_dict(),"stds":sd.to_dict()}
def apply_scaler(df,scaler,cols):
    for c in cols: df[c]=(df[c]-scaler["means"][c])/(scaler["stds"][c]+1e-12)
    return df

# ---------------------------- plotting -----------------------------
def plot_all(df,fs,arrays,outdir):
    outdir.mkdir(parents=True,exist_ok=True)
    (ax_raw,ay_raw,az_raw,ax,ay,az,ax_lpf,ay_lpf,az_lpf,gax,gay,gaz,dax,day,daz,amag,starts,win)=arrays
    t=df["timestamp_ms"].values/1000

    # LPF individual + combined
    for label,raw,lpf in zip(["x","y","z"],[ax,ay,az],[ax_lpf,ay_lpf,az_lpf]):
        plt.figure(figsize=(10,4))
        plt.plot(t,raw,label=f"{label.upper()} Raw",alpha=0.6)
        plt.plot(t,lpf,label=f"{label.upper()} LPF10",lw=1.2)
        plt.title(f"{label.upper()} Raw vs LPF10"); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(outdir/f"{label}_raw_vs_lpf.png",dpi=150); plt.close()

    plt.figure(figsize=(10,4))
    plt.plot(t,ax_lpf,label="X_LPF"); plt.plot(t,ay_lpf,label="Y_LPF"); plt.plot(t,az_lpf,label="Z_LPF")
    plt.title("Combined Low-Pass Filtered Acceleration (XYZ)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(outdir/"combined_lpf_xyz.png",dpi=150); plt.close()

    # Dynamic accel + magnitude
    plt.figure(figsize=(10,4))
    plt.plot(t,dax,label="dX"); plt.plot(t,day,label="dY"); plt.plot(t,daz,label="dZ")
    plt.title("Dynamic Acceleration (XYZ)"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(outdir/"dynamic_xyz.png",dpi=150); plt.close()

    plt.figure(figsize=(10,4))
    plt.plot(t,amag,label="|a_dyn|",color="black")
    plt.title("Dynamic Magnitude"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(outdir/"amag.png",dpi=150); plt.close()

    # FFT plots
    N=int(5*fs)
    for lbl,sig in zip(["X","Y","Z"],[ax_lpf,ay_lpf,az_lpf]):
        f=rfftfreq(N,1/fs);P=np.abs(rfft(sig[:N]*windows.hann(N)))
        plt.plot(f,P,label=lbl)
    plt.title("FFT Spectrum (Combined XYZ, first 5 s)")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Amplitude"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(outdir/"fft_xyz_combined.png",dpi=150); plt.close()

    for lbl,sig in zip(["x","y","z"],[ax_lpf,ay_lpf,az_lpf]):
        f=rfftfreq(N,1/fs);P=np.abs(rfft(sig[:N]*windows.hann(N)))
        plt.figure(figsize=(8,4))
        plt.plot(f,P,color="tab:blue"); plt.title(f"FFT Spectrum ({lbl.upper()} axis, first 5 s)")
        plt.xlabel("Frequency (Hz)"); plt.ylabel("Amplitude"); plt.grid(True)
        plt.tight_layout(); plt.savefig(outdir/f"fft_{lbl}.png",dpi=150); plt.close()

# -------------------------- summary report --------------------------
def write_summary_report(input_path, fs, win_s, hop_s, disable_bias, feats, amag, outdir):
    report_path = outdir / "summary.txt"
    duration = len(amag)/fs
    overlap = 100*(1 - hop_s/win_s)
    mean_amp = np.mean(amag); std_amp = np.std(amag); p2p_amp = np.ptp(amag)
    f = rfftfreq(len(amag), 1/fs)
    P = np.abs(rfft(amag*np.hanning(len(amag))))
    dom_freq = f[np.argmax(P)]
    stroke_rate = dom_freq * 60.0

    with open(report_path, "w") as ftxt:
        ftxt.write("=== Preprocessing Summary ===\n")
        ftxt.write(f"Input file: {Path(input_path).name}\n")
        ftxt.write(f"Sampling rate: {fs:.1f} Hz\n")
        ftxt.write(f"Windows extracted: {len(feats)} (window={win_s:.1f}s, hop={hop_s:.1f}s)\n")
        ftxt.write(f"Overlap: {overlap:.1f}%\n")
        ftxt.write(f"Total duration: {duration:.1f} s\n")
        ftxt.write(f"Bias removal: {'Disabled' if disable_bias else 'Enabled'}\n\n")
        ftxt.write("--- Key Feature Averages (Dynamic Magnitude) ---\n")
        ftxt.write(f"Mean amplitude: {mean_amp:.3f} g\n")
        ftxt.write(f"Std amplitude: {std_amp:.3f} g\n")
        ftxt.write(f"Peak-to-Peak: {p2p_amp:.3f} g\n")
        ftxt.write(f"Dominant frequency (approx): {dom_freq:.2f} Hz\n")
        ftxt.write(f"Estimated stroke rate: {stroke_rate:.1f} cycles/min\n")

    print(f"📄 Summary report saved → {report_path}")

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
    ap.add_argument("--disable_bias",action="store_true",help="Disable initial bias removal")
    args=ap.parse_args()

    df=pd.read_csv(args.inp)
    df=normalize_and_map_columns(df)
    if args.units=="mg": df[["ax_g","ay_g","az_g"]] /= 1000.0

    feats,arr=preprocess_and_features(df,fs=args.fs,y_axis_boost=args.y_boost,
                                      win_s=args.win_s,hop_s=args.hop_s,
                                      disable_bias=args.disable_bias)
    if feats.empty:
        print("⚠️ No valid windows."); return

    if args.label: feats["label"]=args.label

    feature_cols=[c for c in feats.columns if c not in("t_start_ms","t_end_ms","label")]
    scaler=fit_scaler(feats,feature_cols)
    feats_scaled=apply_scaler(feats.copy(),scaler,feature_cols)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    feats_scaled.to_csv(args.out,index=False)
    with open(args.scaler,"w") as f: json.dump(scaler,f,indent=2)

    plots_dir=Path(args.plots) if args.plots else Path(args.out).parent/"plots"
    plot_all(df,args.fs,arr,plots_dir)

    # Save summary report
    (_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,amag,_,_)=arr
    write_summary_report(args.inp, args.fs, args.win_s, args.hop_s, args.disable_bias, feats, amag, plots_dir)

    print(f"\n✅ Features → {args.out}")
    print(f"✅ Scaler   → {args.scaler}")
    print(f"🖼️ Plots    → {plots_dir}")

if __name__=="__main__":
    main()
