#!/usr/bin/env python3
"""
Full preprocessing with before/after filter plots and flexible windows.

Example:
  & "E:/Program Files/Python/python.exe" "E:/documents/Thesis code/preprocessingv2.py" `
    --in "E:/documents/Thesis code/LOGS/LOG005_clean.csv" `
    --out "E:/documents/Thesis code/backstrokefeatures_paper_5s.csv" `
    --scaler "E:/documents/Thesis code/backstrokescaler_paper_5s.json" `
    --plots "E:/documents/Thesis code/backstrokeplots_5s" `
    --fs 100 --units g --win_s 5 --hop_s 2.5 --zoom_s 5 --zoom_stride_s 5 `
    --label backstroke
"""

import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt

# ------------------------ column handling ------------------------
def normalize_and_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
    df.columns = [re.sub(r'\s+', '_', str(c).strip().lower()) for c in df.columns]
    for c in list(df.columns):
        if c.startswith("az_g") and c != "az_g":
            df = df.rename(columns={c: "az_g"})
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
def preprocess_and_features(df,fs=100.0,y_axis_boost=1.0,pre_lpf_hz=10.0,grav_lpf_hz=0.3,win_s=2.0,hop_s=1.0):
    n0=int(fs*2)
    ax=df["ax_g"].values-np.mean(df["ax_g"].values[:n0])
    ay=df["ay_g"].values-np.mean(df["ay_g"].values[:n0])
    az=df["az_g"].values-np.mean(df["az_g"].values[:n0])
    ax_raw,ay_raw,az_raw=ax.copy(),ay.copy(),az.copy()

    ax=kalman_1d(ax);ay=kalman_1d(ay);az=kalman_1d(az)
    ax_lpf=butter_lpf(ax,pre_lpf_hz,fs)
    ay_lpf=butter_lpf(ay,pre_lpf_hz,fs)
    az_lpf=butter_lpf(az,pre_lpf_hz,fs)
    gax=butter_lpf(ax_lpf,grav_lpf_hz,fs)
    gay=butter_lpf(ay_lpf,grav_lpf_hz,fs)
    gaz=butter_lpf(az_lpf,grav_lpf_hz,fs)
    dax,day,daz=ax_lpf-gax,(ay_lpf-gay)*y_axis_boost,az_lpf-gaz
    amag=np.sqrt(dax**2+day**2+daz**2)

    win,hop=int(win_s*fs),int(hop_s*fs)
    rows=[]
    for i in range(0,len(df)-win+1,hop):
        t0=int(df["timestamp_ms"].iloc[i]);t1=int(df["timestamp_ms"].iloc[i+win-1])
        feats={"t_start_ms":t0,"t_end_ms":t1}
        segs={"x":dax[i:i+win],"y":day[i:i+win],"z":daz[i:i+win],"mag":amag[i:i+win]}
        for nm,s in segs.items(): feats.update({f"{nm}_{k}":v for k,v in time_features(s).items()})
        feats.update({f"y_{k}":v for k,v in spectral_features(segs["y"],fs).items()})
        feats.update({f"mag_{k}":v for k,v in spectral_features(segs["mag"],fs).items()})
        rows.append(feats)
    arrays=(ax_raw,ay_raw,az_raw,ax_lpf,ay_lpf,az_lpf,gax,gay,gaz,dax,day,daz,amag)
    return pd.DataFrame(rows),arrays

# -------------------------- scaling utils --------------------------
def fit_scaler(df,cols): 
    mu=df[cols].mean(); sd=df[cols].std(ddof=0)
    return {"means":mu.to_dict(),"stds":sd.to_dict()}

def apply_scaler(df,scaler,cols):
    for c in cols: df[c]=(df[c]-scaler["means"][c])/(scaler["stds"][c]+1e-12)
    return df

# ------------------------------ plots ------------------------------
def plot_overview(df,fs,ax_raw,ay_raw,az_raw,ax,ay,az,gax,gay,gaz,dax,day,daz,amag,outdir):
    outdir.mkdir(parents=True,exist_ok=True);t=df["timestamp_ms"].values/1000
    # raw vs filtered (X)
    plt.figure(figsize=(11,5))
    plt.plot(t,ax_raw,label="X raw");plt.plot(t,ax,label="X after LPF10")
    plt.title("Raw vs filtered (X)");plt.xlabel("Time (s)");plt.ylabel("g")
    plt.legend();plt.grid(True);plt.tight_layout();plt.savefig(outdir/"x_raw_vs_lpf.png",dpi=150);plt.close()
    # gravity tracking
    plt.figure(figsize=(11,5))
    plt.plot(t,ax,label="X LPF10");plt.plot(t,gax,label="X gravity (0.3Hz)")
    plt.title("X LPF vs gravity");plt.legend();plt.grid(True);plt.tight_layout();plt.savefig(outdir/"x_lpf_gravity.png",dpi=150);plt.close()
    # dynamic xyz
    plt.figure(figsize=(11,5))
    plt.plot(t,dax,label="dax");plt.plot(t,day,label="day");plt.plot(t,daz,label="daz")
    plt.title("Dynamic accel");plt.legend();plt.grid(True);plt.tight_layout();plt.savefig(outdir/"dynamic_xyz.png",dpi=150);plt.close()
    # dynamic magnitude
    plt.figure(figsize=(11,5))
    plt.plot(t,amag,label="|a_dyn|");plt.title("Dynamic magnitude");plt.legend();plt.grid(True)
    plt.tight_layout();plt.savefig(outdir/"amag.png",dpi=150);plt.close()

def plot_zoom_segments(df,fs,ax,ay,az,gax,gay,gaz,dax,day,daz,amag,outdir,zoom_s=5.0,stride_s=5.0):
    t=df["timestamp_ms"].values/1000;N=len(df);w=int(zoom_s*fs);stride=int(stride_s*fs)
    idx=0;outdir.mkdir(parents=True,exist_ok=True)
    for i0 in range(0,max(1,N-w+1),stride):
        i1=min(i0+w,N);tt=t[i0:i1]
        plt.figure(figsize=(10,4))
        plt.plot(tt,ax[i0:i1],label="X LPF10");plt.plot(tt,gax[i0:i1],label="X gravity")
        plt.title(f"X LPF vs grav {tt[0]:.1f}-{tt[-1]:.1f}s");plt.legend();plt.grid(True)
        plt.tight_layout();plt.savefig(outdir/f"x_lpf_gravity_zoom_{idx:03d}.png",dpi=150);plt.close()
        plt.figure(figsize=(10,4))
        plt.plot(tt,dax[i0:i1],label="dax");plt.plot(tt,day[i0:i1],label="day");plt.plot(tt,daz[i0:i1],label="daz")
        plt.title(f"Dynamic xyz {tt[0]:.1f}-{tt[-1]:.1f}s");plt.legend();plt.grid(True)
        plt.tight_layout();plt.savefig(outdir/f"dynamic_xyz_zoom_{idx:03d}.png",dpi=150);plt.close()
        plt.figure(figsize=(10,4))
        plt.plot(tt,amag[i0:i1],label="|a|");plt.title(f"|a| {tt[0]:.1f}-{tt[-1]:.1f}s")
        plt.legend();plt.grid(True);plt.tight_layout()
        plt.savefig(outdir/f"amag_zoom_{idx:03d}.png",dpi=150);plt.close()
        idx+=1

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
    ap.add_argument("--win_s",type=float,default=2.0)
    ap.add_argument("--hop_s",type=float,default=1.0)
    ap.add_argument("--zoom_s",type=float,default=5.0)
    ap.add_argument("--zoom_stride_s",type=float,default=5.0)
    ap.add_argument("--label", default=None, help="Optional class label to attach to all rows")
    args = ap.parse_args()

    # read & normalize
    df = pd.read_csv(args.inp)
    df = normalize_and_map_columns(df)
    if args.units == "mg":
        df[["ax_g","ay_g","az_g"]] /= 1000.0

    # extract features
    feats, arr = preprocess_and_features(
        df, fs=args.fs, y_axis_boost=args.y_boost,
        win_s=args.win_s, hop_s=args.hop_s
    )
    if feats.empty:
        print("⚠️ No valid windows.")
        return

    # attach label and session_id
    if args.label:
        feats["label"] = args.label
    feats["session_id"] = Path(args.inp).stem

    # scale & save
    feature_cols = [c for c in feats.columns if c not in ("t_start_ms","t_end_ms","label","session_id")]
    scaler = fit_scaler(feats, feature_cols)
    feats_scaled = apply_scaler(feats.copy(), scaler, feature_cols)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    feats_scaled.to_csv(args.out, index=False)
    with open(args.scaler,"w") as f: json.dump(scaler, f, indent=2)

    # plots
    plots_dir = Path(args.plots) if args.plots else Path(args.out).parent/"plots"
    (ax_raw,ay_raw,az_raw,ax,ay,az,gax,gay,gaz,dax,day,daz,amag) = arr
    plot_overview(df,args.fs,ax_raw,ay_raw,az_raw,ax,ay,az,gax,gay,gaz,dax,day,daz,amag,plots_dir)
    plot_zoom_segments(df,args.fs,ax,ay,az,gax,gay,gaz,dax,day,daz,amag,plots_dir,
                       zoom_s=args.zoom_s,stride_s=args.zoom_stride_s)

    print(f"✅ Features → {args.out}")
    print(f"✅ Scaler   → {args.scaler}")
    print(f"🖼️ Plots    → {plots_dir}")

if __name__=="__main__":
    main()
