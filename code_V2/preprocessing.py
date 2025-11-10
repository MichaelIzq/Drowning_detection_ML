#!/usr/bin/env python3
"""
prepare_and_preprocess.py
-------------------------
Preprocess accelerometer CSV (time_ms, ax_g, ay_g, az_g*) into filtered,
windowed feature data for swimming/drowning detection.

Usage example (PowerShell or VS Code terminal):

    & "E:/Program Files/Python/python.exe" "E:/documents/Thesis code/preprocessing.py" `
        --in "E:/documents/Thesis code/LOGS/LOG002_clean.csv" `
        --out "E:/documents/Thesis code/features.csv" `
        --scaler "E:/documents/Thesis code/scaler.json" `
        --fs 100 --units g
"""

import argparse, json, re
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from numpy.fft import rfft, rfftfreq
from pathlib import Path

# ---------------------------------------------------------------------
# Utility filters
# ---------------------------------------------------------------------
def butter_lpf(x, fc, fs, order=4):
    b, a = butter(order, fc / (0.5 * fs), btype="low")
    return filtfilt(b, a, x)

# ---------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------
def time_features(x):
    mean = np.mean(x); std = np.std(x)
    var = np.var(x); skew = ((x-mean)**3).mean()/(std**3+1e-12)
    kurt = ((x-mean)**4).mean()/(std**4+1e-12)
    return dict(mean=mean, std=std, var=var, skew=skew, kurt=kurt,
                min=np.min(x), max=np.max(x), sma=np.mean(np.abs(x)),
                energy=np.mean(x**2))

def spectral_features(x, fs):
    N = len(x)
    X = np.abs(rfft(x*np.hanning(N))) + 1e-12
    freqs = rfftfreq(N, 1/fs)
    power = X**2
    dom_idx = np.argmax(power)
    dom_freq, dom_power = freqs[dom_idx], power[dom_idx]/np.sum(power)
    centroid = np.sum(freqs*power)/np.sum(power)
    pnorm = power/np.sum(power)
    entropy = -np.sum(pnorm*np.log(pnorm))
    return dict(dom_freq=dom_freq, dom_power=dom_power,
                spec_centroid=centroid, spec_entropy=entropy)



# ---------------------------------------------------------------------
# Robust column normalization
# ---------------------------------------------------------------------
def normalize_and_map_columns(df):
       # --- 🔧 Auto-fix column names for accelerometer data ---
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Drop unnamed or empty columns
    df = df.loc[:, ~df.columns.str.contains('^unnamed', na=False)]

    # Handle merged or variant names like 'az_gfreestyle' or 'z_acc'
    for c in list(df.columns):
        if c.startswith("az_g") and c != "az_g":
            df = df.rename(columns={c: "az_g"})
        elif c.startswith("z_acc"):
            df = df.rename(columns={c: "az_g"})
        elif c == "z":
            df = df.rename(columns={c: "az_g"})

    # Print what we actually have
    print("✅ Column headers after normalization:", list(df.columns))

    # Flexible mapping
    mapping = {}
    for c in ["timestamp_ms","time_ms","timestamp","time"]:
        if c in df.columns: mapping[c] = "timestamp_ms"; break
    for c in ["ax_g","x_acc","acc_x","ax","x"]:
        if c in df.columns: mapping[c] = "ax_g"; break
    for c in ["ay_g","y_acc","acc_y","ay","y"]:
        if c in df.columns: mapping[c] = "ay_g"; break
    for c in ["az_g","z_acc","acc_z","az","z"]:
        if c in df.columns: mapping[c] = "az_g"; break

    df = df.rename(columns=mapping)
    missing = [c for c in ["ax_g","ay_g","az_g"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {missing}. Headers seen: {list(df.columns)}")
    return df

# ---------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------
def preprocess(df, fs=100.0):
    ax, ay, az = df["ax_g"].values, df["ay_g"].values, df["az_g"].values
    # Remove initial bias
    ax -= np.mean(ax[:int(fs*2)]); ay -= np.mean(ay[:int(fs*2)]); az -= np.mean(az[:int(fs*2)])
    # Filter chain
    ax = butter_lpf(ax,10,fs); ay = butter_lpf(ay,10,fs); az = butter_lpf(az,10,fs)
    gax, gay, gaz = butter_lpf(ax,0.3,fs), butter_lpf(ay,0.3,fs), butter_lpf(az,0.3,fs)
    dax, day, daz = ax-gax, ay-gay, az-gaz
    amag = np.sqrt(dax**2+day**2+daz**2)

    win, hop = int(2*fs), int(1*fs)
    rows=[]
    for i in range(0,len(df)-win,hop):
        t0,t1=df["timestamp_ms"].iloc[i],df["timestamp_ms"].iloc[i+win-1]
        feats={"t_start_ms":int(t0),"t_end_ms":int(t1)}
        for axis,seg in zip(["x","y","z","mag"],[dax[i:i+win],day[i:i+win],daz[i:i+win],amag[i:i+win]]):
            feats.update({f"{axis}_{k}":v for k,v in time_features(seg).items()})
            if axis in ["y","mag"]:
                feats.update({f"{axis}_{k}":v for k,v in spectral_features(seg,fs).items()})
        rows.append(feats)
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------
# Scaling utilities
# ---------------------------------------------------------------------
def fit_scaler(df, cols):
    mu=df[cols].mean(); sigma=df[cols].std(ddof=0)
    return {"means":mu.to_dict(),"stds":sigma.to_dict()}

def apply_scaler(df,scaler,cols):
    for c in cols:
        df[c]=(df[c]-scaler["means"][c])/(scaler["stds"][c]+1e-12)
    return df

# ---------------------------------------------------------------------
# Optional plot
# ---------------------------------------------------------------------
def plot_preview(df, fs):
    import matplotlib.pyplot as plt
    from scipy.signal import butter, filtfilt
    def lpf(x,fc): b,a=butter(4,fc/(0.5*fs),"low"); return filtfilt(b,a,x)
    ax,ay,az=df["ax_g"],df["ay_g"],df["az_g"]
    amag=np.sqrt(ax**2+ay**2+az**2)
    t=df["timestamp_ms"]/1000
    plt.figure(figsize=(10,5))
    plt.plot(t, amag, label="|a| raw")
    plt.plot(t, lpf(amag,10), label="|a| 10 Hz LPF")
    plt.xlabel("Time (s)"); plt.ylabel("g")
    plt.title("Acceleration magnitude preview")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--scaler", required=True)
    ap.add_argument("--fs", type=float, default=100.0)
    ap.add_argument("--units", choices=["g","mg"], default="g")
    args=ap.parse_args()

    df=pd.read_csv(args.inp)
    df=normalize_and_map_columns(df)
    if args.units=="mg":
        df[["ax_g","ay_g","az_g"]]/=1000.0

    feats=preprocess(df,fs=args.fs)
    if feats.empty: print("⚠️ No valid windows."); return

    cols=[c for c in feats.columns if c not in ("t_start_ms","t_end_ms")]
    scaler=fit_scaler(feats,cols)
    feats_scaled=apply_scaler(feats.copy(),scaler,cols)

    feats_scaled.to_csv(args.out,index=False)
    with open(args.scaler,"w") as f: json.dump(scaler,f,indent=2)
    print(f"✅ Saved features → {args.out}")
    print(f"✅ Saved scaler → {args.scaler}")

    # optional visualization
    try:
        plot_preview(df, args.fs)
    except Exception as e:
        print("Plot skipped:", e)

# ---------------------------------------------------------------------
if __name__=="__main__":
    main()
