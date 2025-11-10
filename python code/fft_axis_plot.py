#!/usr/bin/env python3
"""
Generate separate FFT plots for X, Y, and Z axes:
 - Normalized (0–1)
 - Absolute amplitude (real energy comparison)

Usage example:
& "E:/Program Files/Python/python.exe" "E:\documents\Thesis code\fft_axis_plot.py" `
  --in "E:\documents\Thesis code\clean_&_filtered\back\LOG005_split2_clean.csv" `
  --plots "E:/documents/Thesis code/plots/fft_axes_back" `
  --fs 100 --units g
"""

import argparse, re
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt, windows

# ------------------------ column mapping ------------------------
def normalize_and_map_columns(df):
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
    df.columns = [re.sub(r'\s+', '_', str(c).strip().lower()) for c in df.columns]
    def pick(cands): return next((c for c in cands if c in df.columns), None)
    m_time = pick(["timestamp_ms","time_ms","timestamp","time"])
    m_ax   = pick(["ax_g","x_acc","acc_x","ax","x"])
    m_ay   = pick(["ay_g","y_acc","acc_y","ay","y"])
    m_az   = pick(["az_g","z_acc","acc_z","az","z"])
    for need,var in zip(["timestamp_ms","ax_g","ay_g","az_g"],[m_time,m_ax,m_ay,m_az]):
        if var is None: raise ValueError(f"Missing {need}; got {list(df.columns)}")
    return df.rename(columns={m_time:"timestamp_ms",m_ax:"ax_g",m_ay:"ay_g",m_az:"az_g"})

# ------------------------ filters ------------------------
def butter_lpf(x, fc, fs, order=4):
    b, a = butter(order, fc / (0.5 * fs), btype="low")
    return filtfilt(b, a, x)

def kalman_1d(z, q=1e-5, r=1e-3):
    xhat = np.zeros_like(z, dtype=float); P = 1; x = float(z[0])
    for i, zi in enumerate(z):
        xpred = x; P += q
        K = P / (P + r)
        x = xpred + K * (zi - xpred)
        P = (1 - K) * P
        xhat[i] = x
    return xhat

# ------------------------ FFT plotting ------------------------
def plot_fft(sig, fs, axis_name, outdir):
    """Generate normalized and absolute FFT plots for one axis."""
    N = len(sig)
    if N < 16:
        print(f"⚠️ Not enough samples for FFT on {axis_name}.")
        return

    f = rfftfreq(N, 1/fs)
    X = np.abs(rfft(sig * windows.hann(N)))
    X_norm = X / (np.max(X) + 1e-12)

    # ----- Normalized FFT -----
    plt.figure(figsize=(10,5))
    plt.plot(f, X_norm, color="tab:blue", lw=1.5)
    plt.title(f"FFT Spectrum (Normalized) – {axis_name} Axis", fontsize=12)
    plt.xlabel("Frequency (Hz)", fontsize=10)
    plt.ylabel("Normalized Amplitude", fontsize=10)
    plt.xlim(0, 5)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.6)
    plt.tight_layout()
    plt.savefig(outdir / f"fft_{axis_name.lower()}_norm.png", dpi=150)
    plt.close()

    # ----- Absolute FFT -----
    plt.figure(figsize=(10,5))
    plt.plot(f, X, color="tab:orange", lw=1.5)
    plt.title(f"FFT Spectrum (Absolute Amplitude) – {axis_name} Axis", fontsize=12)
    plt.xlabel("Frequency (Hz)", fontsize=10)
    plt.ylabel("Amplitude (a.u.)", fontsize=10)
    plt.xlim(0, 5)
    plt.grid(True, alpha=0.6)
    plt.tight_layout()
    plt.savefig(outdir / f"fft_{axis_name.lower()}_abs.png", dpi=150)
    plt.close()

# ------------------------ main ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input cleaned CSV file")
    ap.add_argument("--plots", required=True, help="Output directory for FFT plots")
    ap.add_argument("--fs", type=float, default=100.0, help="Sampling frequency (Hz)")
    ap.add_argument("--units", choices=["g","mg"], default="g", help="Acceleration units")
    args = ap.parse_args()

    # Read + normalize
    df = pd.read_csv(args.inp)
    df = normalize_and_map_columns(df)
    if args.units == "mg":
        df[["ax_g","ay_g","az_g"]] /= 1000.0

    fs = args.fs

    # Apply filters
    ax = butter_lpf(kalman_1d(df["ax_g"].values), 10, fs)
    ay = butter_lpf(kalman_1d(df["ay_g"].values), 10, fs)
    az = butter_lpf(kalman_1d(df["az_g"].values), 10, fs)

    # Gravity removal
    gax = butter_lpf(ax, 0.3, fs)
    gay = butter_lpf(ay, 0.3, fs)
    gaz = butter_lpf(az, 0.3, fs)
    dax, day, daz = ax - gax, ay - gay, az - gaz

    # Create output directory
    outdir = Path(args.plots)
    outdir.mkdir(parents=True, exist_ok=True)

    # Generate both normalized and absolute FFT plots
    plot_fft(dax, fs, "X", outdir)
    plot_fft(day, fs, "Y", outdir)
    plot_fft(daz, fs, "Z", outdir)

    print(f"✅ Saved normalized and absolute FFT plots to: {outdir}")

if __name__ == "__main__":
    main()
