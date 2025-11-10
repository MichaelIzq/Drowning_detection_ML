#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Folder containing feature CSVs")
    ap.add_argument("--pattern", default="*_features*.csv",
                    help="Filename pattern to match (default: *_features*.csv)")
    ap.add_argument("--recursive", action="store_true",
                    help="Recurse into subfolders")
    ap.add_argument("--out", default="all_strokes_combined.csv",
                    help="Output CSV filename")
    ap.add_argument("--infer_label_from_filename", action="store_true",
                    help="If a CSV has no 'label' column, infer from filename prefix before first '_'")
    args = ap.parse_args()

    feature_dir = Path(args.dir)
    if not feature_dir.exists():
        raise FileNotFoundError(f"Folder not found: {feature_dir}")

    # Find files
    files = (feature_dir.rglob(args.pattern) if args.recursive
             else feature_dir.glob(args.pattern))
    files = sorted(files)

    print(f"🔎 Searching in: {feature_dir.resolve()}")
    print(f"🔎 Pattern: {args.pattern} (recursive={args.recursive})")
    if not files:
        print("❌ No files matched. Tips:")
        print("  • Check the path and spelling (use forward slashes or raw string).")
        print("  • Try a broader pattern like *.csv")
        print("  • Are your files in a subfolder? Add --recursive")
        return

    print("✅ Found files:")
    for f in files:
        print("  -", f.name)

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Clean unnamed columns if any
        df = df.loc[:, ~df.columns.str.contains("^Unnamed", na=False)]
        # Ensure label exists (optionally infer from filename)
        if "label" not in df.columns:
            if args.infer_label_from_filename:
                label = f.stem.split("_")[0]  # e.g., freestyle_features4 -> freestyle
                df["label"] = label
                print(f"ℹ️  Added label='{label}' from filename for {f.name}")
            else:
                raise ValueError(f"'label' column missing in {f.name}. "
                                 f"Add --infer_label_from_filename or fix the file.")
        # Drop session_id if present
        if "session_id" in df.columns:
            df = df.drop(columns=["session_id"])
        dfs.append(df)

    if not dfs:
        print("❌ No valid dataframes loaded.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    # Optional shuffle for training
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    out_path = feature_dir / args.out
    combined.to_csv(out_path, index=False)
    print(f"\n🎉 Combined dataset saved → {out_path}")
    print(f"   Rows: {len(combined)}   Columns: {len(combined.columns)}")
    print("   Labels:", sorted(combined['label'].unique()))

if __name__ == "__main__":
    main()

"""
& "E:/Program Files/Python/python.exe" "E:/documents/Thesis code/combine_all_features.py" `
  --dir "E:/documents/Thesis code/feature_test" `
  --pattern "*features*.csv" `
  --out "combined_testing_sample.csv"
  """