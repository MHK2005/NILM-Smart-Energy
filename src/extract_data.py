"""
Phase 1: dataset inspection and extraction.

Run:  python -m src.extract_data

1. Finds the 1-minute CSV file(s) under data/raw/.
2. Checks that the needed columns exist (dataid, localminute, grid, solar,
   and the appliance columns) and reports any that are missing.
3. Prints and saves a per-home summary of every home in the file:
   number of readings, time range, and for every appliance column the
   number of valid readings and the share of readings above a rough
   activity level (only used for inspection, NOT as the ON threshold).
4. Extracts the selected homes (development + unseen) into
   data/processed/selected_raw.csv.gz with timestamps converted to UTC.
   The original dataset files are never modified.
"""
import pandas as pd

from src.utils import (APPLIANCES, DEV_HOMES, UNSEEN_HOMES, GRID_COL, SOLAR_COLS,
                       PROCESSED_DIR, RESULTS_DIR, ensure_dirs, find_raw_csv)

ID_COL, TIME_COL = "dataid", "localminute"
INSPECT_LEVEL_KW = 0.1   # rough activity level, for the inspection table only
CHUNK = 500_000


def map_columns(header):
    """Match expected column names to the file header (case-insensitive)."""
    lower = {c.lower().strip(): c for c in header}
    wanted = [ID_COL, TIME_COL, GRID_COL] + SOLAR_COLS + APPLIANCES
    mapping, missing = {}, []
    for w in wanted:
        if w in lower:
            mapping[w] = lower[w]
        else:
            missing.append(w)
    return mapping, missing


def main():
    ensure_dirs()
    files = find_raw_csv()
    print("Dataset files found:")
    for f in files:
        print(f"  {f}  ({f.stat().st_size / 1e9:.2f} GB)")
    src = files[0]
    if len(files) > 1:
        print(f"Using {src.name}. This project uses the Austin file only.")

    header = pd.read_csv(src, nrows=0).columns.tolist()
    print(f"\n{len(header)} columns: {', '.join(header)}")
    mapping, missing = map_columns(header)
    if missing:
        print(f"WARNING: columns not found and need manual mapping: {missing}")
    for req in [ID_COL, TIME_COL, GRID_COL]:
        if req not in mapping:
            raise SystemExit(f"Required column '{req}' missing. Stop.")

    power_cols = [c for c in header if c not in (ID_COL, TIME_COL)]
    selected = set(DEV_HOMES + UNSEEN_HOMES)
    keep_cols = [mapping[c] for c in [ID_COL, TIME_COL, GRID_COL] + SOLAR_COLS + APPLIANCES
                 if c in mapping]

    dtypes = {c: "float32" for c in power_cols}
    first, last, rows, valid, active, parts, n_rows = [], [], [], [], [], [], 0
    for ch in pd.read_csv(src, chunksize=CHUNK, dtype=dtypes):
        n_rows += len(ch)
        g = ch.groupby(ID_COL)
        rows.append(g.size())
        first.append(g[TIME_COL].min())
        last.append(g[TIME_COL].max())
        valid.append(g[power_cols].count())
        active.append((ch[power_cols] > INSPECT_LEVEL_KW).groupby(ch[ID_COL]).sum())
        parts.append(ch.loc[ch[ID_COL].isin(selected), keep_cols])
        print(f"  read {n_rows:,} rows", end="\r")
    print()
    summ = pd.DataFrame({
        "rows": pd.concat(rows).groupby(level=0).sum(),
        "first": pd.concat(first).groupby(level=0).min(),
        "last": pd.concat(last).groupby(level=0).max(),
    })
    summ = summ.join(pd.concat(valid).groupby(level=0).sum().add_suffix("_valid"))
    summ = summ.join(pd.concat(active).groupby(level=0).sum().add_suffix("_active"))

    # ---- summary table of all homes ----
    summ.index.name = "home"
    summ.to_csv(RESULTS_DIR / "dataset_summary_all_homes.csv")
    print(f"\nTotal rows: {n_rows:,}   Homes: {len(summ)}")
    print("Data types: dataid int, localminute text with UTC offset (-06 CST / -05 CDT), "
          "power columns float in kW.")
    print("\nAppliance availability (homes with valid readings / homes with activity "
          f"> {INSPECT_LEVEL_KW} kW):")
    for a in APPLIANCES:
        v = summ[a + "_valid"].astype(int)
        act = summ[a + "_active"].astype(int)
        print(f"  {a:14s} metered in {int((v > 0).sum()):2d} homes, active in "
              f"{int((act > 0).sum()):2d} homes")
    print("\nHomes where air1, drye1 and car1 are all active:",
          [h for h in summ.index if all(summ.loc[h, a + "_active"] > 0
                                        for a in ["air1", "drye1", "car1"])])

    # ---- extract selected homes ----
    df = pd.concat(parts, ignore_index=True)
    df = df.rename(columns={v: k for k, v in mapping.items()})
    # The file mixes -06 and -05 offsets (daylight saving). UTC gives one
    # continuous minute timeline without duplicated or missing hours.
    df["ts_utc"] = pd.to_datetime(df[TIME_COL], utc=True, format="%Y-%m-%d %H:%M:%S%z")
    df = df.drop(columns=TIME_COL)
    out = PROCESSED_DIR / "selected_raw.csv.gz"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df):,} rows for homes {sorted(selected)} -> {out}")

    sel = summ.loc[sorted(selected), ["rows", "first", "last"] +
                   [a + "_valid" for a in APPLIANCES]]
    print("\nSelected homes:")
    print(sel.to_string())


if __name__ == "__main__":
    main()
