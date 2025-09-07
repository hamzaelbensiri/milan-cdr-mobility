from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import csv

def sniff_delim(path: Path) -> str:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        sample = fh.read(4096)
    try:
        return csv.Sniffer().sniff(sample).delimiter
    except Exception:
        return ","

def read_daily_csv(path: Path) -> pd.DataFrame:
    sep = sniff_delim(path)
    df = pd.read_csv(path, sep=sep, low_memory=False)
    rename_map = {
        "datetime": "time_interval",
        "time": "time_interval",
        "CellID": "square_id",
        "cellId": "square_id",
        "smsin": "sms_in",
        "smsout": "sms_out",
        "callin": "call_in",
        "callout": "call_out",
    }
    return df.rename(columns=rename_map)

def load_all_days(data_dir: Path, pattern="sms-call-internet-mi-*.csv"):
    files = sorted((data_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern} in {data_dir}")
    return files

STANDARD_NUM = ["sms_in","sms_out","call_in","call_out","internet"]

def coerce_and_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df["time_interval"] = pd.to_datetime(df["time_interval"], errors="coerce")
    df["square_id"] = pd.to_numeric(df["square_id"], errors="coerce").astype("Int64")
    for c in STANDARD_NUM:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0
    grp = (df.dropna(subset=["time_interval","square_id"])
             .groupby(["time_interval","square_id"], as_index=False)[STANDARD_NUM]
             .sum())
    grp["square_id"] = grp["square_id"].astype("int32")
    for c in STANDARD_NUM:
        grp[c] = grp[c].astype("float32" if c != "internet" else "float64")
    return grp[["square_id","time_interval"] + STANDARD_NUM]

def build_master_parquet(files, out_parquet: Path) -> pd.DataFrame:
    chunks = []
    for f in files:
        df = read_daily_csv(f)
        grp = coerce_and_aggregate(df)
        chunks.append(grp)
    cdr = pd.concat(chunks, ignore_index=True)
    cdr.sort_values(["square_id","time_interval"], inplace=True)
    cdr.reset_index(drop=True, inplace=True)
    out_parquet.parent.mkdir(exist_ok=True, parents=True)
    cdr.to_parquet(out_parquet, index=False)
    return cdr

def features_weekday_weekend_shape(cdr: pd.DataFrame, value_col="internet"):
    cdr = cdr.copy()
    cdr["dow"] = cdr["time_interval"].dt.dayofweek
    cdr["hour"] = cdr["time_interval"].dt.hour

    grp = (cdr.groupby(["square_id","dow","hour"], observed=True)[value_col]
             .mean().reset_index())

    wkday = (grp[grp["dow"].between(0,4)]
             .groupby(["square_id","hour"], observed=True)[value_col]
             .mean().unstack("hour").fillna(0.0))
    wkend = (grp[grp["dow"].between(5,6)]
             .groupby(["square_id","hour"], observed=True)[value_col]
             .mean().unstack("hour").fillna(0.0))

    for h in range(24):
        if h not in wkday.columns: wkday[h] = 0.0
        if h not in wkend.columns: wkend[h] = 0.0
    wkday = wkday[sorted(wkday.columns)]
    wkend = wkend[sorted(wkend.columns)]

    def _row_norm(df):
        arr = df.to_numpy(dtype=float)
        sums = arr.sum(axis=1, keepdims=True)
        sums[sums==0] = 1.0
        return arr / sums

    wkday_norm = _row_norm(wkday)
    wkend_norm = _row_norm(wkend)
    X_shape = np.hstack([wkday_norm, wkend_norm])

    avg_vol = grp.groupby("square_id")[value_col].mean().reindex(wkday.index).fillna(0.0).to_numpy()
    log_avg_vol = np.log1p(avg_vol).reshape(-1,1)

    X = np.hstack([X_shape, log_avg_vol])
    cell_ids = wkday.index.to_numpy()
    return X, cell_ids, wkday, wkend, avg_vol

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

def choose_k_and_fit(X, k_min=3, k_max=8, random_state=42):
    results, models = [], {}
    for k in range(k_min, k_max+1):
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        db  = davies_bouldin_score(X, labels)
        results.append((k, sil, db))
        models[k] = (km, labels)
    results_sorted = sorted(results, key=lambda t:(-t[1], t[2]))
    best_k, best_sil, best_db = results_sorted[0]
    km, labels = models[best_k]
    return best_k, {"silhouette":best_sil, "db":best_db}, km, labels, results

def export_artifacts(root: Path, cdr: pd.DataFrame, labels_df: pd.DataFrame, centroids_csv: Path|None=None):
    out_dir = root / "artifacts"
    out_dir.mkdir(exist_ok=True)

    clusters = labels_df.copy()
    if centroids_csv and Path(centroids_csv).exists():
        xy = pd.read_csv(centroids_csv)
        clusters = clusters.merge(xy, on="square_id", how="left")
    clusters.to_csv(out_dir / "clusters_by_cell.csv", index=False)

    hourly_by_cell = (cdr.assign(hour=cdr["time_interval"].dt.hour)
                        .groupby(["square_id","hour"])["internet"]
                        .mean().reset_index()
                        .rename(columns={"internet":"avg_internet_bytes"}))
    hourly_by_cell = hourly_by_cell.merge(labels_df, on="square_id", how="left")
    if centroids_csv and Path(centroids_csv).exists():
        hourly_by_cell = hourly_by_cell.merge(pd.read_csv(centroids_csv), on="square_id", how="left")
    hourly_by_cell.to_csv(out_dir / "hourly_by_cell_with_cluster.csv", index=False)

    hourly_city = (cdr.assign(hour=cdr["time_interval"].dt.hour)
                     .groupby("hour")["internet"]
                     .mean().reset_index()
                     .rename(columns={"internet":"avg_internet_bytes"}))
    hourly_city.to_csv(out_dir / "hourly_city_totals.csv", index=False)
    return out_dir

