# app.py
import io
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="ICS Anomaly Demo", layout="wide")

# ---------------- parameters you can tweak quickly ----------------
DEFAULT_FREQ_MIN = 1        # resample to 1 minute
MAX_FFILL_MIN = 5           # forward fill gaps up to 5 minutes
ROLL_MIN = 15               # rolling window minutes for mean and std
TRAIN_FRAC = 0.60           # early fraction used as normal for IF and baseline
IF_CONTAM = 0.01            # expected fraction anomalous for IF
IF_PERCENTILE = 0.99        # top scores kept as alerts
Z_THRESHOLD = 3.0           # baseline alert threshold on max |z|
MODEL_PATH = Path("models/iforest.joblib")
# ------------------------------------------------------------------

st.title("ICS anomaly detector")

st.sidebar.header("Pipeline settings")
# Resample minutes
freq_min = st.sidebar.number_input(
    "Resample minutes",
    min_value=1, max_value=60, value=DEFAULT_FREQ_MIN,
    help="Example: If set to 1, you keep one point per minute. If set to 5, 10:00..10:04 collapse into one averaged point at 10:05."
)
st.sidebar.caption("Keep one data point every N minutes. Smaller keeps more detail. Larger smooths the line.")

# Max gap forward fill minutes
ffill_min = st.sidebar.number_input(
    "Max gap forward fill minutes",
    min_value=0, max_value=120, value=MAX_FFILL_MIN,
    help="""Example: 10:00 pressure is 50
10:01 pressure is 51
10:02 missing
10:03 missing
10:04 pressure is 52

If max gap forward fill is 2 minutes
We are allowed to fill at most two missing minutes.
So 10:02 becomes 51 and 10:03 becomes 51.
At 10:04 we have 52 again.

If max gap forward fill is 1 minute
We can fill only one minute.
So 10:02 becomes 51 and 10:03 stays missing"""
)
st.sidebar.caption("Basically how many minutes of missing data are we able to tolerate.")
st.sidebar.caption("We do that by copying the last value forward for short gaps so the line doesn’t break.")

# Rolling window minutes
roll_min = st.sidebar.number_input(
    "Rolling window minutes",
    min_value=3, max_value=120, value=ROLL_MIN,
    help="Example: 15 → each time uses the last 15 minutes to smooth; 5 = faster but noisier, 30 = smoother but slower"
)
st.sidebar.caption("Determines how much the graph wiggles")
st.sidebar.caption("Look back this many minutes to compute a rolling average and spread. Smaller = faster; larger = smoother.")

# Train fraction early window
train_frac = st.sidebar.slider(
    "Train fraction (Which part of dataset you want the model to train off of)",
    min_value=0.10, max_value=0.90, value=TRAIN_FRAC, step=0.05,
    help="Example: 0.60 = first 60% of data is 'normal' which is what we will train our model off of (train), last 40% is anaomlous and our trained model will score that part of data scored. If the file’s start is messy, try 0.50."
)
st.sidebar.caption("Use the first part of the file as normal which is the part that we will train our ML model off of. The rest is the anamolous part of the data that our ML model will score.")

# Force Isolation Forest if model bundle exists
use_iforest = st.sidebar.checkbox(
    "Force Isolation Forest if model bundle exists",
    value=True,
    help="If models/iforest.joblib exists, use it. Uncheck to run the simple Z-score baseline instead."
)
st.sidebar.caption("Use saved ML model, or fall back to rule-based baseline.")

# IF percentile for alerts
if_percentile = st.sidebar.slider(
    "Highest anomaly percentile for alerts (Threshold value setter)",
    min_value=0.90, max_value=0.999, value=IF_PERCENTILE, step=0.001,
    help="""Here, our model looks at all the scores, sorts them from smallest to biggest,
and picks the value where the chosen percentile of scores are below it
and the remaining top part are above it. For example, 0.99 means 99 percent below
and the top 1 percent above. This becomes the threshold value.
Anything above this threshold is treated as an alert and shown on the graph and in the table. """
)
st.sidebar.caption("Keep the top X% highest scores as alerts. Higher = fewer alerts. Lower = more alerts.")

with st.sidebar.expander("Advanced"):
    if_contam = st.slider(
        "Assumed anomaly rate in training data (%)",
        min_value=0.001, max_value=0.20, value=IF_CONTAM, step=0.001,
        help="Example: 0.01 means you expect ~1% anomalies in the TRAIN window; model gets stricter with smaller values."
    )

    max_jump_pct = st.slider(
        "Max percent jump rule",
        min_value=5.0, max_value=200.0, value=50.0, step=5.0,
        help="Example: 50 means if any sensor changes more than 50% between steps, flag a rule-based alert."
    )

# Z baseline threshold
z_threshold = st.sidebar.number_input(
    "Z baseline threshold (rule-based alert and not ML)",
    min_value=0.0,
    value=Z_THRESHOLD,
    step=0.1,
    format="%.2f",
    help=(
        "Example: 3.0 means alert when the worst sensor is ≥ 3 standard deviations from normal. "
        "Raise to 3.5–4.0 to be stricter."
    ),
)
st.sidebar.caption("A rule-based detector. No ML. Alert when the largest absolute z value reaches this number.")
st.sidebar.caption("Instead of using the ML (Isolation Forest) to learn a threshold, you can manually set one yourself here.")

uploaded = st.file_uploader("Upload a CSV with a time column called 'timestamp' and numeric sensor columns", type=["csv"])

def ensure_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        # try to detect first datetime like column
        for c in df.columns:
            try:
                ts = pd.to_datetime(df[c], errors="raise")
                df = df.drop(columns=[c]).assign(timestamp=ts)
                break
            except Exception:
                continue
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    return df

def clean_df(df: pd.DataFrame, freq_min: int, ffill_min: int) -> pd.DataFrame:
    df = ensure_timestamp_index(df)
    # keep numeric columns
    num = df.select_dtypes("number")
    if num.empty:
        raise ValueError("No numeric columns found after parsing timestamp")

    # resample to fixed rate
    rs = num.resample(f"{freq_min}min").mean()

    # convert minutes to "how many rows can we fill"
    if ffill_min > 0 and freq_min > 0:
        max_steps = ffill_min // freq_min   # integer division
        if max_steps >= 1:
            rs = rs.ffill(limit=max_steps)
        # if max_steps == 0: do not forward-fill at all
    # if ffill_min == 0: also no forward fill

    return rs
def make_features(df: pd.DataFrame, roll_min:int, freq_min:int) -> pd.DataFrame:
    win = max(1, int(roll_min / freq_min))
    out = df.copy()
    for c in df.columns:
        m = df[c].rolling(win, min_periods=win).mean()
        s = df[c].rolling(win, min_periods=win).std()
        out[f"{c}_roll_mean_{roll_min}m"] = m
        out[f"{c}_roll_std_{roll_min}m"]  = s
        out[f"{c}_roll_z_{roll_min}m"]    = (df[c] - m) / s
    return out.dropna()



def apply_step_rule(df_clean: pd.DataFrame, max_jump_pct: float) -> pd.Series:
    """
    Simple rule: if any sensor changes by more than max_jump_pct (e.g. 50)
    between this row and the previous row, mark this timestamp as a rule hit.
    Returns a boolean Series indexed by time.
    """
    # percent change vs previous row, in %
    pct_change = df_clean.pct_change().abs() * 100.0
    # row is True if any column exceeds threshold
    rule_hit = (pct_change > max_jump_pct).any(axis=1)
    return rule_hit


def add_top3_z(alerts: pd.DataFrame, feat: pd.DataFrame, roll_min: int) -> pd.DataFrame:
    """
    For each alert timestamp, find the top 3 sensors by |rolling z|.
    Adds two columns:
      - top3_features: "pressure_psig, temperature_f, flow_m3h"
      - top3_z: "4.20, 3.80, 3.10"
    """
    if alerts.empty:
        alerts = alerts.copy()
        alerts["top3_features"] = ""
        alerts["top3_z"] = ""
        return alerts

    zcols = [c for c in feat.columns if f"_roll_z_{roll_min}m" in c]
    Z = feat[zcols]

    features_list = []
    zvalues_list = []

    for ts in alerts.index:
        if ts not in Z.index:
            features_list.append("")
            zvalues_list.append("")
            continue

        row = Z.loc[ts].abs().sort_values(ascending=False).head(3)
        # "pressure_psig_roll_z_15m" → "pressure_psig"
        feat_names = [name.split("_roll_z_")[0] for name in row.index]
        features_list.append(", ".join(feat_names))
        zvalues_list.append(", ".join(f"{v:.2f}" for v in row.values))

    alerts = alerts.copy()
    alerts["top3_features"] = features_list
    alerts["top3_z"] = zvalues_list
    return alerts

def human_feature_name(raw: str) -> str:
    """
    Turn a technical feature name into a friendlier description.
    Examples:
      "pressure_psig_roll_z_15m" → "pressure (distance from normal over last 15 minutes)"
      "temperature_f" → "temperature"
    """
    if not isinstance(raw, str):
        return "unknown feature"

    base = raw

    # Strip rolling suffixes
    for token in ["_roll_z_", "_roll_std_", "_roll_mean_"]:
        if token in base:
            base = base.split(token)[0]
            break

    # Map common sensor names to nicer labels
    mapping = {
        "pressure_psig": "pressure",
        "temperature_f": "temperature",
        "flow_m3h": "flow rate",
    }
    nice = mapping.get(base, base.replace("_", " "))

    # Add a small hint if it was a z or std feature
    if "_roll_z_" in raw:
        return f"{nice} (distance from normal over the window)"
    elif "_roll_std_" in raw:
        return f"{nice} (how unstable it has been over the window)"
    elif "_roll_mean_" in raw:
        return f"{nice} (average over the window)"
    else:
        return nice
def build_incidents_markdown(alerts: pd.DataFrame, threshold: float, max_incidents: int = 3) -> str:
    """
    Build a small incidents.md text from the top alerts.
    Each incident becomes one bullet line in plain English.
    """
    if alerts.empty:
        return "# Incidents\n\nNo alerts were generated for this run.\n"

    lines = ["# Incidents", ""]

    # Take the most abnormal rows first
    subset = alerts.sort_values("score", ascending=False).head(max_incidents)

    for ts, row in subset.iterrows():
        # 1) Time
        if isinstance(ts, pd.Timestamp):
            time_str = ts.strftime("%Y-%m-%d %H:%M")
        else:
            time_str = str(ts)

        # 2) Score and threshold
        score = float(row.get("score", float("nan")))

        # 3) Main reason (feature)
        reason_raw = str(row.get("reason", "unknown"))
        reason_nice = human_feature_name(reason_raw)

        # 4) Rule fired or not
        rule_hit = bool(row.get("rule_hit", False))
        rule_text = " The simple percent jump rule also fired at this time." if rule_hit else ""

        # 5) Top 3 features and z values
        top3f_raw = str(row.get("top3_features", "") or "")
        top3z_raw = str(row.get("top3_z", "") or "")

        top3f = [p.strip() for p in top3f_raw.split(",") if p.strip()]
        top3z = [p.strip() for p in top3z_raw.split(",") if p.strip()]

        if top3f and top3z:
            triples = []
            for f, z in zip(top3f, top3z):
                triples.append(f"{human_feature_name(f)} (~{z} standard deviations)")
            # Join like "temperature (~3.26…), pressure (~2.74…), flow rate (~2.21…)"
            top3_text = " The top three abnormal features were " + ", ".join(triples) + "."
        else:
            top3_text = ""

        # Final sentence for this incident
        line = (
            f"- At {time_str}, the anomaly score was {score:.3f} "
            f"(above the {threshold:.3f} threshold), and the main driver was {reason_nice}."
            f"{rule_text}{top3_text}"
        )
        lines.append(line)

    lines.append("")  # trailing blank line
    return "\n".join(lines)

def build_top3_bullets(row) -> str:
    """
    Turn top3_features and top3_z into a small bullet list string.
    Example:
      - pressure_psig_roll_z_15m (z=4.10)
      - temperature_f_roll_z_15m (z=3.20)
      - flow_m3h_roll_z_15m (z=2.50)
    """
    feats = str(row.get("top3_features", "") or "").split(",")
    zs = str(row.get("top3_z", "") or "").split(",")

    lines = []
    for f, z in zip(feats, zs):
        f = f.strip()
        z = z.strip()
        if not f:
            continue
        if z:
            lines.append(f"- {f} (z={z})")
        else:
            lines.append(f"- {f}")
    return "\n".join(lines)



def run_iforest(feat: pd.DataFrame, train_frac: float, contam: float, percentile: float):
    # pick rolling mean and std columns only
    X_cols = [c for c in feat.columns if "_roll_mean_" in c or "_roll_std_" in c]
    X = feat[X_cols].copy()
    # time split
    cut = int(train_frac * len(X))
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    if len(X_test) == 0:
        raise ValueError("Not enough rows after time split for test window")
    # scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    # train and score
    clf = IsolationForest(n_estimators=256, contamination=contam, random_state=42, n_jobs=-1)
    clf.fit(X_train_s)
    scores = -clf.score_samples(X_test_s)  # larger is more unusual
    scores_df = pd.DataFrame({"score": scores}, index=X_test.index)
    thr = float(np.quantile(scores, percentile))
    alerts = scores_df[scores_df["score"] >= thr].copy()
    # reason as most deviant standardized feature
    X_test_std = pd.DataFrame(X_test_s, index=X_test.index, columns=X_cols)
    alerts["reason"] = X_test_std.loc[alerts.index].abs().idxmax(axis=1)
    bundle = {"model": clf, "scaler": scaler, "features": X_cols}
    return scores_df, alerts, thr, bundle

def run_z_baseline(df_clean: pd.DataFrame, train_frac: float, z_thr: float, roll_min:int, freq_min:int):
    # use rolling z if available, else global z from early window
    feat = make_features(df_clean, roll_min=roll_min, freq_min=freq_min)
    zcols = [c for c in feat.columns if "_roll_z_" in c]
    Z = feat[zcols]
    score = Z.abs().max(axis=1)
    scores_df = pd.DataFrame({"score": score})
    thr = z_thr
    alerts = scores_df[scores_df["score"] >= thr].copy()
    # reason is the column with largest |z|
    alerts["reason"] = Z.loc[alerts.index].abs().idxmax(axis=1)
    return scores_df, alerts, thr

def draw_timeline(scores: pd.DataFrame, alerts: pd.DataFrame, thr: float, title: str):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(scores.index, scores["score"])
    ax.axhline(thr, linestyle="--")
    if not alerts.empty:
        ax.plot(alerts.index, alerts["score"], linestyle="", marker="o")
    ax.set_title(title)
    ax.set_xlabel("time"); ax.set_ylabel("score")
    st.pyplot(fig)

if uploaded is None:
    st.info("Upload a CSV to begin")
else:
    try:
        raw = pd.read_csv(uploaded)
        df_clean = clean_df(raw, freq_min=freq_min, ffill_min=ffill_min)
        st.success(f"Loaded and cleaned. Rows {len(df_clean)}")
        st.dataframe(df_clean.head(8))
        
        # apply simple percent-jump rule on the cleaned data
        rule_hits = apply_step_rule(df_clean, max_jump_pct=max_jump_pct)


        # choose engine
        model_available = MODEL_PATH.exists()
        method = "IForest" if (use_iforest and model_available) else ("IForest" if use_iforest else "Z")
        st.write(f"Detector selected: {'Isolation Forest' if method=='IForest' else 'Z baseline'}")

        # if a saved model exists and we want to force IF, we still need features from the current file
        feat = make_features(df_clean, roll_min=roll_min, freq_min=freq_min)

        if method == "IForest":
            # use saved model if present, else train on the fly
            if model_available:
                bundle = joblib.load(MODEL_PATH)
                X_cols = bundle["features"]
                missing = [c for c in X_cols if c not in feat.columns]
                if missing:
                    st.warning("Uploaded data missing some trained features. Training a new temporary model instead.")
                    scores_df, alerts, thr, _ = run_iforest(feat, train_frac, if_contam, if_percentile)
                else:
                    X = feat[X_cols].copy()
                    cut = int(train_frac * len(X))
                    X_test = X.iloc[cut:]
                    if len(X_test) == 0:
                        raise ValueError("Not enough rows after time split for test window")
                    X_test_s = bundle["scaler"].transform(X_test)
                    scores = -bundle["model"].score_samples(X_test_s)
                    scores_df = pd.DataFrame({"score": scores}, index=X_test.index)
                    thr = float(np.quantile(scores, if_percentile))
                    # reason from standardized X_test
                    X_test_std = pd.DataFrame(X_test_s, index=X_test.index, columns=X_cols)
                    alerts = scores_df[scores_df["score"] >= thr].copy()
                    alerts["reason"] = X_test_std.loc[alerts.index].abs().idxmax(axis=1)
            else:
                scores_df, alerts, thr, _ = run_iforest(feat, train_frac, if_contam, if_percentile)
        else:
            scores_df, alerts, thr = run_z_baseline(df_clean, train_frac, z_threshold, roll_min, freq_min)

        # add rule flags and top-3 z explanation to the alerts
        scores_df = scores_df.copy()
        scores_df["rule_hit"] = rule_hits.reindex(scores_df.index).fillna(False)

        alerts = alerts.copy()
        alerts["rule_hit"] = rule_hits.reindex(alerts.index).fillna(False)
        alerts = add_top3_z(alerts, feat, roll_min=roll_min)

        st.subheader("Score timeline")
        draw_timeline(scores_df, alerts, thr, "Anomaly score over time  dots are alerts")

        st.subheader("Alerts")
        st.write(f"Alerts count: {len(alerts)}")
        st.write(f"Rule-backed: {int(alerts.get('rule_hit', False).sum())}")
        st.write(f"Threshold our model is measuring with: {thr:.3f}")

        # Prepare a nicer view of the alerts table
        alerts_view = alerts.sort_values("score", ascending=False).head(200).reset_index()

        # Build bullet list column if top3 data exists
        if "top3_features" in alerts_view.columns and "top3_z" in alerts_view.columns:
            alerts_view["Top 3 (bullets)"] = alerts_view.apply(build_top3_bullets, axis=1)
        else:
            alerts_view["Top 3 (bullets)"] = ""

        # Rename the index column to 'timestamp' if needed
        if "timestamp" in alerts_view.columns:
            time_col = "timestamp"
        elif "index" in alerts_view.columns:
            time_col = "index"
        else:
            time_col = alerts_view.columns[0]  # fallback
        # remove the top 3 bullet points that combines both columns into one for simplicity sake
        alerts_view = alerts_view.drop(columns=["Top 3 (bullets)"], errors="ignore")
        st.dataframe(
            alerts_view,
            column_config={
                time_col: st.column_config.DatetimeColumn(
                    "Time",
                    help="Timestamp of the resampled point (usually one row per minute).",
                ),
                "score": st.column_config.NumberColumn(
                    "Anomaly score",
                    help="Higher = more unusual. Alerts are rows with score above the threshold.",
                ),
                "reason": st.column_config.TextColumn(
                    "Main feature (raw)",
                    help="Feature that looked most abnormal at that time (technical name).",
                ),
                "rule_hit": st.column_config.CheckboxColumn(
                    "Rule-backed",
                    help="True if the physical rule which is a certain percentage of change detected from current minute to prev. minute (e.g 50% increase or decrease from current minute to prev minute) ",
                ),
                "top3_features": st.column_config.TextColumn(
                    "Top 3 features (raw)",
                    help="""Technical names of the three most abnormal features at that time. These can be names of: how high/low the certain reading is compared to normal (z value) OR how unstable it is compared to normal (std). Ex; \n\n
                    That is the z value columns, like pressure_psig_roll_z_15m.
                   - So normal pressure for the last 15 minutes is about 100
                   - Right now pressure is 130
                   - That is maybe 3 standard deviations away
                    This means:
                    Pressure is very high or very low compared to its usual level.

                    That is the std columns, like pressure_psig_roll_std_15m.
                    Standard deviation is about how much the values jump around inside the window.
                    Example:

                    Case A: last 15 minutes were 100, 101, 99, 100, 101
                    Very stable, low std

                    Case B: last 15 minutes were 80, 120, 90, 130, 70
                    Very jumpy, high std
                    """,
                ),
                "top3_z": st.column_config.TextColumn(
                    "Top 3 z-scores",
                    help="How many standard deviations from normal each of the top 3 features is in repective order to Top 3 features presented. We are using Z scores which is z = (value − mean_normal) / std_normal so that all readings of different sensors with different units and measurements are uniform and comparable to one another",
                ),
            },
            use_container_width=True,
        )

        # Build incidents.md from the current alerts (all of them)
        incidents_md = build_incidents_markdown(alerts, thr, max_incidents=len(alerts))

        # download buttons
        csv_scores = scores_df.reset_index().to_csv(index=False).encode()
        csv_alerts = alerts.reset_index().to_csv(index=False).encode()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button(
                "Download scores CSV",
                data=csv_scores,
                file_name="scores.csv",
                mime="text/csv",
            )

        with col2:
            st.download_button(
                "Download alerts CSV",
                data=csv_alerts,
                file_name="alerts.csv",
                mime="text/csv",
            )

        with col3:
            st.download_button(
                "Download incidents.md",
                data=incidents_md.encode("utf-8"),
                file_name="incidents.md",
                mime="text/markdown",
            )

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)
