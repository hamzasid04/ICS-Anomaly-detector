#!/usr/bin/env python
"""
Train Isolation Forest on the data set from the ics_features.csv file that will produce scores and alerts.

Inputs will be taken from :
  data_features/ics_features.csv which will contain rolling mean and rolling std columns

Outputs will be in:
  outputs/scores_iforest.csv
  outputs/alerts_iforest.csv
  models/iforest.joblib
  outputs/iforest_scores.png
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data_features/ics_features.csv")
    ap.add_argument("--model_out", default="models/iforest.joblib")
    ap.add_argument("--scores_out", default="outputs/scores_iforest.csv")
    ap.add_argument("--alerts_out", default="outputs/alerts_iforest.csv")
    ap.add_argument("--plot_out", default="outputs/iforest_scores.png")
    ap.add_argument("--train_frac", type=float, default=0.60, help="early fraction of time used for training")
    ap.add_argument("--contamination", type=float, default=0.005, help="expected anomaly rate for the model fit")
    ap.add_argument("--percentile", type=float, default=0.99, help="score percentile used as alert threshold")
    args = ap.parse_args()

    # folders
    for p in [Path(args.model_out).parent, Path(args.scores_out).parent, Path(args.alerts_out).parent, Path(args.plot_out).parent]:
        p.mkdir(parents=True, exist_ok=True)

    # 1 load features
    feat = pd.read_csv(args.features, parse_dates=["timestamp"]).set_index("timestamp").sort_index()

    # 2 We pick the columns we actually want to feed the model: the rolling averages and rolling standard deviations for each sensor.
    # these columns include  "_roll_mean_" and "_roll_std_". 
    X_cols = [c for c in feat.columns if ("_roll_mean_" in c) or ("_roll_std_" in c)]
    if not X_cols:
        raise ValueError("No rolling mean or rolling std columns found in features")
    X = feat[X_cols].copy()
    X = X.fillna(method="ffill").fillna(method="bfill").fillna(0.0)

    # 3 time split
    # So here, we will split the time where our earlier timestapms are the timestamps that we deem not having anamolous readings and hence will use that dataset to train our isolation model off of and make it know what is normal
    # the later timestamps in our example will have anamoulous readings that we will use isolation model to take the readings out.
    cut = int(args.train_frac * len(X))
    if cut == 0 or cut >= len(X):
        raise ValueError("Bad train fraction or too few rows")
    # picks where “early” ends (default 60% early, 40% later).
    #If len(X) = 1,000 rows and train_frac = 0.60:
    #cut = int(0.60 * 1000) = 600
    #Train on rows 0..599 (the earlier 600 timestamps)
    #Test on rows 600..999 (the later 400 timestamps)
    X_train, X_test = X.iloc[:cut], X.iloc[cut:] 

    # 4 scale
    # Since pressures, and temp and other readings from other sensors will have different units and hence different readings.
    # We will scale it so that the readings are all uniform and are similar size so large numbers will not mess up our math and instead will have similar and consistent numberings.
    # we do that by taking the mean of readings like pressure or temp and then the standard deviation (std)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 5 fit Isolation Forest
    # Here we will train our isolation ofrest on the normal part. In other words the part where we have earlier timestamps
    clf = IsolationForest(
        n_estimators=256,
        contamination=args.contamination,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_s)

    # 6 score later window  higher score means more anomalous
    # So here, we discussed that we trained our isolation model off of the earlier timestamp part of dataset because we deemed that to be normal.
    # Now that it has been trained, we will use our model to detect and rank the anamoly that is present in the later timestamp  part of dataset. 
    # The higher the score = the more anamolous it is.
    scores = -clf.score_samples(X_test_s)
    scores_df = pd.DataFrame({"score": scores}, index=X_test.index)
    scores_df.to_csv(args.scores_out)

    # 7 threshold by percentile and build alerts
    # This is where we will have a bunch of scores or ranks that our isolation model will have scored. In these scores, we will choose the 1% of the most anomolous score to deem as our alert when trying to detecct anamoly
    thr = float(np.quantile(scores, args.percentile))
    alerts = scores_df[scores_df["score"] >= thr].copy()

    # reason  the standardized feature with largest absolute value at that time
    # this will tell you which feature wheather that be temp, pressure, etc. had deviated the most during the time of anamoly.
    X_test_std = pd.DataFrame(X_test_s, index=X_test.index, columns=X_cols)
    alerts["reason"] = X_test_std.loc[alerts.index].abs().idxmax(axis=1)
    alerts.to_csv(args.alerts_out)

    # 8 save model and scaler
    # we will save our isolation forest model that we trained for future use.
    # we will also save the scaler or the fitted StandardScaler which remembers the train means/standard deviations.
    # ex; scaled = (value − mean_trained) ÷ std_trained
    # here we save the mean_trained and std_trained values cuz we know that is the standard and normal values we should be comparing it with and then when isnerting new values, we resuse the standard mean and std trained values.
    joblib.dump({"model": clf, "scaler": scaler, "features": X_cols}, args.model_out)

    # 9 plot timeline
    #We create and save a PNG graph of the score over time, with a dashed line at the threshold and dots where alerts happened.
    plt.figure(figsize=(11, 4))
    plt.plot(scores_df.index, scores_df["score"])
    plt.axhline(thr, linestyle="--")
    plt.plot(alerts.index, alerts["score"], linestyle="", marker="o")
    plt.title(f"Isolation Forest scores  threshold at {int(args.percentile*100)}th percentile")
    plt.xlabel("time")
    plt.ylabel("score")
    plt.tight_layout()
    plt.savefig(args.plot_out, dpi=150)
    plt.close()

    # 10 acceptance prints
    #These are console printouts
    print(f"Rows train test  {len(X_train)}  {len(X_test)}")
    print(f"Score threshold  {thr:.4f}")
    print(f"Alerts count     {len(alerts)}")
    print(f"Wrote            {args.scores_out}")
    print(f"Wrote            {args.alerts_out}")
    print(f"Wrote            {args.model_out}")
    print(f"Wrote            {args.plot_out}")

if __name__ == "__main__":
    main()
