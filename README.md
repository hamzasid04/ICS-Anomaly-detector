# ICS Anomaly Detector

ICS Anomaly Detector is a small demo app that shows how to spot unusual behavior in industrial sensor data using a mix of machine learning (Isolation Forest) and simple rules.

You upload time stamped readings from sensors such as pressure, temperature, and flow.  
The app cleans the data, builds rolling features, scores each time step for how weird it looks, and then shows you a timeline, an alerts table, and a human readable incidents report.

---

## What this app does

In very simple terms:

* You give the app a CSV with a `timestamp` column and numeric sensor columns  
  for example `pressure_psig`, `temperature_f`, `flow_m3h`.
* The app cleans and resamples the data so you have one row per N minutes  
  N is chosen in the sidebar.
* It builds rolling features over a window  
  for example last 15 minutes average, spread, and z score for each sensor.
* It trains an Isolation Forest model on the early part of the timeline  
  this is treated as “normal”.
* It scores the later part of the timeline for anomalies  
  higher score means more unusual.
* It applies a simple physical rule  
  for example “if any sensor jumps more than 50 percent in one step, flag it”.
* It shows:
  * A line chart of anomaly score over time with alert markers
  * An alerts table with reason, rule hit, and top three abnormal features
  * Download buttons for scores, alerts, and a plain English `incidents.md` file

---

## Folder layout

A typical layout looks like this:

```text
project_root/
  app.py
  models/
    iforest.joblib        optional saved model bundle
  data/
    raw/                  optional
    processed/            optional (for ics_cleaned.csv)
    features/             optional (for ics_features.csv)
  outputs/                where you can store generated CSVs and plots
  README.md
```

You can adapt this to your own project.  
The only required file for the app is `app.py`.  
Demo CSV files such as `ics_cleaned.csv` are useful for testing.

---

## Requirements

* Python 3.10 or newer
* Recommended packages

```bash
pip install streamlit pandas numpy scikit-learn matplotlib joblib
```

If you have a `requirements.txt` file, you can instead run:

```bash
pip install -r requirements.txt
```

---

## How to run the app

From the project folder where `app.py` lives:

```bash
streamlit run app.py
```

Your browser will open at a local Streamlit address (something like `http://localhost:8501`).

---

## Input data format

The app expects a CSV with:

* One time column named `timestamp`  
  If the column has a different name but looks like datetimes, the app will try to auto detect it.
* One or more numeric sensor columns such as:
  * `pressure_psig`
  * `temperature_f`
  * `flow_m3h`

Example header row:

```text
timestamp,pressure_psig,temperature_f,flow_m3h
```

The demo file `ics_cleaned.csv` follows this pattern.

---

## Pipeline settings in the sidebar

All main controls live in the sidebar.

### Resample minutes

* How often to keep a data point  
  for example `1` keeps one point per minute, `5` keeps one point every five minutes.
* Smaller values keep more detail.  
  Larger values smooth the timeline.

### Max gap forward fill minutes

* How many minutes of missing data you are willing to fill by copying the last value forward.
* Example with max gap equal to 2 minutes  

  * 10:00 pressure 50  
  * 10:01 pressure 51  
  * 10:02 missing  
  * 10:03 missing  
  * 10:04 pressure 52  

  With max gap equal to 2, both 10:02 and 10:03 are filled from 10:01.

### Rolling window minutes

* Size of the window (in minutes) used to compute:
  * rolling average
  * rolling standard deviation
  * rolling z score
* Smaller window reacts faster but is noisier.  
  Larger window is smoother but slower to react.

### Train fraction

* Fraction of the timeline used as “normal” to train the Isolation Forest model.  
  Example: `0.60` means the first sixty percent of rows are used for training, the last forty percent are scored for anomalies.

### Highest anomaly percentile for alerts

* Used to set the Isolation Forest threshold.
* The model computes a score for every row in the test part.  
  The app sorts all scores and picks a threshold where some chosen percent of scores are below it.
* Example: if you use `0.99`, it picks a number where 99 percent of scores are lower and the top 1 percent are higher.  
  Those top rows become alerts.

### Advanced section

Inside the Advanced expander:

* Assumed anomaly rate in training data  
  A small guess such as `0.01` tells Isolation Forest that you expect about one percent of training rows to be outliers.
* Max percent jump rule  
  Example: `50` means if any sensor changes more than 50 percent between one minute and the next, the rule marks that time as a hit.

### Z baseline threshold

* A simple rule based detector that does not use machine learning.
* The app computes a combined z score per time step  
  maximum absolute z across sensors.
* When that number crosses your chosen threshold (for example 3 point 0) it counts as an alert.

---

## What happens when you upload a CSV

Once you choose and upload a CSV:

1. The app parses the timestamp column and sorts rows by time.  
2. It keeps numeric sensor columns and resamples to a fixed frequency.  
3. It forward fills short gaps according to your forward fill setting.  
4. It builds rolling features:
   * rolling average of each sensor over the window
   * rolling standard deviation of each sensor over the window
   * rolling z score for each sensor over the window  
     z equals (current value minus mean over window) divided by standard deviation over window.
5. Depending on the settings it either:
   * trains an Isolation Forest on the early fraction and scores the later fraction  
   or
   * runs the z baseline rule detector.
6. It applies the simple percent jump rule to the cleaned raw signals.
7. It builds:
   * a score timeline
   * an alerts table with reason, rule backed flag, and top three abnormal features
   * an incidents report in plain English.

---

## Understanding the outputs

### Score timeline

The “Score timeline” section shows:

* A line of anomaly score over time  
  higher means more unusual.
* A horizontal line showing the threshold.
* Markers on the times where alerts occur.

### Alerts table

The alerts table includes:

* Time  
  resampled timestamp, usually one row per minute.
* Anomaly score  
  higher means more unusual. Anything above the threshold is highlighted as an alert.
* Main feature  
  the feature that looked most abnormal at that time 
  for example rolling z of temperature or rolling standard deviation of pressure.
* Rule backed  
  check mark if the simple percent jump rule also fired at that time.
* Top three features  
  technical names of the three most abnormal features at that time.
* Top three z scores  
  how many standard deviations from normal each of those top three features is, in the same order.

Together, these columns tell you when something weird happened, how weird it was, and which sensors were mostly responsible.

### Download buttons

The app provides three main downloads:

* `scores.csv`  
  score for every time point in the scored window, plus rule hits.
* `alerts.csv`  
  the subset of rows above the threshold, including reason, rule hit, and top three features and z scores.
* `incidents.md`  
  a small Markdown report that explains each alert in plain language, one sentence per incident.  
  Example  
  “At 2025 dash 01 dash 02 08:12, the anomaly score was 0.611 (above the 0.552 threshold), and the main driver was temperature (how unstable it has been over the window). The simple percent jump rule also fired at this time. The top three abnormal features were temperature, pressure, and flow rate, at about 3.26, 2.74, and 2.21 standard deviations away from normal.”

You can open `incidents.md` in any Markdown viewer, code editor, or even GitHub.

---

## How Isolation Forest is used here

Very short explanation:

* Isolation Forest imagine randomly cutting up the data space into smaller and smaller slices.
* Points that are very different from the crowd get isolated in fewer cuts.
* Fewer cuts means higher anomaly score.

In this app:

* The model is trained only on the early part of the timeline  
  treated as normal history.
* The later part is scored.  
  The model does not know the future labels. It just says which moments look least like the normal training data.

The percentile slider then decides how many of the top scores become alerts.

---

## Troubleshooting

* Error about missing `sklearn`  
  Install `scikit-learn` with `pip install scikit-learn`.
* No numeric columns found  
  Check that your CSV has numeric sensor columns beyond the timestamp.
* No rows after time split  
  Your dataset might be too short, or the train fraction slider might be too close to one.
* No alerts  
  This can happen if the threshold is too high or the data is very calm.  
  Try lowering the percentile or the z baseline threshold to see some alerts for testing.

---

## Demo workflow

A simple way to demo the app:

1. Start the app with `streamlit run app.py`.
2. Upload a demo file such as `ics_cleaned.csv`.
3. Keep the default settings first and show:
   * score timeline with a few scattered alerts
   * alerts table
   * rule backed column
   * top three features and z scores.
4. Download `alerts.csv` and `incidents.md` and briefly open them to show how they can be used in an investigation.

This gives a full story from raw data to anomaly detection to human friendly explanation.

# MADE BY: Hamza Siddiqui
# LinkedIN: www.linkedin.com/in/hamsid