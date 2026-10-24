# Glucose Twin

Predicts whether a patient's glucose will go above 180 mg/dL in the next 2 hours, using the Shanghai T2DM dataset
(patient history from the Summary file + CGM glucose, meal timing and insulin from the per-patient files).

## Run it
1. `pip install -r requirements.txt`
2. Put your data in `data/` like this (names can differ slightly, but keep "T2DM" and "Summary" in them):
   ```
   data/Shanghai_T2DM/               <- the CGM files
   data/Shanghai_T2DM_Summary.xlsx
   ```
   In your Downloads folder, that is the `Shanghai_T2DM` folder and the `Shanghai_T2DM_Summary` file shown in your screenshot.
3. `python train.py --data data`  (about a minute; prints baseline vs model results)
4. `streamlit run app.py`

For the optional T1DM run: `python train.py --data data --kind T1DM`.

## Test without the real data
`python make_demo_data.py` then `python train.py --data data_demo`. This data is SYNTHETIC and only checks that the app runs. Never report its scores.

## How it works
- `pipeline.py`: reads the files, converts mmol/L to mg/dL if needed, fills gaps of up to 30 min, splits at longer gaps, builds features from past data only, labels the next 2 h.
- `train.py`: splits by patient (train/validation/test never share a patient), trains LightGBM, compares with a "current glucose only" baseline.
- `app.py`: the interface. Pick a recording and a moment, see the risk, the reasons, and try what-if changes.

## If something breaks
The loader finds columns by name: a date/time column, a column containing "cgm", "dietary" for meals, and "insulin". If your files differ, edit `load_cgm` in `pipeline.py`, or send me the column names.
