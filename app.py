import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import streamlit as st

from common import CATEGORICAL, FEATURES, clean

HERE = Path(__file__).parent
ART = HERE / "artifacts"

st.set_page_config(page_title="Heart Disease Risk", page_icon="❤️", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load():
    return joblib.load(ART / "model.joblib"), json.loads((ART / "meta.json").read_text())


try:
    model, meta = load()
except FileNotFoundError:
    st.error("Trained model not found. Run `python train.py` locally and commit the `artifacts/` folder.")
    st.stop()
except Exception as e:  # e.g. pickle saved with an incompatible scikit-learn version
    st.error(f"Could not load the saved model ({type(e).__name__}). Make sure the scikit-learn version in "
             "`requirements.txt` matches the one used for training, then retrain.")
    st.stop()

if meta.get("sklearn_version") and meta["sklearn_version"] != sklearn.__version__:
    st.warning(f"Model was trained with scikit-learn {meta['sklearn_version']} but this app runs "
               f"{sklearn.__version__}. Pin the training version in requirements.txt.")


OPTIONS = {
    "Sex": {"M": "Male", "F": "Female"},
    "ChestPainType": {"ASY": "Asymptomatic", "ATA": "Atypical angina",
                      "NAP": "Non-anginal pain", "TA": "Typical angina"},
    "RestingECG": {"Normal": "Normal", "ST": "ST-T wave abnormality",
                   "LVH": "Left ventricular hypertrophy"},
    "ExerciseAngina": {"N": "No", "Y": "Yes"},
    "ST_Slope": {"Up": "Upsloping", "Flat": "Flat", "Down": "Downsloping"},
}
DEFAULT = dict(Age=54, Sex="M", ChestPainType="NAP", RestingBP=130, Cholesterol=240,
               FastingBS=0, RestingECG="Normal", MaxHR=140, ExerciseAngina="N",
               Oldpeak=1.0, ST_Slope="Flat")
EXAMPLES = {
    "Lower-risk example": dict(Age=40, Sex="F", ChestPainType="ATA", RestingBP=120, Cholesterol=200,
                               FastingBS=0, RestingECG="Normal", MaxHR=170, ExerciseAngina="N",
                               Oldpeak=0.0, ST_Slope="Up"),
    "Higher-risk example": dict(Age=62, Sex="M", ChestPainType="ASY", RestingBP=150, Cholesterol=260,
                                FastingBS=1, RestingECG="ST", MaxHR=105, ExerciseAngina="Y",
                                Oldpeak=2.5, ST_Slope="Flat"),
}
for k, v in DEFAULT.items():
    st.session_state.setdefault(k, v)


def load_example(name):
    for k, v in EXAMPLES[name].items():
        st.session_state[k] = v


def predict(df: pd.DataFrame) -> pd.DataFrame:
    df = clean(df)
    out = df.copy()
    out["disease_probability"] = model.predict_proba(df[FEATURES])[:, 1]
    out["prediction"] = ["Heart disease likely" if p >= 0.5 else "Heart disease unlikely"
                         for p in out["disease_probability"]]
    return out


st.title("❤️ Heart Disease Risk Predictor")
st.caption("Machine learning model trained on the Heart Failure Prediction dataset "
           f"({meta['n_rows']} patients, 11 clinical features).")
st.warning("**Educational demo only — not a medical device.** It cannot diagnose anything. "
           "Anyone concerned about their heart health should see a doctor.")

with st.sidebar:
    t = meta["test"]
    st.header("Model")
    st.write(f"**{meta['best_model']}**, chosen by 5-fold cross-validation ROC-AUC.")
    st.metric("Test accuracy", f"{t['accuracy']:.1%}")
    st.metric("Test recall (disease caught)", f"{t['recall']:.1%}")
    st.metric("Test ROC-AUC", f"{t['roc_auc']:.3f}")
    st.caption(f"Held-out test set: {t['n_test']} patients.")
    with st.expander("All models compared"):
        cvt = pd.DataFrame(meta["cv_table"]).set_index("model").round(3)
        st.dataframe(cvt)

tab_one, tab_batch = st.tabs(["Single patient", "Batch (CSV upload)"])

with tab_one:
    c1, c2, _ = st.columns([1, 1, 4])
    c1.button("Load lower-risk example", on_click=load_example, args=("Lower-risk example",))
    c2.button("Load higher-risk example", on_click=load_example, args=("Higher-risk example",))

    a, b, c = st.columns(3)
    a.number_input("Age", 1, 120, key="Age")
    a.selectbox("Sex", list(OPTIONS["Sex"]), format_func=OPTIONS["Sex"].get, key="Sex")
    a.selectbox("Chest pain type", list(OPTIONS["ChestPainType"]),
                format_func=OPTIONS["ChestPainType"].get, key="ChestPainType")
    b.number_input("Resting blood pressure (mm Hg)", 0, 250, key="RestingBP",
                   help="Enter 0 if unknown; it will be estimated.")
    b.number_input("Cholesterol (mg/dl)", 0, 700, key="Cholesterol",
                   help="Enter 0 if unknown; it will be estimated.")
    b.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1],
                format_func=lambda x: "Yes" if x else "No", key="FastingBS")
    c.selectbox("Resting ECG", list(OPTIONS["RestingECG"]),
                format_func=OPTIONS["RestingECG"].get, key="RestingECG")
    c.number_input("Max heart rate achieved", 40, 250, key="MaxHR")
    c.selectbox("Exercise-induced angina", list(OPTIONS["ExerciseAngina"]),
                format_func=OPTIONS["ExerciseAngina"].get, key="ExerciseAngina")
    st.number_input("Oldpeak (ST depression)", -3.0, 8.0, step=0.1, key="Oldpeak")
    st.selectbox("ST slope", list(OPTIONS["ST_Slope"]),
                 format_func=OPTIONS["ST_Slope"].get, key="ST_Slope")

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([{f: st.session_state[f] for f in FEATURES}])
        res = predict(row).iloc[0]
        p = float(res["disease_probability"])
        left, right = st.columns([1, 2])
        with left:
            (st.error if p >= 0.5 else st.success)(f"### {res['prediction']}\n{p:.0%} estimated probability")
        with right:
            st.write("Estimated probability of heart disease")
            st.progress(p)

with tab_batch:
    st.write("Upload a CSV with these columns: " + ", ".join(f"`{f}`" for f in FEATURES) +
             ". Extra columns (such as `HeartDisease`) are ignored.")
    st.download_button("Download CSV template",
                       pd.DataFrame([DEFAULT])[FEATURES].to_csv(index=False),
                       "template.csv", "text/csv")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        data = pd.read_csv(up)
        missing = [f for f in FEATURES if f not in data.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            bad = {c: sorted(set(data[c]) - set(OPTIONS[c])) for c in CATEGORICAL
                   if set(data[c]) - set(OPTIONS[c])}
            if bad:
                st.warning(f"Unexpected category values (treated as unknown): {bad}")
            result = predict(data)
            st.dataframe(result[["prediction", "disease_probability"] + FEATURES],
                         use_container_width=True)
            st.download_button("Download predictions", result.to_csv(index=False),
                               "predictions.csv", "text/csv")
