# Heart Disease Risk Predictor

A machine learning project that estimates whether a patient is likely to have heart disease from 11
routine clinical measurements, with a Streamlit app for trying the model interactively.

> **Educational project only — not a medical device.** It cannot diagnose anything and must not be
> used for health decisions.

## Dataset

[Heart Failure Prediction dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
(Kaggle, `heart.csv`): 918 patients, 11 features, binary target `HeartDisease` (1 = disease, 0 = normal).

| Type | Features |
|---|---|
| Numeric | Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak |
| Categorical | Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope |

`Cholesterol` and `RestingBP` contain zeros that are physiologically impossible and really mean
"not recorded". They are treated as missing values and imputed.

## Approach

1. **Cleaning:** zeros in `Cholesterol` and `RestingBP` become missing values.
2. **Splitting:** stratified 80/20 train/test split.
3. **Preprocessing (inside a scikit-learn Pipeline):** median imputation and standard scaling for
   numeric columns, one-hot encoding for categorical columns. Everything is fit on training data only,
   so no information leaks from the test set, and the exact same steps run at prediction time.
4. **Model comparison:** Logistic Regression, Decision Tree, KNN, SVM, Naive Bayes and Random Forest are
   compared with 5-fold stratified cross-validation on the training set, ranked by ROC-AUC.
5. **Final evaluation:** the best model is scored once on the held-out test set (accuracy, precision,
   recall, F1, ROC-AUC, confusion matrix), then refit on all data for the app.

Recall is reported alongside accuracy because, in a screening setting, missing a patient with disease
is usually costlier than a false alarm.

## The app

- Enter a patient's 11 measurements, or load a lower-risk or higher-risk example.
- See the predicted class and the estimated probability of heart disease.
- Upload a CSV for batch predictions and download the results.
- The sidebar shows the chosen model, its test metrics, and the cross-validation comparison of all models.

## Usage

Download `heart.csv` from Kaggle and place it in the `data/` folder (it stays on your machine and is not
committed), then:

```bash
pip install -r requirements.txt
python train.py          # compares models, prints metrics, saves the best one to artifacts/
streamlit run app.py
```

The app only needs the saved model in `artifacts/`, not the dataset.
