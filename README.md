# ECON3916-Project-2

# World Bank Project Risk Predictor
Predicts whether a World Bank development project will receive a Satisfactory or Unsatisfactory outcome rating using structural characteristics observable at the time of approval.

**ECON 3916 — Statistical & Machine Learning | Spring 2026 | Ethan McGeever**

---

## Repository Contents

| File | Description |
|---|---|
| `3916-final-project-starter.ipynb` | Full analysis pipeline: EDA, modeling, evaluation |
| `app.py` | Streamlit dashboard |
| `requirements.txt` | Python dependencies |
| `model.pkl` | Saved Random Forest model |
| `training_columns.json` | One-hot encoded column names for input alignment |
| `dataset/` | IEG Project Performance Ratings data |

---

## Environment Setup

    conda create -n wb-predictor python=3.10
    conda activate wb-predictor
    pip install -r requirements.txt

---

## Data

Data source: World Bank IEG Project Performance Ratings, accessed April 2026.
https://ieg.worldbankgroup.org/ratings

Place the downloaded file in the `dataset/` folder before running the notebook.

---

## Running the Notebook

    jupyter notebook "3916-final-project-starter.ipynb"

Run all cells top-to-bottom. This regenerates `model.pkl` and `training_columns.json`. All random operations use `random_state=42`.

---

## Running the App Locally

streamlit run app.py

Because model.pkl was too big to push to git, on the first run, the app trains the Random Forest from the dataset CSV and saves model.pkl automatically. All random operations use random_state=42.
