# intelligent-app-testing-# 🧠 Intelligent App Testing System
### Data Science in SDLC — Bug Risk Analysis, Prediction & Prioritization

---

## 📌 Overview
This project uses **data science** to make software testing smarter.
Instead of manually retesting everything, the system learns from past bugs
to predict where issues are likely to occur in future releases.

---

## 🗂️ Project Structure
```
intelligent-app-testing/
│
├── data/
│   ├── generate_dataset.py      ← Run this first to create the dataset
│   ├── bug_dataset.csv          ← Generated synthetic dataset (500 records)
│   └── clean_bug_dataset.csv    ← Created after running the notebook
│
├── notebook/
│   └── analysis.ipynb           ← Full Jupyter Notebook analysis
│
├── app/
│   └── streamlit_app.py         ← Interactive Streamlit Dashboard
│
├── requirements.txt             ← All Python dependencies
└── README.md                    ← You are here
```

---

## ⚙️ Setup Instructions

### Step 1 — Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/intelligent-app-testing.git
cd intelligent-app-testing
```

### Step 2 — Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Generate the Dataset
```bash
cd data
python generate_dataset.py
cd ..
```

### Step 5 — Run the Jupyter Notebook
```bash
jupyter notebook notebook/analysis.ipynb
```

### Step 6 — Run the Streamlit App
```bash
streamlit run app/streamlit_app.py
```
Then open **http://localhost:8501** in your browser.

---

## 🚀 Deploy to Streamlit Cloud (Free)

1. Push this project to **GitHub**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Select your repo → Branch: `main` → File: `app/streamlit_app.py`
5. Click **Deploy** ✅

> **Note:** Make sure `bug_dataset.csv` is committed to the `data/` folder before deploying.

---

## 📊 Features

| Feature | Description |
|---|---|
| 📊 EDA | Bar charts, pie charts, heatmaps, trend lines |
| ⚠️ Risk Scoring | Weighted module risk score (0–100) |
| 🤖 ML Model | Random Forest — predicts bug reopen probability |
| 📝 NLP | TF-IDF cosine similarity — detects duplicate bugs |
| ✅ Fix Validation | Logic to assess if fixed bugs will reappear |
| 💡 Insights | 10 actionable recommendations + priority list |

---

## 🧰 Tech Stack
- **Python 3.10+**
- **Pandas, NumPy** — Data handling
- **Matplotlib, Seaborn, Plotly** — Visualization
- **Scikit-learn** — ML + NLP
- **Streamlit** — Interactive Web Dashboard

---

## 📁 Dataset Fields

| Column | Description |
|---|---|
| App_Version | Software version (v1.0 – v3.0) |
| Module | Feature area (Login, Payment, etc.) |
| Bug_ID | Unique bug identifier |
| Bug_Description | Text description of the bug |
| Severity | Low / Medium / High |
| Status | Open / Fixed / Reopened |
| Occurrences | How many times the bug occurred |
| Time_to_Fix_Days | Days taken to fix |
| Release_Date | Date the version was released |

---

## 💡 Key Questions Answered
1. Which modules are most prone to bugs?
2. Which bugs are likely to reoccur?
3. Is the latest version improving compared to previous ones?
4. What should be the testing priority for the next release?

---

*Built for the Data Science in SDLC assignment.*
