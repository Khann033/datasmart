[README.md](https://github.com/user-attachments/files/27315197/README.md)
## Live Demo

[Open DataSmart](https://datasmart-v2b2dg99txhfxpvggmwpfd.streamlit.app)

No installation needed — open the link, upload a CSV, and the app does the rest.
# DataSmart

> A machine learning web app that cleans, analyzes, and builds predictive models from any CSV file — no coding required.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-F7931E?style=flat-square)
![Built with Claude AI](https://img.shields.io/badge/Built%20with-Claude%20AI-8A6CF7?style=flat-square)
![Status](https://img.shields.io/badge/Status-Live-22c55e?style=flat-square)

---

## Live Demo

**[Open DataSmart](https://datasmart-jh2t3aduvhlc9ljwaatzni.streamlit.app)**

No installation needed — open the link, upload a CSV, and the app does the rest.

---

## About This Project

I built this project while learning machine learning through the Kaggle ML course. As I was going through the lessons — understanding how models learn from data, how predictions work, how to evaluate accuracy — I realized I wanted to build something real, not just run notebook exercises.

The idea came from a simple question: *what if someone could upload any dataset and get instant cleaning, analysis, and predictions without writing a single line of code?*

That became DataSmart.

---

## How It Was Built

I want to be transparent about how this project came together, because I think honesty about AI-assisted development is important.

**What I did:**

- Came up with the original idea and defined what the app should do
- Decided on the features: data cleaning, analysis, ML model training, CSV download
- Chose the design direction — clean, professional, no clutter
- Tested every feature with real datasets (Titanic, housing data)
- Identified bugs and problems during testing and described exactly what was wrong
- Made decisions on what to keep, change, or remove based on how it felt to use
- Wrote this README and the project description
- Deployed the app on Streamlit Cloud
- Managed the full GitHub workflow

**What Claude AI did:**

- Wrote the Python code for the Streamlit app based on my requirements
- Suggested the tech stack (Streamlit, scikit-learn, pandas)
- Debugged errors when I sent screenshots of what went wrong
- Redesigned the UI when I said it looked too AI-generated
- Removed emojis and rewrote labels when I flagged they looked robotic

**My honest take:**

Using AI as a coding assistant is a real skill. Knowing *what* to build, *how* to test it, *what* looks wrong, and *how* to direct the process — that's product thinking. The code is AI-generated. The product decisions, the testing, the vision, and the judgment calls are mine.

---

## What It Does

**Overview tab** — loads your CSV and instantly shows row count, column types, missing values, and duplicates

**Data Cleaning tab** — one click fills missing values, removes duplicates, and flags outliers. Download the cleaned file when done.

**Analysis tab** — distribution charts, outlier box plots, correlation heatmap, and category breakdowns

**Model tab** — pick a target column, select features, choose an algorithm, and train a model. See accuracy metrics and feature importance charts.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Streamlit | Web interface |
| Pandas | Data loading and cleaning |
| Scikit-learn | Machine learning models |
| Matplotlib + Seaborn | Charts and visualizations |
| Python 3.11 | Core language |

---

## Run It Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## What I Learned

- How machine learning models actually learn from data
- The difference between regression and classification tasks
- Why data cleaning matters before training any model
- How to evaluate model accuracy using MAE, R2 Score, and Accuracy
- How to deploy a Python app and make it publicly accessible
- How to work with AI tools effectively — directing, testing, and refining

---

## Author

**Khann033**
GitHub: [github.com/Khann033](https://github.com/Khann033)

---

*This project is part of my machine learning learning journey. Built with curiosity, tested with real data, and shipped with help from Claude AI.*

