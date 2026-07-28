# Fake Job Posting Detector

A trust-and-safety style ML classifier that flags fraudulent job listings
before they go live — the kind of pre-publish screening model used by
hiring platforms such as LinkedIn, Indeed, or Glassdoor to protect job
seekers from scams (fake "work from home" offers, advance-fee fraud,
identity-theft-via-application schemes, etc.).

Built for the *Theoretical Foundations of Data Science* course Python
project (Ariel University, Dr. Elad Aigner-Horev). Project rules: choose
your own dataset, visualize it, discuss your preconceptions, try several
ML models, and deploy the whole thing via GitHub with a written report.

---

## 1. Why this project / company angle

Hiring platforms sit on a two-sided marketplace problem: they need to let
almost any employer post a job cheaply and instantly, but a small fraction
of postings are outright fraud — designed to harvest applicants' personal
data, bank details, or upfront "processing fees." This is a natural home
for a lightweight, interpretable, low-latency classifier that scores a
posting **at submission time**, before it's shown to a single job seeker.
It's also a well-documented real business problem: LinkedIn, Indeed and
Glassdoor all publish trust & safety engineering content on exactly this
kind of model.

## 2. Dataset

**Real / Fake Job Posting Prediction** (Kaggle, ~18,000 postings, ~866
labeled fraudulent — a realistic ~4.8% positive rate):
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

Columns include `title`, `location`, `department`, `salary_range`,
`company_profile`, `description`, `requirements`, `benefits`,
`telecommuting`, `has_company_logo`, `has_questions`, `employment_type`,
`required_experience`, `required_education`, `industry`, `function`, and
the target `fraudulent`.

### Getting the real data
1. Download `fake_job_postings.csv` from the Kaggle link above (free
   Kaggle account required).
2. Place it at `data/fake_job_postings.csv`.
3. Run everything with that path instead of the synthetic file, e.g.
   `python3 src/train.py data/fake_job_postings.csv`.

### Synthetic fallback data
This sandbox environment couldn't reach kaggle.com directly, so
`src/generate_synthetic_data.py` generates a **synthetic stand-in**
dataset with the same schema, the same ~4.8% class imbalance, and
deliberate 12% label noise/overlap between classes (so it isn't
trivially separable — see Section 5). All results in this README/report
were produced end-to-end on that synthetic file
(`data/synthetic_job_postings.csv`) to prove the pipeline works; swapping
in the real Kaggle CSV requires no code changes, only better (real) data.

## 3. Repository structure

```
fake-job-detector/
├── README.md                     <- this file
├── requirements.txt
├── data/
│   └── synthetic_job_postings.csv
├── src/
│   ├── generate_synthetic_data.py  <- synthetic data generator (fallback)
│   ├── features.py                 <- TF-IDF + meta-feature engineering
│   ├── eda.py                      <- exploratory data analysis + plots
│   ├── train.py                    <- trains & compares 5 models
│   ├── plot_results.py             <- model comparison / confusion matrix plots
│   └── predict.py                  <- CLI to score a single new posting
├── models/                       <- saved best model + featurizer (joblib)
├── reports/
│   ├── fig1_class_balance.png ... fig6_confusion_matrix.png
│   ├── model_comparison.csv / .json
│   └── PROJECT_REPORT.md         <- full write-up: journey, findings, GPT workflow log
└── notebooks/
    └── walkthrough.ipynb         <- end-to-end narrative notebook
```

## 4. Quickstart

```bash
pip install -r requirements.txt

# 1. Get data (synthetic fallback is already generated in data/)
python3 src/generate_synthetic_data.py

# 2. Exploratory data analysis -> reports/fig1..fig4
python3 src/eda.py data/synthetic_job_postings.csv

# 3. Train & compare 5 models -> reports/model_comparison.csv, models/*.joblib
python3 src/train.py data/synthetic_job_postings.csv

# 4. Model comparison + confusion matrix plots -> reports/fig5, fig6
python3 src/plot_results.py

# 5. Score a new posting
python3 src/predict.py \
  --title "Work From Home - Earn \$5000/week!!!" \
  --description "No experience necessary, start earning TODAY!!! Just send your bank details." \
  --has-company-logo 0 --has-questions 0
```

## 5. Models & results

Five classical models were trained on identical TF-IDF (title +
description + requirements + company profile + benefits, 1-2 grams, 3000
features) plus 10 structured meta-features (logo presence, salary
disclosed, description length, punctuation/uppercase ratios, etc.):
Logistic Regression, Multinomial Naive Bayes, Random Forest, Linear SVM,
Gradient Boosting. Full numbers, plots, and discussion are in
[`reports/PROJECT_REPORT.md`](reports/PROJECT_REPORT.md).

Headline result on the synthetic held-out test set: **Linear SVM** was
the best model by F1 on the fraud class (precision 1.00, recall 0.85, F1
0.92, ROC-AUC 0.99), with Random Forest close behind. Naive Bayes, the
weakest model, badly over-predicted fraud (precision 0.27) — a useful,
explainable failure mode discussed in the report.

## 6. GPT / AI workflow disclosure

See [`reports/PROJECT_REPORT.md`](reports/PROJECT_REPORT.md), section
"AI workflow log," for a step-by-step account of how this project was
built with AI assistance (Claude), including what was AI-suggested vs.
independently decided, and what was verified by actually running the
code.

## 7. License

MIT — see course submission guidelines for any additional academic
integrity requirements around code reuse.
