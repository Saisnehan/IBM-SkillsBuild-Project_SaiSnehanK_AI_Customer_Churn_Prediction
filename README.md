# AI-Powered Customer Churn Prediction & Retention Recommendation System

> **IBM SkillsBuild Data Analytics with AI Academic Internship Program**  
> Conducted by **BharatCares in association with AICTE**  
> Student: **Sai Snehan_K**

---

## Project Overview

This project builds a complete end-to-end AI system that:

1. Predicts the probability that a telecom customer will churn.
2. Classifies customers into **LOW / MEDIUM / HIGH** risk categories.
3. Generates personalised **retention recommendations** based on customer attributes.
4. Presents all results through an **interactive Streamlit dashboard**.

The entire data-science and machine-learning backend lives in a single Jupyter Notebook. A Streamlit application loads the saved model artifact and provides a browser-based interface for predictions.

---

## Problem Statement

Customer churn â€” the rate at which customers stop doing business with a company â€” is a critical metric for telecom providers. Acquiring a new customer costs significantly more than retaining an existing one. By predicting which customers are at risk of churning *before* they leave, businesses can deploy targeted retention strategies and reduce revenue loss.

---

## Objectives

- Analyse the Telco Customer Churn dataset to understand structure and quality.
- Clean and preprocess customer data for ML readiness.
- Perform Exploratory Data Analysis to discover churn-related patterns.
- Engineer informative features (TenureGroup, ServiceCount, ChargePerMonth).
- Build and compare three classification models: Logistic Regression, Random Forest, Gradient Boosting.
- Select the best model using recall, F1-score, and ROC-AUC.
- Predict individual churn probability with `predict_proba()`.
- Classify customers into risk tiers.
- Generate actionable retention recommendations.
- Deploy an interactive Streamlit dashboard.
- Demonstrate AI-assisted development using IBM Bob.

---

## Dataset

| Property | Details |
|---|---|
| **Name** | Telco Customer Churn |
| **Source** | [https://www.kaggle.com/blastchar/telco-customer-churn](https://www.kaggle.com/blastchar/telco-customer-churn) |
| **File** | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| **Target** | `Churn` (Yes â†’ 1, No â†’ 0) |
| **Records** | ~7,043 customers |
| **Features** | 21 (demographics, services, billing, contract) |

---

## Technologies Used

| Category | Library / Tool |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Machine learning | `scikit-learn` |
| Model persistence | `joblib` |
| Web application | `streamlit` |
| Documentation | `python-docx` |
| Development AI | IBM Bob |

---

## Project Architecture

```
Kaggle Dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv)
        â”‚
        â–¼
SaiSnehan_AI_Customer_Churn_Prediction.ipynb  â† Complete backend
        â”‚
        â”œâ”€â”€ Data Loading & Inspection
        â”œâ”€â”€ Data Cleaning (TotalCharges, duplicates, types)
        â”œâ”€â”€ Exploratory Data Analysis (10+ visualisations)
        â”œâ”€â”€ Feature Engineering (TenureGroup, ServiceCount, ChargePerMonth)
        â”œâ”€â”€ Data Preprocessing (ColumnTransformer + Pipeline)
        â”œâ”€â”€ Train / Test Split (80/20, stratified)
        â”œâ”€â”€ Model Training (LR, RF, GB)
        â”œâ”€â”€ Model Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
        â”œâ”€â”€ Feature Importance / Explainability
        â”œâ”€â”€ Churn Probability Prediction (predict_proba)
        â”œâ”€â”€ Risk Classification (LOW / MEDIUM / HIGH)
        â””â”€â”€ Retention Recommendation System
                â”‚
                â–¼
        models/churn_model.pkl  +  models/model_metadata.pkl
                â”‚
                â–¼
        app.py  â”€â”€â–º  Streamlit Frontend
                      â”œâ”€â”€ Dashboard (KPIs, charts)
                      â”œâ”€â”€ Customer Prediction (interactive form)
                      â”œâ”€â”€ Analytics (risk, features, performance)
                      â””â”€â”€ Model Information
```

---

## Features

- **Dashboard**: Total customers, churn rate, high-risk count, average monthly charges, and visual breakdowns by contract, tenure, payment method.
- **Customer Prediction**: Input form for all 19 customer attributes â†’ churn probability, risk level, key risk factors, and personalised recommendations.
- **Analytics**: Churn by service type, risk distribution histogram, feature importance chart, model performance table.
- **Model Information**: Evaluation metrics, model comparison, risk threshold documentation, project limitations, IBM Bob contribution.

---

## Machine Learning Models

| Model | Notes |
|---|---|
| Logistic Regression | Baseline model; interpretable coefficients |
| Random Forest | Ensemble; handles non-linearities well |
| Gradient Boosting | Sequential boosting; strong predictive performance |

All models use `class_weight='balanced'` (LR/RF) or are evaluated with recall focus to handle the ~26% churn class imbalance.

---

## Evaluation Metrics

For churn prediction, **Recall** and **ROC-AUC** are prioritised over Accuracy because:
- A missed churner (False Negative) is more costly than a false alarm (False Positive).
- ROC-AUC measures the model's overall discrimination ability across all thresholds.

The final model is selected based on the highest ROC-AUC computed from the actual dataset.

---

## IBM Bob Contribution

IBM Bob was used as an AI-assisted development tool for:

- Code generation and notebook structuring
- Scikit-learn pipeline design
- Feature engineering suggestions
- Debugging (e.g. TotalCharges type conversion)
- Markdown documentation writing
- Business insight interpretation
- Streamlit layout design

> **IBM Bob did NOT train the machine-learning models.** All training was performed by Python/Scikit-learn on the Kaggle dataset.

---

## Project Structure

```
AI-Customer-Churn/
â”‚
â”œâ”€â”€ data/
â”‚   â””â”€â”€ WA_Fn-UseC_-Telco-Customer-Churn.csv   â† Place dataset here
â”‚
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ churn_model.pkl                          â† Saved after notebook run
â”‚   â””â”€â”€ model_metadata.pkl                       â† Saved after notebook run
â”‚
â”œâ”€â”€ SaiSnehan_AI_Customer_Churn_Prediction.ipynb â† Complete backend/ML code
â”œâ”€â”€ app.py                                        â† Streamlit frontend
â”œâ”€â”€ requirements.txt                              â† Dependencies
â”œâ”€â”€ README.md                                     â† This file
â””â”€â”€ SaiSnehan_ProjectReport.docx                  â† Internship project report
```

---

## Installation

1. **Clone / download** this project folder.

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset Setup

1. Download the dataset from Kaggle:  
   [https://www.kaggle.com/blastchar/telco-customer-churn](https://www.kaggle.com/blastchar/telco-customer-churn)

2. Place the downloaded CSV file at:
   ```
   data/WA_Fn-UseC_-Telco-Customer-Churn.csv
   ```

> The dataset is already included in this project if you received the complete project folder.

---

## Running the Notebook

1. Launch Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

2. Open `SaiSnehan_AI_Customer_Churn_Prediction.ipynb`.

3. Run all cells **from top to bottom** (Kernel â†’ Restart & Run All).

4. The notebook will:
   - Load and clean the dataset
   - Perform EDA and visualisations
   - Train all three models
   - Evaluate and compare models
   - Save `models/churn_model.pkl` and `models/model_metadata.pkl`

> **Always run the notebook before the Streamlit app** to generate the model artifacts.

---

## Running the Streamlit Application

After the notebook has been run and the model saved:

```bash
streamlit run app.py
```

The browser will open automatically at `http://localhost:8501`.

Navigate using the sidebar:
- ðŸ  Dashboard
- ðŸ”® Customer Churn Prediction
- ðŸ“ˆ Analytics
- â„¹ï¸ Model Information

---

## Results

> All results are computed from the actual Telco Customer Churn dataset during notebook execution.  
> The values below will be populated after running the notebook.

| Metric | Value (after notebook run) |
|---|---|
| Dataset size | ~7,043 customers |
| Overall churn rate | ~26% |
| Best model | See notebook output |
| Best ROC-AUC | See notebook output |
| Best Recall | See notebook output |
| Best F1-Score | See notebook output |

---

## Screenshots

> Screenshots are generated when the Streamlit application is running.

- [Insert Streamlit Dashboard Screenshot Here]
- [Insert Customer Churn Prediction Screenshot Here]
- [Insert Risk Distribution Screenshot Here]
- [Insert Feature Importance Screenshot Here]
- [Insert Model Comparison Screenshot Here]

---

## Limitations

- The model is trained on historical data and may not reflect future customer behaviour.
- Findings are specific to this Telco dataset.
- Risk thresholds (LOW/MEDIUM/HIGH) are project-defined, not universal.
- The model identifies correlations â€” not causal relationships.
- No real-time behavioural data or customer service history included.
- No hyperparameter optimisation performed.

---

## Future Enhancements

- Hyperparameter optimisation (GridSearchCV / Optuna)
- SHAP values for individual prediction explainability
- Real-time CRM integration
- A/B testing framework for retention campaigns
- Automated retraining pipeline
- Customer lifetime value integration

---

## Conclusion

This project successfully demonstrates a complete AI-powered churn prediction and retention recommendation pipeline â€” from raw data to an interactive web dashboard â€” using Python, Scikit-learn, and Streamlit, aligned with IBM SkillsBuild internship objectives.

---

## References

- Kaggle Telco Customer Churn Dataset: https://www.kaggle.com/blastchar/telco-customer-churn
- Scikit-learn Documentation: https://scikit-learn.org/stable/
- Streamlit Documentation: https://docs.streamlit.io/
- IBM SkillsBuild: https://skillsbuild.org/
- Pandas Documentation: https://pandas.pydata.org/docs/

