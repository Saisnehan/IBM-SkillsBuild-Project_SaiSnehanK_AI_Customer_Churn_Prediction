"""
AI-Powered Customer Churn Prediction & Retention Recommendation System
=======================================================================
Streamlit Frontend Application

Architecture:
    Saved model (models/churn_model.pkl) --> This app --> User Interface

All prediction logic uses the trained pipeline saved by the Jupyter notebook.
No model training occurs inside this file.

IBM SkillsBuild Data Analytics with AI Academic Internship Program
BharatCares in association with AICTE
Student: Sai Snehan_K
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")

# -- Page configuration --------------------------------------------------------
st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Constants -----------------------------------------------------------------
DATA_PATH     = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH    = "models/churn_model.pkl"
METADATA_PATH = "models/model_metadata.pkl"

RISK_COLORS = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#27ae60"}

# Navigation page labels (plain text, no emoji)
NAV_DASHBOARD   = "Dashboard"
NAV_PREDICTION  = "Customer Churn Prediction"
NAV_ANALYTICS   = "Analytics"
NAV_MODEL_INFO  = "Model Information"


# -- Risk classification -------------------------------------------------------
def classify_risk(probability):
    """Classify churn probability into LOW / MEDIUM / HIGH risk tier."""
    if probability < 0.30:
        return "LOW"
    elif probability < 0.60:
        return "MEDIUM"
    else:
        return "HIGH"


# -- Retention recommendations -------------------------------------------------
def generate_retention_recommendations(customer_series, churn_probability, risk_category):
    """
    Generate personalised retention recommendations for a customer.
    These are analytical suggestions -- not guaranteed interventions.
    """
    risk_factors = []
    recommendations = []

    if customer_series.get("Contract") == "Month-to-month":
        risk_factors.append("Month-to-month contract (higher churn association)")
        recommendations.append("Offer a discounted 1-year or 2-year contract upgrade.")

    if int(customer_series.get("tenure", 99)) <= 12:
        risk_factors.append("Short tenure -- new customer (higher churn risk)")
        recommendations.append("Assign a dedicated onboarding specialist for the first year.")

    if customer_series.get("InternetService") == "Fiber optic":
        risk_factors.append("Fiber optic internet (associated with higher churn in this dataset)")
        recommendations.append(
            "Conduct a service satisfaction check-in; address any speed/reliability concerns."
        )

    if customer_series.get("TechSupport") in ["No", "No internet service"]:
        risk_factors.append("No Tech Support subscription")
        recommendations.append("Offer a free trial or discounted Tech Support bundle.")

    if customer_series.get("OnlineSecurity") in ["No", "No internet service"]:
        risk_factors.append("No Online Security subscription")
        recommendations.append(
            "Highlight Online Security benefits; offer an introductory discount."
        )

    if customer_series.get("PaymentMethod") == "Electronic check":
        risk_factors.append("Electronic check payment (associated with higher churn)")
        recommendations.append(
            "Incentivise switching to automatic bank transfer or credit card payment."
        )

    monthly = float(customer_series.get("MonthlyCharges", 0))
    if monthly > 70:
        risk_factors.append("High monthly charges ($%.2f)" % monthly)
        recommendations.append(
            "Review billing for potential bundle discounts or service optimisation."
        )

    senior = str(customer_series.get("SeniorCitizen", "0"))
    if senior in ["1", "Yes"]:
        risk_factors.append("Senior citizen customer")
        recommendations.append(
            "Offer a senior-specific loyalty programme or simplified billing."
        )

    if customer_series.get("PaperlessBilling") == "Yes":
        risk_factors.append("Paperless billing (associated with higher churn in this dataset)")
        recommendations.append(
            "Ensure billing communication is clear and easy to understand digitally."
        )

    if risk_category == "HIGH":
        recommendations.append("Escalate to retention team for immediate personalised outreach.")
        recommendations.append("Consider a loyalty reward or retention offer within 7 days.")
    elif risk_category == "MEDIUM":
        recommendations.append("Schedule a proactive customer satisfaction call.")
        recommendations.append(
            "Evaluate current service package for value improvement opportunities."
        )
    else:
        recommendations.append(
            "Continue standard engagement; monitor for service usage changes."
        )

    if not risk_factors:
        risk_factors.append("No prominent individual risk factors detected by the model.")

    return {
        "churn_probability": round(churn_probability * 100, 1),
        "risk_category": risk_category,
        "risk_factors": risk_factors,
        "recommendations": recommendations,
    }


# -- Feature engineering (mirrors notebook -- applied at inference time) -------
def engineer_features(df):
    """Apply the same feature engineering as in the notebook."""
    df = df.copy()

    def assign_tenure_group(tenure):
        if tenure <= 12:
            return "New (0-12m)"
        elif tenure <= 24:
            return "Early (13-24m)"
        elif tenure <= 48:
            return "Established (25-48m)"
        else:
            return "Loyal (49m+)"

    df["TenureGroup"] = df["tenure"].apply(assign_tenure_group)

    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
    ]
    df["ServiceCount"] = df[service_cols].apply(lambda row: (row == "Yes").sum(), axis=1)

    df["ChargePerMonth"] = df.apply(
        lambda r: r["TotalCharges"] / r["tenure"] if r["tenure"] > 0 else r["MonthlyCharges"],
        axis=1,
    )

    # SeniorCitizen as string (matches notebook preprocessing)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    return df


# -- Cached data / model loaders -----------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None, (
            "Dataset not found. Please place "
            "WA_Fn-UseC_-Telco-Customer-Churn.csv in the data/ folder."
        )
    try:
        df = pd.read_csv(DATA_PATH)
        df.drop(columns=["customerID"], errors="ignore", inplace=True)
        df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        df.drop_duplicates(inplace=True)
        return df, None
    except Exception as exc:
        return None, "Error loading dataset: %s" % exc


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, (
            "Trained model not found. Please run "
            "SaiSnehan_AI_Customer_Churn_Prediction.ipynb first to generate "
            "models/churn_model.pkl."
        )
    try:
        pipeline = joblib.load(MODEL_PATH)
        metadata = joblib.load(METADATA_PATH) if os.path.exists(METADATA_PATH) else {}
        return pipeline, metadata, None
    except Exception as exc:
        return None, None, "Error loading model: %s" % exc


# -- Sidebar navigation --------------------------------------------------------
def sidebar():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [NAV_DASHBOARD, NAV_PREDICTION, NAV_ANALYTICS, NAV_MODEL_INFO],
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**IBM SkillsBuild Internship**\n\n"
        "Data Analytics with AI\n\n"
        "BharatCares x AICTE\n\n"
        "Student: Sai Snehan_K"
    )
    return page


# ==============================================================================
# PAGE 1 -- DASHBOARD
# ==============================================================================
def page_dashboard(df, pipeline, metadata):
    st.title("Dashboard -- Customer Churn Overview")

    if df is None:
        st.error("Dataset not available. Cannot display dashboard.")
        return

    # KPI metrics
    total_customers = len(df)
    total_churned   = int(df["Churn"].sum())
    churn_rate      = df["Churn"].mean() * 100
    avg_monthly     = df["MonthlyCharges"].mean()

    high_risk_count = "Run notebook first"
    probs = None
    if pipeline is not None:
        try:
            df_fe = engineer_features(df.drop(columns=["Churn"], errors="ignore"))
            probs = pipeline.predict_proba(df_fe)[:, 1]
            high_risk_count = int((probs >= 0.60).sum())
        except Exception:
            high_risk_count = "N/A"

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Customers",     "%d" % total_customers)
    col2.metric("Churned Customers",   "%d" % total_churned)
    col3.metric("Churn Rate",          "%.1f%%" % churn_rate)
    col4.metric("High-Risk Customers", str(high_risk_count))
    col5.metric("Avg Monthly Charges", "$%.2f" % avg_monthly)

    st.markdown("---")

    # Row 1: Churn distribution + Contract
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        counts = df["Churn"].value_counts()
        ax.bar(["No Churn", "Churn"], counts.values,
               color=["steelblue", "tomato"], edgecolor="white")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 20, str(v), ha="center", fontweight="bold")
        ax.set_ylabel("Customers")
        ax.set_title("Overall Churn Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.subheader("Churn by Contract Type")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        contract_rate = df.groupby("Contract")["Churn"].mean().mul(100)
        contract_rate.plot(kind="bar", ax=ax, color="coral", edgecolor="white", rot=20)
        ax.set_ylabel("Churn Rate (%)")
        ax.set_title("Churn Rate by Contract")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Row 2: Tenure + Risk distribution
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Churn by Tenure Group")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        df_temp = df.copy()
        df_temp["TenureGroup"] = pd.cut(
            df_temp["tenure"],
            bins=[0, 12, 24, 48, df_temp["tenure"].max() + 1],
            labels=["New (0-12m)", "Early (13-24m)", "Established (25-48m)", "Loyal (49m+)"],
        )
        tg_rate = df_temp.groupby("TenureGroup", observed=True)["Churn"].mean().mul(100)
        tg_rate.plot(kind="bar", ax=ax, color="mediumseagreen", edgecolor="white", rot=20)
        ax.set_ylabel("Churn Rate (%)")
        ax.set_title("Churn Rate by Tenure Group")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_d:
        st.subheader("Risk Distribution (Full Dataset)")
        if pipeline is not None and probs is not None:
            try:
                risks = [classify_risk(p) for p in probs]
                risk_series = pd.Series(risks).value_counts()
                risk_order  = ["LOW", "MEDIUM", "HIGH"]
                risk_ordered = risk_series.reindex(risk_order, fill_value=0)
                fig, ax = plt.subplots(figsize=(5, 3.5))
                bars = ax.bar(risk_ordered.index, risk_ordered.values,
                              color=["steelblue", "gold", "tomato"], edgecolor="white")
                for bar, val in zip(bars, risk_ordered.values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 5,
                        str(val), ha="center", fontweight="bold"
                    )
                ax.set_ylabel("Customers")
                ax.set_title("Risk Category Distribution")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.info("Risk distribution unavailable: %s" % e)
        else:
            st.info("Run the notebook to generate the model, then reload.")

    # Row 3: Payment method
    st.subheader("Churn by Payment Method")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    pm_rate = df.groupby("PaymentMethod")["Churn"].mean().mul(100)
    pm_rate.plot(kind="bar", ax=ax, color="orchid", edgecolor="white", rot=15)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Payment Method")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ==============================================================================
# PAGE 2 -- CUSTOMER CHURN PREDICTION
# ==============================================================================
def page_prediction(pipeline):
    st.title("Customer Churn Prediction")

    if pipeline is None:
        st.error(
            "Trained model not found. "
            "Please run SaiSnehan_AI_Customer_Churn_Prediction.ipynb first."
        )
        return

    st.markdown(
        "Enter the customer details below and click **Predict Churn** "
        "to get the churn probability, risk category, and personalised retention recommendations."
    )

    with st.form("prediction_form"):
        st.subheader("Customer Demographics")
        col1, col2, col3, col4 = st.columns(4)
        gender     = col1.selectbox("Gender",         ["Male", "Female"])
        senior     = col2.selectbox("Senior Citizen", ["No", "Yes"])
        partner    = col3.selectbox("Partner",        ["Yes", "No"])
        dependents = col4.selectbox("Dependents",     ["No", "Yes"])

        st.subheader("Account & Billing")
        col5, col6, col7 = st.columns(3)
        tenure          = col5.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = col6.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
        total_charges   = col7.number_input(
            "Total Charges ($)", 0.0, 10000.0,
            float(tenure * monthly_charges), step=1.0
        )

        col8, col9, col10 = st.columns(3)
        contract       = col8.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless      = col9.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = col10.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"]
        )

        st.subheader("Phone & Internet Services")
        col11, col12, col13 = st.columns(3)
        phone_service    = col11.selectbox("Phone Service",   ["Yes", "No"])
        multiple_lines   = col12.selectbox("Multiple Lines",  ["No", "Yes", "No phone service"])
        internet_service = col13.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

        st.subheader("Add-On Services")
        col14, col15, col16, col17 = st.columns(4)
        online_security = col14.selectbox("Online Security",   ["No", "Yes", "No internet service"])
        online_backup   = col15.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
        device_prot     = col16.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support    = col17.selectbox("Tech Support",      ["No", "Yes", "No internet service"])

        col18, col19 = st.columns(2)
        streaming_tv     = col18.selectbox("Streaming TV",     ["No", "Yes", "No internet service"])
        streaming_movies = col19.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        submitted = st.form_submit_button("Predict Churn", use_container_width=True)

    if submitted:
        customer_raw = {
            "gender":           gender,
            "SeniorCitizen":    "1" if senior == "Yes" else "0",
            "Partner":          partner,
            "Dependents":       dependents,
            "tenure":           int(tenure),
            "PhoneService":     phone_service,
            "MultipleLines":    multiple_lines,
            "InternetService":  internet_service,
            "OnlineSecurity":   online_security,
            "OnlineBackup":     online_backup,
            "DeviceProtection": device_prot,
            "TechSupport":      tech_support,
            "StreamingTV":      streaming_tv,
            "StreamingMovies":  streaming_movies,
            "Contract":         contract,
            "PaperlessBilling": paperless,
            "PaymentMethod":    payment_method,
            "MonthlyCharges":   float(monthly_charges),
            "TotalCharges":     float(total_charges),
        }

        try:
            customer_df = pd.DataFrame([customer_raw])
            customer_fe = engineer_features(customer_df)

            prob   = pipeline.predict_proba(customer_fe)[0, 1]
            risk   = classify_risk(prob)
            result = generate_retention_recommendations(customer_raw, prob, risk)

            st.markdown("---")
            st.subheader("Prediction Results")

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("CHURN PROBABILITY", "%.1f%%" % result["churn_probability"])
            with res_col2:
                risk_color = RISK_COLORS[risk]
                st.markdown(
                    "<h3 style='color:%s'>RISK LEVEL: %s</h3>" % (risk_color, risk),
                    unsafe_allow_html=True,
                )

            # Probability gauge bar
            prob_pct   = result["churn_probability"]
            bar_color  = RISK_COLORS[risk]
            st.markdown(
                "<div style='background:#eee;border-radius:10px;height:24px;width:100%;'>"
                "<div style='background:%s;width:%s%%;border-radius:10px;height:24px;"
                "text-align:center;color:white;font-weight:bold;line-height:24px;'>%s%%</div>"
                "</div>" % (bar_color, prob_pct, prob_pct),
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            col_rf, col_rec = st.columns(2)
            with col_rf:
                st.markdown("**Key Risk Factors**")
                for factor in result["risk_factors"]:
                    st.markdown("- %s" % factor)

            with col_rec:
                st.markdown("**Retention Recommendations**")
                for rec in result["recommendations"]:
                    st.markdown("- %s" % rec)

            st.info(
                "These are analytical suggestions based on model output and customer attributes. "
                "They are not guaranteed to prevent churn. Supplement with domain expertise and "
                "controlled experiments."
            )

        except Exception as exc:
            st.error("Prediction failed: %s" % exc)
            st.exception(exc)


# ==============================================================================
# PAGE 3 -- ANALYTICS
# ==============================================================================
def page_analytics(df, pipeline, metadata=None):
    st.title("Analytics")

    if df is None:
        st.error("Dataset not available.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "Churn Analysis", "Risk Distribution", "Feature Importance", "Model Performance"
    ])

    # Tab 1: Churn Analysis
    with tab1:
        st.subheader("Internet Service & Tech Support Analysis")
        col_a, col_b = st.columns(2)

        with col_a:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            isp_rate = df.groupby("InternetService")["Churn"].mean().mul(100)
            isp_rate.plot(kind="bar", ax=ax, color="mediumseagreen", edgecolor="white", rot=15)
            ax.set_title("Churn Rate by Internet Service")
            ax.set_ylabel("Churn Rate (%)")
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_b:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ts_rate = df.groupby("TechSupport")["Churn"].mean().mul(100)
            ts_rate.plot(kind="bar", ax=ax, color="teal", edgecolor="white", rot=15)
            ax.set_title("Churn Rate by Tech Support")
            ax.set_ylabel("Churn Rate (%)")
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Monthly Charges vs Churn")
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.hist(df[df["Churn"] == 0]["MonthlyCharges"], bins=30, alpha=0.6,
                color="steelblue", label="No Churn")
        ax.hist(df[df["Churn"] == 1]["MonthlyCharges"], bins=30, alpha=0.6,
                color="tomato", label="Churn")
        ax.set_xlabel("Monthly Charges ($)")
        ax.set_ylabel("Count")
        ax.set_title("Monthly Charges Distribution by Churn")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Tab 2: Risk Distribution
    with tab2:
        if pipeline is None:
            st.info("Run the notebook to generate the model, then reload the app.")
        else:
            try:
                df_fe = engineer_features(df.drop(columns=["Churn"], errors="ignore"))
                probs = pipeline.predict_proba(df_fe)[:, 1]
                risks = [classify_risk(p) for p in probs]
                risk_df = pd.DataFrame({
                    "Probability":  probs,
                    "Risk":         risks,
                    "Actual_Churn": df["Churn"].values,
                })

                st.subheader("Risk Category Distribution")
                col_a, col_b = st.columns(2)

                with col_a:
                    risk_counts  = risk_df["Risk"].value_counts()
                    risk_order   = ["LOW", "MEDIUM", "HIGH"]
                    risk_ordered = risk_counts.reindex(risk_order, fill_value=0)
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    ax.bar(risk_ordered.index, risk_ordered.values,
                           color=["steelblue", "gold", "tomato"], edgecolor="white")
                    for i, v in enumerate(risk_ordered.values):
                        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
                    ax.set_title("Risk Category Counts")
                    ax.set_ylabel("Customers")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                with col_b:
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    ax.hist(probs, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
                    ax.axvline(0.30, color="gold",  linestyle="--", lw=1.5, label="30% (MEDIUM)")
                    ax.axvline(0.60, color="tomato", linestyle="--", lw=1.5, label="60% (HIGH)")
                    ax.set_title("Churn Probability Distribution")
                    ax.set_xlabel("Churn Probability")
                    ax.set_ylabel("Count")
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                st.subheader("Risk Category vs Actual Churn")
                cross = pd.crosstab(risk_df["Risk"], risk_df["Actual_Churn"], margins=True)
                cross.columns = ["No Churn", "Churn", "Total"]
                st.dataframe(cross)

            except Exception as e:
                st.error("Risk analysis error: %s" % e)

    # Tab 3: Feature Importance
    with tab3:
        if pipeline is None:
            st.info("Run the notebook to generate the model, then reload the app.")
        else:
            try:
                NUMERICAL_FEATURES   = ["tenure", "MonthlyCharges", "TotalCharges",
                                        "ServiceCount", "ChargePerMonth"]
                CATEGORICAL_FEATURES = [
                    "gender", "SeniorCitizen", "Partner", "Dependents",
                    "PhoneService", "MultipleLines", "InternetService",
                    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies",
                    "Contract", "PaperlessBilling", "PaymentMethod",
                    "TenureGroup",
                ]
                preprocessor_ = pipeline.named_steps["preprocessor"]
                classifier    = pipeline.named_steps["classifier"]
                ohe = preprocessor_.named_transformers_["cat"].named_steps["onehot"]
                cat_names     = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
                feature_names = NUMERICAL_FEATURES + cat_names

                if hasattr(classifier, "feature_importances_"):
                    importances = classifier.feature_importances_
                    label = "Feature Importance"
                elif hasattr(classifier, "coef_"):
                    importances = np.abs(classifier.coef_[0])
                    label = "Absolute Coefficient"
                else:
                    importances = None

                if importances is not None:
                    fi_df = (
                        pd.DataFrame({"Feature": feature_names, label: importances})
                        .sort_values(label, ascending=False)
                        .head(20)
                    )
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.barplot(data=fi_df, y="Feature", x=label, ax=ax, palette="viridis")
                    ax.set_title("Top 20 Feature Importances")
                    ax.set_xlabel(label)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    st.dataframe(fi_df.reset_index(drop=True))
                else:
                    st.info("Feature importance not available for this model type.")

            except Exception as e:
                st.error("Feature importance error: %s" % e)

    # Tab 4: Model Performance
    with tab4:
        st.info(
            "Model evaluation metrics are computed during notebook execution. "
            "See the Model Information page for the full comparison table."
        )
        if metadata and metadata.get("all_metrics"):
            comp_df = pd.DataFrame(metadata["all_metrics"]).set_index("Model")
            st.subheader("Model Comparison (from last notebook run)")
            st.dataframe(comp_df.style.format("{:.4f}"))
            fig, ax = plt.subplots(figsize=(9, 4))
            comp_df.plot(kind="bar", ax=ax, edgecolor="white")
            ax.set_title("Model Performance Comparison")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1.1)
            ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
            ax.tick_params(axis="x", rotation=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


# ==============================================================================
# PAGE 4 -- MODEL INFORMATION
# ==============================================================================
def page_model_info(metadata):
    st.title("Model Information")

    if not metadata:
        st.warning(
            "Model metadata not available. "
            "Run SaiSnehan_AI_Customer_Churn_Prediction.ipynb first."
        )
        return

    st.subheader("Selected Model")
    st.success(
        "**%s** -- chosen by highest ROC-AUC score." % metadata.get("best_model_name", "N/A")
    )

    st.subheader("Evaluation Metrics (Selected Model on Test Set)")
    metrics = metadata.get("metrics", {})
    if metrics:
        m_df = pd.DataFrame([metrics])
        m_df.columns = [c.upper() for c in m_df.columns]
        st.dataframe(m_df.style.format("{:.4f}"))

    st.subheader("All Model Comparison")
    all_metrics = metadata.get("all_metrics", [])
    if all_metrics:
        comp_df = pd.DataFrame(all_metrics).set_index("Model")
        st.dataframe(comp_df.style.format("{:.4f}"))

        fig, ax = plt.subplots(figsize=(9, 4))
        comp_df.plot(kind="bar", ax=ax, edgecolor="white")
        ax.set_title("Model Performance Comparison")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.1)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Risk Threshold Configuration (Project-Defined)")
    thresholds = {
        "LOW":    "0% - 30%",
        "MEDIUM": "30% - 60%",
        "HIGH":   "60% - 100%",
    }
    st.table(pd.DataFrame.from_dict(thresholds, orient="index", columns=["Probability Range"]))
    st.caption(
        "These thresholds are project-defined for this internship project. "
        "They are not universal industry standards. Adjust based on business context."
    )

    st.subheader("Project Limitations")
    limitations = [
        "Model trained on historical data -- may not reflect current customer behaviour.",
        "Findings are specific to this Telco dataset and may not generalise.",
        "Class imbalance (~26% churn) affects model confidence.",
        "Risk thresholds are project-defined, not universal.",
        "The model identifies correlations, not causal relationships.",
        "No real-time behavioural data included.",
        "No hyperparameter optimisation performed.",
    ]
    for lim in limitations:
        st.markdown("- %s" % lim)

    st.subheader("IBM Bob Contribution")
    st.markdown(
        "IBM Bob was used as an AI-assisted development tool throughout this project for:\n"
        "- Code generation and structuring\n"
        "- Debugging data type and pipeline issues\n"
        "- Feature engineering suggestions\n"
        "- Documentation and Markdown explanation writing\n"
        "- Business insight interpretation\n"
        "- Streamlit layout design suggestions\n\n"
        "**IBM Bob did NOT train the machine-learning models.**  \n"
        "All model training was performed by Python/Scikit-learn "
        "using the Kaggle Telco Customer Churn dataset."
    )


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    st.markdown(
        "<h1 style='text-align:center;'>"
        "AI-Powered Customer Churn Prediction &amp; Retention Recommendation System"
        "</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:gray;'>"
        "IBM SkillsBuild Data Analytics with AI Academic Internship | "
        "BharatCares x AICTE | Student: Sai Snehan_K"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = sidebar()

    with st.spinner("Loading dataset and model..."):
        df, data_error       = load_dataset()
        pipeline, metadata, model_error = load_model()

    if data_error and page == NAV_DASHBOARD:
        st.warning(data_error)
    if model_error and page not in [NAV_DASHBOARD, NAV_ANALYTICS]:
        st.warning(model_error)

    if page == NAV_DASHBOARD:
        page_dashboard(df, pipeline, metadata)
    elif page == NAV_PREDICTION:
        page_prediction(pipeline)
    elif page == NAV_ANALYTICS:
        page_analytics(df, pipeline, metadata)
    elif page == NAV_MODEL_INFO:
        page_model_info(metadata)


if __name__ == "__main__":
    main()
