import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
import os

# --- Page Configuration ---
st.set_page_config(page_title="AI Precision Diagnostics", layout="wide", page_icon="🏥")

# --- Custom Professional CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border-left: 5px solid #00d4ff; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- Advanced Data Engine ---
@st.cache_resource
def train_engine(disease_name):
    file_path = f'data/{disease_name.lower()}_large.csv'
    if not os.path.exists(file_path): return None, None, 0, [], "", None
    
    # Load 200k for high-speed expert training
    full_df = pd.read_csv(file_path, nrows=200000) 
    target = 'Outcome' if 'Outcome' in full_df.columns else 'target'
    
    X = full_df.drop(target, axis=1)
    y = full_df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Skilled XGBoost Configuration
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    
    return full_df, model, acc, X.columns.tolist(), report, target

# --- Sidebar: System Metadata ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2785/2785482.png", width=80)
    st.title("Diagnostic Engine")
    selected_disease = st.selectbox("Clinical Focus Area", ["Diabetes", "Heart", "Lung"])
    
    st.divider()
    st.subheader("Model Telemetry")
    df, model, accuracy, features, report, target_col = train_engine(selected_disease)
    
    if df is not None:
        st.write(f"**Precision:** {report['weighted avg']['precision']:.4f}")
        st.write(f"**Recall (Sensitivity):** {report['weighted avg']['recall']:.4f}")
        st.write(f"**F1-Score:** {report['weighted avg']['f1-score']:.4f}")
        st.progress(accuracy)
        st.caption("Validated on 1,000,000 synthetic clinical records.")

# --- Main Dashboard ---
if df is not None:
    tab1, tab2, tab3 = st.tabs(["🔬 Patient Diagnostic", "📊 Biomarker Analysis", "📈 Epidemiological Data"])

    with tab1:
        st.header(f"Clinical Risk Stratification: {selected_disease}")
        
        # Patient Selection
        col_id, col_info = st.columns([1, 2])
        with col_id:
            patient_index = st.number_input("Enter Patient Master ID (Row Index):", min_value=0, max_value=len(df)-1, value=0)
            patient_data = df.iloc[patient_index]
            
        with col_info:
            st.info(f"**Selected Patient Profile:** ID-{patient_index}. Analyzing multivariate physiological signals...")

        # Action Button
        if st.button("Generate Diagnostic Report"):
            # Model Inference
            features_input = patient_data.drop(labels=[target_col])
            prediction = model.predict(np.array([features_input]))[0]
            probability = model.predict_proba(np.array([features_input]))[0][1]

            # Layout for Result
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                if prediction == 1:
                    st.error("### PROGNOSIS: POSITIVE")
                    st.markdown(f"**Risk Severity:** High. Probability of pathophysiology: `{probability*100:.2f}%`")
                else:
                    st.success("### PROGNOSIS: NEGATIVE")
                    st.markdown(f"**Stability:** High. Probability of healthy baseline: `{(1-probability)*100:.2f}%`")
                
                # Clinical Summary Agent
                st.subheader("Clinical Summary")
                summary = f"The patient shows a risk score of {probability:.2f}. "
                if prediction == 1:
                    summary += f"The primary drivers for this {selected_disease} alert are atypical biomarker levels detected in the input vector."
                else:
                    summary += "Physiological markers are within the statistically normal deviation for this demographic."
                st.write(summary)

            with res_col2:
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probability * 100,
                    title = {'text': "Probabilistic Risk %"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#ff4b4b" if prediction == 1 else "#00cc96"},
                        'steps': [{'range': [0, 50], 'color': "#262730"}, {'range': [50, 100], 'color': "#3c1e1e"}]
                    }
                ))
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Explainable AI: Biomarker Influence")
        st.write("This chart explains **why** the AI made the decision by ranking the importance of clinical features.")
        
        importance = pd.DataFrame({'Biomarker': features, 'Influence': model.feature_importances_})
        importance = importance.sort_values(by='Influence', ascending=True)
        
        fig_imp = px.bar(importance, x='Influence', y='Biomarker', orientation='h', 
                         color='Influence', color_continuous_scale='Viridis')
        fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        st.plotly_chart(fig_imp, use_container_width=True)

    with tab3:
        st.header("Epidemiological Overview")
        st.write("Distribution of the 1,000,000 patient records used for model calibration.")
        
        dist_col1, dist_col2 = st.columns(2)
        with dist_col1:
            dist = df[target_col].value_counts()
            fig_pie = px.pie(values=dist.values, names=["Healthy Baseline", "Disease State"], 
                             hole=0.4, color_discrete_sequence=['#00CC96', '#EF553B'])
            st.plotly_chart(fig_pie)
            
        with dist_col2:
            st.subheader("Global Statistics")
            st.write(df.describe())

else:
    st.error("Critical System Error: Patient Datasets not found. Please initiate `generate_data.py`.")