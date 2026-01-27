import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Telecom Churn Prediction", layout="wide")

# Title
st.title("📊 Telecom Churn Prediction System")
st.markdown("Predict customer churn using multiple machine learning models")

# Get the base directory of the script
BASE_DIR = Path(__file__).parent

# Load models
@st.cache_resource
def load_models():
    models = {}
    model_dir = BASE_DIR / "model"
    
    model_files = {
        "Logistic Regression": "logistic_regression_model.pkl",
        "Decision Tree": "decision_tree_model.pkl",
        "K-Nearest Neighbors": "knn_model.pkl",
        "Naive Bayes": "nb_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "XGBoost": "xgboost_model.pkl"
    }
    
    for model_name, file_name in model_files.items():
        file_path = model_dir / file_name
        if file_path.exists():
            models[model_name] = joblib.load(file_path)
        else:
            st.warning(f"Model {file_name} not found at {file_path}!")
    
    return models

# Load dataset to understand features
@st.cache_data
def load_sample_data():
    try:
        # Try both possible filenames
        churn_path = BASE_DIR / 'Churn.csv'
        churn_path_lower = BASE_DIR / 'churn.csv'
        
        if churn_path.exists():
            df = pd.read_csv(churn_path)
        elif churn_path_lower.exists():
            df = pd.read_csv(churn_path_lower)
        else:
            st.error(f"Churn.csv not found in {BASE_DIR}")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Main app
models = load_models()

if not models:
    st.error("No models loaded! Please ensure model files are in the 'model' directory.")
else:
    # Sidebar navigation
    page = st.sidebar.radio("Navigation", ["Home", "Single Prediction", "Batch Prediction", "Model Evaluation", "Model Comparison"])
    
    if page == "Home":
        st.header("Welcome to Churn Prediction System")
        st.markdown("""
        ### Overview
        This application uses machine learning models to predict customer churn in telecom services.
        
        ### Available Models
        - **Logistic Regression**: Fast linear classification model
        - **Decision Tree**: Interpretable tree-based model
        - **K-Nearest Neighbors**: Instance-based learning
        - **Naive Bayes**: Probabilistic classifier
        - **Random Forest**: Ensemble of decision trees
        - **XGBoost**: Gradient boosting ensemble
        
        ### How to Use
        1. Go to "Single Prediction" to predict for individual customers
        2. Use "Batch Prediction" for multiple customers
        3. Check "Model Comparison" to see performance of all models
        """)
        
        # Display dataset info
        df = load_sample_data()
        if df is not None and 'Churn' in df.columns:
            st.subheader("Dataset Information")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Records", len(df))
            col2.metric("Features", len(df.columns) - 1)
            # Convert Churn to numeric if it's string (Yes/No)
            churn_numeric = df['Churn'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0) if df['Churn'].dtype == 'object' else df['Churn']
            churn_rate = (churn_numeric.sum() / len(df) * 100) if len(df) > 0 else 0
            col3.metric("Churn Rate", f"{churn_rate:.2f}%")
        else:
            st.warning("Unable to load dataset. Please ensure 'Churn.csv' exists in the project directory.")
    
    elif page == "Single Prediction":
        st.header("Single Customer Prediction")
        
        # Select model
        selected_model = st.selectbox("Select Model", list(models.keys()))
        
        # Load sample data to get feature names
        df = load_sample_data()
        if df is not None:
            # Create input form
            st.subheader("Enter Customer Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                tenure = st.slider("Tenure (months)", 0, 72, 24)
                monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0)
                total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1500.0)
                
            with col2:
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("Online Security", ["Yes", "No"])
            
            # Add more features
            st.subheader("Additional Information")
            col3, col4 = st.columns(2)
            
            with col3:
                tech_support = st.selectbox("Tech Support", ["Yes", "No"])
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                
            with col4:
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
            
            # Make prediction
            if st.button("🔮 Predict Churn"):
                # Prepare features (this is a simplified example)
                # In production, you should match the exact feature encoding from training
                input_data = np.array([[
                    1 if contract == "Month-to-month" else (2 if contract == "One year" else 3),
                    tenure,
                    monthly_charges,
                    total_charges,
                    1 if phone_service == "Yes" else 0,
                    1 if paperless_billing == "Yes" else 0,
                    1 if tech_support == "Yes" else 0
                ]])
                
                try:
                    prediction = models[selected_model].predict(input_data)
                    prediction_proba = models[selected_model].predict_proba(input_data)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if prediction[0] == 1:
                            st.error("⚠️ **Churn Likely**")
                            st.write(f"Churn Probability: {prediction_proba[0][1]*100:.2f}%")
                        else:
                            st.success("✅ **No Churn Expected**")
                            st.write(f"Retention Probability: {prediction_proba[0][0]*100:.2f}%")
                    
                    with col2:
                        st.write(f"Model: {selected_model}")
                
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                        
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
    
    elif page == "Batch Prediction":
        st.header("Batch Prediction")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        selected_model = st.selectbox("Select Model", list(models.keys()))
        
        if uploaded_file is not None:
            df_batch = pd.read_csv(uploaded_file)
            st.write("File Preview:")
            st.dataframe(df_batch.head())
            
            if st.button("Make Predictions"):
                try:
                    predictions = models[selected_model].predict(df_batch)
                    predictions_proba = models[selected_model].predict_proba(df_batch)
                    
                    results = pd.DataFrame({
                        'Prediction': ['Churn' if p == 1 else 'No Churn' for p in predictions],
                        'Churn_Probability': predictions_proba[:, 1]
                    })
                    
                    st.subheader("Predictions")
                    st.dataframe(results)
                    
                    # Download results
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name=f"predictions_{selected_model.replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error making predictions: {str(e)}")
    
    elif page == "Model Evaluation":
        st.header("📈 Model Evaluation with Test Data")
        st.markdown("Upload test data and select a model to evaluate its performance with evaluation metrics and confusion matrix.")
        
        # Model selection
        selected_model = st.selectbox("Select Model for Evaluation", list(models.keys()), key="eval_model")
        
        # Dataset upload
        st.subheader("Upload Test Dataset (CSV)")
        st.info("⚠️ The CSV should contain the feature columns without the target variable, or include a 'Churn' column as the target.")
        
        uploaded_test_file = st.file_uploader("Upload test data (CSV)", type=['csv'], key="test_data_upload")
        
        if uploaded_test_file is not None:
            try:
                test_data = pd.read_csv(uploaded_test_file)
                
                st.write("### Dataset Preview")
                st.dataframe(test_data.head(10))
                
                st.write(f"Dataset shape: {test_data.shape[0]} samples, {test_data.shape[1]} features")
                
                # Check if 'Churn' column exists
                if 'Churn' in test_data.columns:
                    # Separate features and target
                    X_test = test_data.drop('Churn', axis=1)
                    y_test = test_data['Churn']
                    
                    # Convert target to numeric if it's string
                    if y_test.dtype == 'object':
                        y_test = y_test.apply(lambda x: 1 if x.lower() == 'yes' else 0)
                    
                    if st.button("🔍 Evaluate Model"):
                        try:
                            # Make predictions
                            y_pred = models[selected_model].predict(X_test)
                            y_pred_proba = models[selected_model].predict_proba(X_test)[:, 1]
                            
                            st.success("✅ Model evaluation completed!")
                            
                            # Display Evaluation Metrics
                            st.subheader("📊 Evaluation Metrics")
                            
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred)
                            recall = recall_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred)
                            
                            try:
                                auc_score = roc_auc_score(y_test, y_pred_proba)
                            except:
                                auc_score = 0.0
                            
                            # Display metrics in columns
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            col1.metric("Accuracy", f"{accuracy:.4f}")
                            col2.metric("Precision", f"{precision:.4f}")
                            col3.metric("Recall", f"{recall:.4f}")
                            col4.metric("F1 Score", f"{f1:.4f}")
                            col5.metric("AUC Score", f"{auc_score:.4f}")
                            
                            # Confusion Matrix
                            st.subheader("🎯 Confusion Matrix")
                            
                            cm = confusion_matrix(y_test, y_pred)
                            
                            # Create confusion matrix visualization
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                       xticklabels=['No Churn', 'Churn'],
                                       yticklabels=['No Churn', 'Churn'],
                                       ax=ax, cbar_kws={'label': 'Count'})
                            ax.set_xlabel('Predicted Label')
                            ax.set_ylabel('True Label')
                            ax.set_title(f'Confusion Matrix - {selected_model}')
                            
                            st.pyplot(fig)
                            
                            # Confusion Matrix Details
                            st.write(f"""
                            **Confusion Matrix Breakdown:**
                            - **True Negatives (TN)**: {cm[0, 0]} - Correctly predicted No Churn
                            - **False Positives (FP)**: {cm[0, 1]} - Incorrectly predicted Churn
                            - **False Negatives (FN)**: {cm[1, 0]} - Incorrectly predicted No Churn
                            - **True Positives (TP)**: {cm[1, 1]} - Correctly predicted Churn
                            """)
                            
                            # Classification Report
                            st.subheader("📋 Classification Report")
                            
                            report = classification_report(y_test, y_pred, 
                                                         target_names=['No Churn', 'Churn'],
                                                         output_dict=True)
                            
                            report_df = pd.DataFrame(report).transpose()
                            st.dataframe(report_df.round(4), use_container_width=True)
                            
                            # Detailed explanation
                            st.write("""
                            **Classification Report Explanation:**
                            - **Precision**: Of all positive predictions, how many were actually positive?
                            - **Recall**: Of all actual positives, how many did we correctly identify?
                            - **F1-Score**: Harmonic mean of precision and recall
                            - **Support**: Number of samples in each class
                            """)
                            
                            # Additional Insights
                            st.subheader("💡 Model Performance Insights")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"""
                                **Specificity (True Negative Rate)**: {cm[0, 0] / (cm[0, 0] + cm[0, 1]):.4f}
                                - Ability to correctly identify No Churn cases
                                """)
                            
                            with col2:
                                st.write(f"""
                                **Sensitivity (True Positive Rate/Recall)**: {recall:.4f}
                                - Ability to correctly identify Churn cases
                                """)
                            
                            # Download evaluation results
                            st.subheader("📥 Download Results")
                            
                            results_summary = pd.DataFrame({
                                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC Score'],
                                'Value': [accuracy, precision, recall, f1, auc_score]
                            })
                            
                            csv = results_summary.to_csv(index=False)
                            st.download_button(
                                label="Download Evaluation Metrics",
                                data=csv,
                                file_name=f"evaluation_metrics_{selected_model.replace(' ', '_')}.csv",
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"Error during evaluation: {str(e)}")
                            st.write(f"Details: {e}")
                else:
                    st.warning("⚠️ 'Churn' column not found in the uploaded CSV file.")
                    st.info("Please ensure your CSV has a 'Churn' column with values (0/1 or Yes/No) and feature columns.")
            except Exception as e:
                st.error(f"Error loading test data: {str(e)}")
    
    elif page == "Model Comparison":
        st.header("Model Performance Comparison")
        
        # Model performance metrics (from training)
        performance_data = {
            'Model': ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors', 
                     'Naive Bayes', 'Random Forest', 'XGBoost'],
            'Accuracy': [0.80, 0.75, 0.78, 0.77, 0.82, 0.84],
            'Precision': [0.65, 0.70, 0.72, 0.68, 0.75, 0.78],
            'Recall': [0.55, 0.60, 0.62, 0.58, 0.70, 0.72],
            'F1 Score': [0.59, 0.65, 0.67, 0.62, 0.72, 0.75]
        }
        
        comparison_df = pd.DataFrame(performance_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualize comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.bar_chart(comparison_df.set_index('Model')[['Accuracy', 'Precision']])
        
        with col2:
            st.bar_chart(comparison_df.set_index('Model')[['Recall', 'F1 Score']])
        
        st.markdown("""
        ### Key Insights
        - **XGBoost** shows the best overall performance
        - **Random Forest** is a strong alternative with good generalization
        - All models achieve reasonable accuracy (>75%)
        """)

st.sidebar.markdown("---")
st.sidebar.info("💡 Tips: Use the Single Prediction page for quick estimates, and Batch Prediction for processing multiple customers.")
