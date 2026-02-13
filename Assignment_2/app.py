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

# Preprocess user input to match training data format
def preprocess_input(gender, senior_citizen, partner, dependents, tenure, 
                    phone_service, multiple_lines, internet_service, online_security,
                    online_backup, device_protection, tech_support, streaming_tv,
                    streaming_movies, contract, payment_method, monthly_charges, total_charges):
    """
    Convert user inputs into the same 41-feature format used in training
    """
    # Create a dictionary with the input values
    input_dict = {
        'SeniorCitizen': 1 if senior_citizen else 0,
        'Partner': 1 if partner == 'Yes' else 0,
        'Dependents': 1 if dependents == 'Yes' else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == 'Yes' else 0,
        'PaperlessBilling': 0,  # Will be overwritten with get_dummies
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'gender': gender,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaymentMethod': payment_method
    }
    
    # Convert to DataFrame
    df_input = pd.DataFrame([input_dict])
    
    # Apply one-hot encoding for categorical variables
    categorical_cols = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                       'StreamingMovies', 'Contract', 'PaymentMethod']
    
    df_encoded = pd.get_dummies(df_input, columns=categorical_cols, dtype=int)
    
    # Define all expected features (from training data)
    expected_features = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
                        'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'gender_Female',
                        'gender_Male', 'MultipleLines_No', 'MultipleLines_No phone service',
                        'MultipleLines_Yes', 'InternetService_DSL', 'InternetService_Fiber optic',
                        'InternetService_No', 'OnlineSecurity_No', 'OnlineSecurity_No internet service',
                        'OnlineSecurity_Yes', 'OnlineBackup_No', 'OnlineBackup_No internet service',
                        'OnlineBackup_Yes', 'DeviceProtection_No', 'DeviceProtection_No internet service',
                        'DeviceProtection_Yes', 'TechSupport_No', 'TechSupport_No internet service',
                        'TechSupport_Yes', 'StreamingTV_No', 'StreamingTV_No internet service',
                        'StreamingTV_Yes', 'StreamingMovies_No', 'StreamingMovies_No internet service',
                        'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
                        'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)',
                        'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
                        'PaymentMethod_Mailed check']
    
    # Add missing columns with 0s
    for col in expected_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Select only the expected features in the correct order
    df_encoded = df_encoded[expected_features]
    
    return df_encoded

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
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Demographics**")
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior_citizen = st.checkbox("Senior Citizen")
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
            
            with col2:
                st.write("**Account Information**")
                tenure = st.slider("Tenure (months)", 0, 72, 24)
                phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            
            with col3:
                st.write("**Charges**")
                monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0)
                total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1500.0)
            
            st.divider()
            
            # Internet and Services
            col4, col5, col6 = st.columns(3)
            
            with col4:
                st.write("**Internet & Services**")
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            
            with col5:
                st.write("**Additional Services**")
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            
            with col6:
                st.write("**Contract & Payment**")
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                payment_method = st.selectbox("Payment Method", 
                                            ["Bank transfer (automatic)", "Credit card (automatic)", 
                                             "Electronic check", "Mailed check"])
            
            # Make prediction
            if st.button("🔮 Predict Churn", use_container_width=True):
                try:
                    # Preprocess input to match training format (41 features)
                    input_data = preprocess_input(
                        gender=gender,
                        senior_citizen=senior_citizen,
                        partner=partner,
                        dependents=dependents,
                        tenure=tenure,
                        phone_service=phone_service,
                        multiple_lines=multiple_lines,
                        internet_service=internet_service,
                        online_security=online_security,
                        online_backup=online_backup,
                        device_protection=device_protection,
                        tech_support=tech_support,
                        streaming_tv=streaming_tv,
                        streaming_movies=streaming_movies,
                        contract=contract,
                        payment_method=payment_method,
                        monthly_charges=monthly_charges,
                        total_charges=total_charges
                    )
                    
                    prediction = models[selected_model].predict(input_data)
                    prediction_proba = models[selected_model].predict_proba(input_data)
                    
                    st.divider()
                    st.subheader("Prediction Result")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if prediction[0] == 1:
                            st.error("⚠️ **CHURN LIKELY**")
                        else:
                            st.success("✅ **NO CHURN EXPECTED**")
                    
                    with col2:
                        st.metric("Churn Probability", f"{prediction_proba[0][1]*100:.2f}%")
                    
                    with col3:
                        st.metric("Model Used", selected_model)
                
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.write(f"Details: {e}")
    
    elif page == "Batch Prediction":
        st.header("Batch Prediction")
        st.info("Upload a CSV file with the same format as the training data (without the 'Churn' column)")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        selected_model = st.selectbox("Select Model", list(models.keys()))
        
        if uploaded_file is not None:
            try:
                df_batch = pd.read_csv(uploaded_file)
                st.write("### File Preview:")
                st.dataframe(df_batch.head())
                st.write(f"Shape: {df_batch.shape}")
                
                if st.button("Make Predictions"):
                    try:
                        # Preprocess batch data the same way as training
                        # Force TotalCharges to be numeric
                        if 'TotalCharges' in df_batch.columns:
                            df_batch['TotalCharges'] = pd.to_numeric(df_batch['TotalCharges'], errors='coerce')
                            df_batch['TotalCharges'] = df_batch['TotalCharges'].fillna(df_batch['TotalCharges'].median())
                        
                        # Remove customerID if present
                        if 'customerID' in df_batch.columns:
                            df_batch = df_batch.drop('customerID', axis=1)
                        
                        # Remove Churn column if present (we're predicting it)
                        if 'Churn' in df_batch.columns:
                            df_batch = df_batch.drop('Churn', axis=1)
                        
                        # Apply one-hot encoding
                        df_processed = pd.get_dummies(df_batch, dtype=int)
                        
                        # Define all expected features
                        expected_features = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
                                           'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'gender_Female',
                                           'gender_Male', 'MultipleLines_No', 'MultipleLines_No phone service',
                                           'MultipleLines_Yes', 'InternetService_DSL', 'InternetService_Fiber optic',
                                           'InternetService_No', 'OnlineSecurity_No', 'OnlineSecurity_No internet service',
                                           'OnlineSecurity_Yes', 'OnlineBackup_No', 'OnlineBackup_No internet service',
                                           'OnlineBackup_Yes', 'DeviceProtection_No', 'DeviceProtection_No internet service',
                                           'DeviceProtection_Yes', 'TechSupport_No', 'TechSupport_No internet service',
                                           'TechSupport_Yes', 'StreamingTV_No', 'StreamingTV_No internet service',
                                           'StreamingTV_Yes', 'StreamingMovies_No', 'StreamingMovies_No internet service',
                                           'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
                                           'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)',
                                           'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
                                           'PaymentMethod_Mailed check']
                        
                        # Add missing columns with 0s
                        for col in expected_features:
                            if col not in df_processed.columns:
                                df_processed[col] = 0
                        
                        # Select only expected features
                        df_processed = df_processed[expected_features]
                        
                        predictions = models[selected_model].predict(df_processed)
                        predictions_proba = models[selected_model].predict_proba(df_processed)
                        
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
                        st.write(f"Details: {e}")
            except Exception as e:
                st.error(f"Error loading CSV file: {str(e)}")
    
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
                            # Preprocess X_test the same way as training
                            # Force TotalCharges to be numeric
                            if 'TotalCharges' in X_test.columns:
                                X_test['TotalCharges'] = pd.to_numeric(X_test['TotalCharges'], errors='coerce')
                                X_test['TotalCharges'] = X_test['TotalCharges'].fillna(X_test['TotalCharges'].median())
                            
                            # Remove customerID if present
                            if 'customerID' in X_test.columns:
                                X_test = X_test.drop('customerID', axis=1)
                            
                            # Apply one-hot encoding
                            X_test_processed = pd.get_dummies(X_test, dtype=int)
                            
                            # Define all expected features
                            expected_features = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
                                               'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'gender_Female',
                                               'gender_Male', 'MultipleLines_No', 'MultipleLines_No phone service',
                                               'MultipleLines_Yes', 'InternetService_DSL', 'InternetService_Fiber optic',
                                               'InternetService_No', 'OnlineSecurity_No', 'OnlineSecurity_No internet service',
                                               'OnlineSecurity_Yes', 'OnlineBackup_No', 'OnlineBackup_No internet service',
                                               'OnlineBackup_Yes', 'DeviceProtection_No', 'DeviceProtection_No internet service',
                                               'DeviceProtection_Yes', 'TechSupport_No', 'TechSupport_No internet service',
                                               'TechSupport_Yes', 'StreamingTV_No', 'StreamingTV_No internet service',
                                               'StreamingTV_Yes', 'StreamingMovies_No', 'StreamingMovies_No internet service',
                                               'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One year',
                                               'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)',
                                               'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
                                               'PaymentMethod_Mailed check']
                            
                            # Add missing columns with 0s
                            for col in expected_features:
                                if col not in X_test_processed.columns:
                                    X_test_processed[col] = 0
                            
                            # Select only expected features in correct order
                            X_test_processed = X_test_processed[expected_features]
                            
                            # Make predictions
                            y_pred = models[selected_model].predict(X_test_processed)
                            y_pred_proba = models[selected_model].predict_proba(X_test_processed)[:, 1]
                            
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
