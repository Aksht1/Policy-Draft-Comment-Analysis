

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="Policy Draft Comment Analysis",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, minimal design
st.markdown("""
    <style>
    .main {
        max-width: 900px;
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .stTextArea > div > div > textarea {
        font-size: 16px;
        min-height: 150px;
    }
    h1 {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    h2 {
        color: #1f77b4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
        text-align: center;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.tfidf = None
    st.session_state.lr_model = None
    st.session_state.svm_model = None
    st.session_state.metrics = None

@st.cache_resource
def load_models():
    """Load the TF-IDF vectorizer and trained models."""
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        
        with open('lr_model.pkl', 'rb') as f:
            lr_model = pickle.load(f)
        
        with open('svm_model.pkl', 'rb') as f:
            svm_model = pickle.load(f)
        
        try:
            with open('model_metrics.pkl', 'rb') as f:
                metrics = pickle.load(f)
        except FileNotFoundError:
            metrics = None
        
        return tfidf, lr_model, svm_model, metrics
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.info("Please run 'save_models.py' first to train and save the models.")
        return None, None, None, None

# Load models
if not st.session_state.models_loaded:
    with st.spinner("Loading models..."):
        tfidf, lr_model, svm_model, metrics = load_models()
        if tfidf is not None:
            st.session_state.tfidf = tfidf
            st.session_state.lr_model = lr_model
            st.session_state.svm_model = svm_model
            st.session_state.metrics = metrics
            st.session_state.models_loaded = True

# Main title and description
st.title("📊 Policy Draft Comment Analysis")
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem; color: #666;'>
    <p style='font-size: 18px;'>
        Analyze public policy draft comments using machine learning
    </p>

</div>
""", unsafe_allow_html=True)

# Check if models are loaded
if not st.session_state.models_loaded:
    st.warning("⚠️ Models not loaded. Please ensure model files exist in the project directory.")
    st.stop()

# Sidebar for model selection and additional features
with st.sidebar:
    st.header("⚙️ Settings")
    
    model_choice = st.selectbox(
        "Select Model:",
        ["Support Vector Machine (SVM)", "Logistic Regression"],
        help="Choose the machine learning model for prediction"
    )
    
    st.markdown("---")
    st.header("📈 Model Performance")
    
    if st.session_state.metrics:
        selected_model_key = 'svm' if 'SVM' in model_choice else 'lr'
        model_metrics = st.session_state.metrics[selected_model_key]
        
        st.metric("Accuracy", f"{model_metrics['accuracy']:.3f}")
        st.metric("Precision", f"{model_metrics['precision']:.3f}")
        st.metric("Recall", f"{model_metrics['recall']:.3f}")
        st.metric("F1-Score", f"{model_metrics['f1']:.3f}")

# Main content area
tab1, tab2, tab3 = st.tabs(["🔍 Single Comment Analysis", "📁 Batch Analysis", "📊 Model Metrics"])

# Tab 1: Single Comment Analysis
with tab1:
    st.header("Analyze a Single Comment")
    
    comment_text = st.text_area(
        "Enter Policy Draft Comment:",
        placeholder="Type or paste a policy draft comment here...",
        height=150,
        help="Enter the text of a policy draft comment to analyze its stance"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_button = st.button("🔍 Analyze Comment", type="primary", use_container_width=True)
    
    if analyze_button:
        if not comment_text or not comment_text.strip():
            st.warning("⚠️ Please enter a comment before analyzing.")
        else:
            with st.spinner("Analyzing comment..."):
                # Transform text using TF-IDF
                text_vectorized = st.session_state.tfidf.transform([comment_text])
                
                # Get selected model
                if 'SVM' in model_choice:
                    model = st.session_state.svm_model
                    model_name = "SVM"
                else:
                    model = st.session_state.lr_model
                    model_name = "Logistic Regression"
                
                # Make prediction
                prediction = model.predict(text_vectorized)[0]
                
                # Get prediction probabilities (SVM uses decision_function, LR uses predict_proba)
                if hasattr(model, 'predict_proba'):
                    prediction_proba = model.predict_proba(text_vectorized)[0]
                elif hasattr(model, 'decision_function'):
                    # For SVM, convert decision function scores to probabilities
                    decision_scores = model.decision_function(text_vectorized)[0]
                    # Softmax-like normalization
                    exp_scores = np.exp(decision_scores - np.max(decision_scores))
                    prediction_proba = exp_scores / exp_scores.sum()
                else:
                    prediction_proba = None
                
                # Display result
                st.markdown("---")
                st.markdown("### 📋 Prediction Result")
                
                # Color coding based on prediction
                if prediction == "Support":
                    color = "#28a745"
                    emoji = "✅"
                elif prediction == "Oppose":
                    color = "#dc3545"
                    emoji = "❌"
                else:
                    color = "#ffc107"
                    emoji = "⚖️"
                
                st.markdown(f"""
                <div class="prediction-box" style="background-color: {color}20; border-left: 5px solid {color};">
                    <h2 style="color: {color}; margin: 0;">
                        {emoji} Predicted Stance: <strong>{prediction}</strong>
                    </h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Show prediction probabilities if available
                if prediction_proba is not None:
                    st.markdown("#### Prediction Probabilities:")
                    proba_df = pd.DataFrame({
                        'Stance': model.classes_,
                        'Probability': prediction_proba
                    }).sort_values('Probability', ascending=False)
                    
                    for idx, row in proba_df.iterrows():
                        st.progress(row['Probability'], text=f"{row['Stance']}: {row['Probability']:.2%}")
                
                st.markdown("---")
                st.markdown(f"**Model Used:** {model_name}")
                st.markdown(f"**Comment Length:** {len(comment_text)} characters")

# Tab 2: Batch Analysis
with tab2:
    st.header("Batch Analysis from CSV File")
    
    st.markdown("""
    Upload a CSV file containing policy comments. The file should have a column with comment text.
    The system will analyze all comments and display the results.
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with a column containing comment text"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File loaded successfully! Found {len(df)} rows.")
            
            # Let user select the text column
            text_column = st.selectbox(
                "Select the column containing comment text:",
                df.columns.tolist(),
                help="Choose which column contains the comments to analyze"
            )
            
            if st.button("🔍 Analyze All Comments", type="primary"):
                if text_column:
                    with st.spinner("Analyzing comments... This may take a moment."):
                        # Filter out empty comments
                        df_clean = df[df[text_column].astype(str).str.strip() != ''].copy()
                        
                        if len(df_clean) == 0:
                            st.warning("⚠️ No valid comments found in the selected column.")
                        else:
                            # Get selected model
                            if 'SVM' in model_choice:
                                model = st.session_state.svm_model
                            else:
                                model = st.session_state.lr_model
                            
                            # Transform and predict
                            comments = df_clean[text_column].astype(str).tolist()
                            comments_vectorized = st.session_state.tfidf.transform(comments)
                            predictions = model.predict(comments_vectorized)
                            
                            # Add predictions to dataframe
                            df_clean['Predicted_Stance'] = predictions
                            
                            # Display results
                            st.markdown("### 📊 Analysis Results")
                            
                            # Summary statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Comments", len(df_clean))
                            with col2:
                                support_count = (df_clean['Predicted_Stance'] == 'Support').sum()
                                st.metric("Support", support_count)
                            with col3:
                                oppose_count = (df_clean['Predicted_Stance'] == 'Oppose').sum()
                                st.metric("Oppose", oppose_count)
                            with col4:
                                neutral_count = (df_clean['Predicted_Stance'] == 'Neutral').sum()
                                st.metric("Neutral", neutral_count)
                            
                            # Distribution chart
                            st.markdown("#### Stance Distribution")
                            fig, ax = plt.subplots(figsize=(8, 5))
                            df_clean['Predicted_Stance'].value_counts().plot(kind='bar', ax=ax, color=['#28a745', '#dc3545', '#ffc107'])
                            ax.set_xlabel('Stance')
                            ax.set_ylabel('Count')
                            ax.set_title('Distribution of Predicted Stances')
                            plt.xticks(rotation=0)
                            st.pyplot(fig)
                            
                            # Display results table
                            st.markdown("#### Detailed Results")
                            display_cols = [text_column, 'Predicted_Stance']
                            if len(df_clean) > 100:
                                st.dataframe(df_clean[display_cols].head(100), use_container_width=True)
                                st.info(f"Showing first 100 of {len(df_clean)} results. Download the full results below.")
                            else:
                                st.dataframe(df_clean[display_cols], use_container_width=True)
                            
                            # Download button
                            csv = df_clean.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name="policy_comment_analysis_results.csv",
                                mime="text/csv"
                            )
                else:
                    st.warning("⚠️ Please select a text column.")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please ensure the file is a valid CSV file.")

# Tab 3: Model Metrics
with tab3:
    st.header("Model Performance Metrics")
    
    if st.session_state.metrics:
        # Model comparison
        st.markdown("### 📈 Model Comparison")
        
        comparison_data = {
            'Model': ['Logistic Regression', 'SVM'],
            'Accuracy': [
                st.session_state.metrics['lr']['accuracy'],
                st.session_state.metrics['svm']['accuracy']
            ],
            'Precision': [
                st.session_state.metrics['lr']['precision'],
                st.session_state.metrics['svm']['precision']
            ],
            'Recall': [
                st.session_state.metrics['lr']['recall'],
                st.session_state.metrics['svm']['recall']
            ],
            'F1-Score': [
                st.session_state.metrics['lr']['f1'],
                st.session_state.metrics['svm']['f1']
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Metrics visualization
        st.markdown("#### Performance Metrics Visualization")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy comparison
        axes[0, 0].bar(comparison_df['Model'], comparison_df['Accuracy'], color=['#1f77b4', '#ff7f0e'])
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Precision comparison
        axes[0, 1].bar(comparison_df['Model'], comparison_df['Precision'], color=['#1f77b4', '#ff7f0e'])
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision Comparison')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Recall comparison
        axes[1, 0].bar(comparison_df['Model'], comparison_df['Recall'], color=['#1f77b4', '#ff7f0e'])
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_title('Recall Comparison')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # F1-Score comparison
        axes[1, 1].bar(comparison_df['Model'], comparison_df['F1-Score'], color=['#1f77b4', '#ff7f0e'])
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].set_title('F1-Score Comparison')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Confusion matrices
        st.markdown("#### Confusion Matrices")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Logistic Regression")
            cm_lr = np.array(st.session_state.metrics['lr']['confusion_matrix'])
            labels = st.session_state.metrics['labels']
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title('Logistic Regression Confusion Matrix')
            st.pyplot(fig)
        
        with col2:
            st.markdown("##### Support Vector Machine")
            cm_svm = np.array(st.session_state.metrics['svm']['confusion_matrix'])
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Oranges',
                       xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title('SVM Confusion Matrix')
            st.pyplot(fig)
    else:
        st.info("Model metrics not available. Run 'save_models.py' to generate metrics.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px; padding: 1rem;'>
    <p>Policy Draft Comment Analysis System | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)

# python -m streamlit run app.py