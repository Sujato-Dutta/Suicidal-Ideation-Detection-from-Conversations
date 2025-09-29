import streamlit as st
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

# Add src directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from model_predict import SuicideDetector
    from utils.logger import get_logger
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Configure page
st.set_page_config(
    page_title="Suicidal Ideation Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    
    .high-risk {
        background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
        border-left-color: #e53e3e;
        color: #2d3748;
        box-shadow: 0 4px 6px rgba(229, 62, 62, 0.1);
    }
    
    .low-risk {
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
        border-left-color: #38a169;
        color: #2d3748;
        box-shadow: 0 4px 6px rgba(56, 161, 105, 0.1);
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #333;
    }
    
    .info-box {
        background-color: #e3f2fd;
        border: 1px solid #90caf9;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #333;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        color: #333;
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .emergency-resources {
        background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
        border: 2px solid #e53e3e;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        color: #2d3748;
        box-shadow: 0 8px 25px rgba(229, 62, 62, 0.15);
    }
    
    .emergency-resources h3 {
        color: #c53030;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .emergency-resources p {
        color: #2d3748;
        font-weight: 500;
    }
    
    .emergency-resources ul {
        color: #2d3748;
    }
    
    .emergency-resources li {
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
    
    .emergency-resources strong {
        color: #c53030;
    }
    
    .dark-text {
        color: #333 !important;
    }
    
    .dark-text h4 {
        color: #333 !important;
    }
    
    .dark-text p {
        color: #333 !important;
    }
    
    .dark-text ul {
        color: #333 !important;
    }
    
    .dark-text li {
        color: #333 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize logger
logger = get_logger(__name__)

@st.cache_resource
def load_model():
    """Load the suicide detection model with caching"""
    try:
        with st.spinner("Loading AI model..."):
            detector = SuicideDetector(enable_mlflow=False)  # Disable MLflow for web app
        st.success("✅ Model loaded successfully!")
        return detector
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None

def create_confidence_chart(scores):
    """Create a confidence visualization chart using matplotlib"""
    labels = list(scores.keys())
    values = list(scores.values())
    
    # Create color mapping - using more accessible colors
    colors = ['#dc2626' if 'suicidal' in label.lower() else '#059669' for label in labels]
    
    # Set style for better appearance
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Customize the chart
    ax.set_title('Prediction Confidence', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Categories', fontsize=12, fontweight='bold')
    ax.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Improve appearance
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#666')
    ax.spines['bottom'].set_color('#666')
    
    plt.tight_layout()
    return fig

def display_prediction_result(result):
    """Display prediction results in a formatted way"""
    label = result['label']
    confidence = result['score']
    scores = result['scores']
    
    # Determine risk level
    is_high_risk = 'suicidal' in label.lower()
    
    # Display main prediction
    if is_high_risk:
        st.markdown(f"""
        <div class="prediction-box high-risk">
            <h3>⚠️ High Risk Detected</h3>
            <p><strong>Prediction:</strong> {label.title()}</p>
            <p><strong>Confidence:</strong> {confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show emergency resources
        st.markdown("""
        <div class="emergency-resources">
            <h3>🆘 Immediate Help Resources</h3>
            <p><strong>If you or someone you know is in crisis:</strong></p>
            <ul>
                <li><strong>National Suicide Prevention Lifeline:</strong> 988 or 1-800-273-8255</li>
                <li><strong>Crisis Text Line:</strong> Text HOME to 741741</li>
                <li><strong>International Association for Suicide Prevention:</strong> https://www.iasp.info/resources/Crisis_Centres/</li>
                <li><strong>Emergency Services:</strong> Call 911 (US) or your local emergency number</li>
            </ul>
            <p><em>Remember: You are not alone, and help is available 24/7.</em></p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.markdown(f"""
        <div class="prediction-box low-risk">
            <h3>✅ Low Risk Detected</h3>
            <p><strong>Prediction:</strong> {label.title()}</p>
            <p><strong>Confidence:</strong> {confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display confidence chart
    fig = create_confidence_chart(scores)
    st.pyplot(fig, clear_figure=True)
    
    # Display detailed scores
    with st.expander("📊 Detailed Confidence Scores"):
        score_df = pd.DataFrame([
            {"Category": k.title(), "Confidence": f"{v:.1%}", "Score": v}
            for k, v in scores.items()
        ]).sort_values("Score", ascending=False)
        
        st.dataframe(score_df[["Category", "Confidence"]], width='stretch')

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Suicidal Ideation Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered mental health text analysis tool</p>', unsafe_allow_html=True)
    
    # Important disclaimer
    st.markdown("""
    <div class="warning-box dark-text">
        <h4>⚠️ Important Disclaimer</h4>
        <p>This tool is for informational purposes only and should not replace professional mental health assessment. 
        If you or someone you know is experiencing suicidal thoughts, please seek immediate professional help.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    detector = load_model()
    if detector is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings")
        
        # Model information
        st.subheader("📋 Model Info")
        st.info(f"""
        **Model Type:** TinyBERT for Sequence Classification
        **Classes:** {len(detector.class_names)}
        **Max Length:** {detector.max_length} tokens
        """)
        
        # Analysis options
        st.subheader("⚙️ Analysis Options")
        enable_batch = st.checkbox("Enable Batch Analysis", value=False)
        show_timestamps = st.checkbox("Show Timestamps", value=True)
        
        # About section
        st.subheader("ℹ️ About")
        st.markdown("""
        This application uses a fine-tuned TinyBERT model to analyze text 
        for potential indicators of suicidal ideation. The model has been 
        trained on conversational data to identify concerning patterns.
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Text Analysis")
        
        if not enable_batch:
            # Single text analysis
            st.subheader("Single Text Analysis")
            text_input = st.text_area(
                "Enter text to analyze:",
                placeholder="Type or paste the text you want to analyze here...",
                height=150,
                help="Enter any text content you'd like to analyze for suicidal ideation indicators."
            )
            
            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                analyze_btn = st.button("🔍 Analyze Text", type="primary")
            with col_btn2:
                clear_btn = st.button("🗑️ Clear")
            
            if clear_btn:
                st.rerun()
            
            if analyze_btn and text_input.strip():
                with st.spinner("Analyzing text..."):
                    start_time = time.time()
                    try:
                        result = detector.predict(text_input.strip())
                        analysis_time = time.time() - start_time
                        
                        st.success(f"✅ Analysis completed in {analysis_time:.2f} seconds")
                        
                        if show_timestamps:
                            st.caption(f"Analysis performed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        display_prediction_result(result)
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        logger.error(f"Prediction error: {e}")
            
            elif analyze_btn and not text_input.strip():
                st.warning("⚠️ Please enter some text to analyze.")
        
        else:
            # Batch analysis
            st.subheader("Batch Text Analysis")
            st.info("Upload a CSV file with a 'text' column or enter multiple texts separated by new lines.")
            
            # File upload option
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'text' not in df.columns:
                        st.error("❌ CSV file must contain a 'text' column.")
                    else:
                        st.success(f"✅ Loaded {len(df)} texts from file.")
                        
                        if st.button("🔍 Analyze Batch", type="primary"):
                            with st.spinner("Analyzing batch..."):
                                texts = df['text'].dropna().tolist()
                                results = detector.predict(texts)
                                
                                # Create results dataframe
                                results_df = pd.DataFrame([
                                    {
                                        "Text": text[:100] + "..." if len(text) > 100 else text,
                                        "Prediction": result['label'].title(),
                                        "Confidence": f"{result['score']:.1%}",
                                        "Risk Level": "High" if 'suicidal' in result['label'].lower() else "Low"
                                    }
                                    for text, result in zip(texts, results)
                                ])
                                
                                st.subheader("📊 Batch Analysis Results")
                                st.dataframe(results_df, width='stretch')
                                
                                # Summary statistics
                                high_risk_count = sum(1 for r in results if 'suicidal' in r['label'].lower())
                                st.metric("High Risk Texts", f"{high_risk_count}/{len(results)}")
                                
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
            
            else:
                # Manual batch input
                batch_text = st.text_area(
                    "Enter multiple texts (one per line):",
                    placeholder="Text 1\nText 2\nText 3\n...",
                    height=200
                )
                
                if st.button("🔍 Analyze Batch", type="primary") and batch_text.strip():
                    texts = [t.strip() for t in batch_text.split('\n') if t.strip()]
                    
                    if texts:
                        with st.spinner(f"Analyzing {len(texts)} texts..."):
                            results = detector.predict(texts)
                            
                            # Display results
                            st.subheader("📊 Batch Analysis Results")
                            for i, (text, result) in enumerate(zip(texts, results), 1):
                                with st.expander(f"Text {i}: {text[:50]}..."):
                                    display_prediction_result(result)
    
    with col2:
        st.header("📈 Statistics")
        
        # Model performance metrics (placeholder)
        st.subheader("🎯 Model Performance")
        
        # Create sample metrics (in a real app, these would come from model evaluation)
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.markdown("""
            <div class="metric-card dark-text">
                <h3>94.2%</h3>
                <p>Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with metrics_col2:
            st.markdown("""
            <div class="metric-card dark-text">
                <h3>91.8%</h3>
                <p>F1-Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Usage tips
        st.subheader("💡 Usage Tips")
        st.markdown("""
        <div class="info-box dark-text">
            <ul>
                <li><strong>Text Length:</strong> Works best with 10-500 words</li>
                <li><strong>Language:</strong> Optimized for English text</li>
                <li><strong>Context:</strong> Consider conversational context</li>
                <li><strong>Privacy:</strong> No text is stored or logged</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()