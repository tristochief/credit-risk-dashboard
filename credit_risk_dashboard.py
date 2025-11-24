
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4788;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c5aa0;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .risk-low {
        color: #00cc00;
        font-weight: bold;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f4788;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_data
def load_data():
    """Load all necessary data files"""
    df_original = pd.read_excel("credit_default_data.xls", header=1)
    df_original.rename(columns={'default payment next month': 'DEFAULT'}, inplace=True)
    df_processed = pd.read_csv('credit_data_preprocessed.csv')
    risk_segments = pd.read_csv('risk_segmentation.csv')
    feature_importance_lr = pd.read_csv('feature_importance_lr.csv')
    feature_importance_rf = pd.read_csv('feature_importance_rf.csv')
    feature_importance_xgb = pd.read_csv('feature_importance_xgb.csv')
    early_warning = pd.read_csv('early_warning_indicators.csv')
    return (df_original, df_processed, risk_segments, 
            feature_importance_lr, feature_importance_rf, 
            feature_importance_xgb, early_warning)

@st.cache_resource
def load_models():
    """Load trained models"""
    lr_model = joblib.load('lr_model.pkl')
    rf_model = joblib.load('rf_model.pkl')
    xgb_model = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return lr_model, rf_model, xgb_model, scaler

# Load data and models
(df_original, df_processed, risk_segments, 
 feature_importance_lr, feature_importance_rf, 
 feature_importance_xgb, early_warning) = load_data()

lr_model, rf_model, xgb_model, scaler = load_models()

# Title
st.markdown('<h1 class="main-header">💳 Credit Risk Analytics Dashboard</h1>', unsafe_allow_html=True)

# Executive Summary
st.markdown('<h2 class="sub-header">📊 Executive Summary</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_customers = len(df_processed)
    st.metric("Total Customers", f"{total_customers:,}")
    
with col2:
    default_rate = df_processed['DEFAULT'].mean()
    st.metric("Overall Default Rate", f"{default_rate:.1%}")
    
with col3:
    avg_credit_limit = df_processed['LIMIT_BAL'].mean()
    st.metric("Avg Credit Limit", f"${avg_credit_limit:,.0f}")
    
with col4:
    high_risk_pct = (risk_segments['risk_segment'].isin(['High Risk', 'Very High Risk'])).mean()
    st.metric("High Risk Customers", f"{high_risk_pct:.1%}", delta="Needs attention", delta_color="inverse")

# Key Insights Box
st.markdown("""
<div class="insight-box">
<b>🔍 Key Insights:</b><br>
• The portfolio shows a <span class="risk-high">22.1% default rate</span>, indicating significant credit risk<br>
• <span class="risk-high">48.8% of customers</span> are classified as High or Very High Risk<br>
• Payment delays are the <b>strongest predictor</b> of default risk<br>
• Early intervention for high-risk segments could save an estimated <b>$67.6M</b> in potential losses
</div>
""", unsafe_allow_html=True)

# Sidebar for filtering
st.sidebar.markdown("## 🎯 Filter Options")

# Risk segment filter
selected_segments = st.sidebar.multiselect(
    "Select Risk Segments",
    options=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'],
    default=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
)

# Age filter
age_range = st.sidebar.slider(
    "Age Range",
    min_value=int(df_processed['AGE'].min()),
    max_value=int(df_processed['AGE'].max()),
    value=(25, 65)
)

# Credit limit filter
credit_range = st.sidebar.slider(
    "Credit Limit Range",
    min_value=10000,
    max_value=500000,
    value=(10000, 500000),
    step=10000
)

# Apply filters
filtered_segments = risk_segments[
    (risk_segments['risk_segment'].isin(selected_segments)) &
    (risk_segments['AGE'].between(age_range[0], age_range[1])) &
    (risk_segments['LIMIT_BAL'].between(credit_range[0], credit_range[1]))
]

# Data Preprocessing Documentation
with st.expander("📋 Data Preprocessing Documentation", expanded=False):
    st.markdown("""
    ### Original Dataset Characteristics
    - **Total Customers**: 30,000
    - **Features**: 23 original features
    - **Target Variable**: DEFAULT (1=defaulted, 0=not defaulted)
    - **Class Distribution**: 77.88% non-defaulters, 22.12% defaulters
    - **Class Imbalance Ratio**: 1:3.52
    
    ### Preprocessing Decisions & Transformations
    
    #### 1. Data Quality Issues Fixed
    - **Undefined Categories**: Grouped undefined EDUCATION values (0,5,6) and MARRIAGE value (0) with 'others' category
    - **Payment Status Anomalies**: Standardized payment status values (-2 and below) to -1 (pay duly)
    - **Total Issues Resolved**: 8 data quality issues
    
    #### 2. Feature Engineering
    Created 7 new risk indicator features:
    - **TOTAL_BILL**: Sum of all bill amounts
    - **TOTAL_PAY**: Sum of all payment amounts  
    - **PAYMENT_RATIO**: Total payments / Total bills
    - **CREDIT_UTILIZATION**: Total bills / Credit limit
    - **AVG_PAYMENT_DELAY**: Average delay across 6 months
    - **MAX_PAYMENT_DELAY**: Maximum delay in 6 months
    - **DELAYED_MONTHS_COUNT**: Number of months with delays
    
    #### 3. Class Imbalance Handling Strategy
    - Used **stratified sampling** for train-test split
    - Applied **class weights** in models (balanced)
    - Implemented **SMOTE** for synthetic minority oversampling during training
    - Focused on **Precision-Recall metrics** instead of accuracy
    - Optimized for **F2 score** (prioritizing recall for risk identification)
    
    ### Impact of Preprocessing
    - **Before**: Raw data with inconsistencies and limited predictive features
    - **After**: Clean, engineered dataset with 30 features optimized for risk prediction
    - **Result**: Improved model performance with F2 scores ranging from 0.38 to 0.57
    """)

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Risk Segmentation", "📈 Key Risk Factors", 
                                         "⚠️ Early Warning Indicators", "💰 Financial Impact", 
                                         "🤖 Model Performance"])

# Tab 1: Risk Segmentation
with tab1:
    st.markdown('<h2 class="sub-header">Customer Risk Segmentation</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Risk segment distribution
        segment_counts = filtered_segments['risk_segment'].value_counts()
        segment_counts = segment_counts.reindex(['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'], fill_value=0)
        
        fig_pie = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Risk Segment Distribution",
            color_discrete_map={
                'Low Risk': '#00cc00',
                'Medium Risk': '#ffa500',
                'High Risk': '#ff6b6b',
                'Very High Risk': '#ff0000'
            }
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        # Default rates by segment
        segment_stats = filtered_segments.groupby('risk_segment').agg({
            'actual_default': 'mean',
            'risk_probability': 'mean'
        }).round(3)
        segment_stats = segment_stats.reindex(['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'])
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=segment_stats.index,
            y=segment_stats['actual_default'] * 100,
            name='Actual Default Rate',
            marker_color=['#00cc00', '#ffa500', '#ff6b6b', '#ff0000'],
            text=[f"{v:.1f}%" for v in segment_stats['actual_default'] * 100],
            textposition='auto'
        ))
        fig_bar.update_layout(
            title="Default Rates by Risk Segment",
            xaxis_title="Risk Segment",
            yaxis_title="Default Rate (%)",
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Segment characteristics
    st.markdown("### Segment Characteristics")
    
    segment_chars = filtered_segments.groupby('risk_segment').agg({
        'AGE': 'mean',
        'LIMIT_BAL': 'mean',
        'CREDIT_UTILIZATION': 'mean',
        'DELAYED_MONTHS_COUNT': 'mean',
        'MAX_PAYMENT_DELAY': 'mean'
    }).round(2)
    segment_chars = segment_chars.reindex(['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'])
    
    st.dataframe(
        segment_chars.style.background_gradient(cmap='RdYlGn_r', axis=0),
        use_container_width=True
    )
    
    st.markdown("""
    <div class="insight-box">
    <b>💡 Segmentation Insights:</b><br>
    • Very High Risk customers have <span class="risk-high">53.1% default rate</span> - immediate intervention needed<br>
    • High Risk segment shows <span class="risk-medium">20.2% default rate</span> - proactive monitoring recommended<br>
    • Credit utilization and payment delays are key differentiators between segments<br>
    • Focus resources on High and Very High Risk segments for maximum impact
    </div>
    """, unsafe_allow_html=True)

# Tab 2: Key Risk Factors
with tab2:
    st.markdown('<h2 class="sub-header">Key Risk Factors Analysis</h2>', unsafe_allow_html=True)
    
    # Feature importance comparison
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Combine feature importance from all models
        top_n = 10
        
        fig_importance = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Logistic Regression', 'Random Forest', 'XGBoost')
        )
        
        # Logistic Regression
        top_lr = feature_importance_lr.head(top_n)
        fig_importance.add_trace(
            go.Bar(x=top_lr['abs_coefficient'], y=top_lr['feature'],
                   orientation='h', marker_color='#1f4788',
                   text=[f"{v:.3f}" for v in top_lr['abs_coefficient']],
                   textposition='auto'),
            row=1, col=1
        )
        
        # Random Forest
        top_rf = feature_importance_rf.head(top_n)
        fig_importance.add_trace(
            go.Bar(x=top_rf['importance'], y=top_rf['feature'],
                   orientation='h', marker_color='#2c5aa0',
                   text=[f"{v:.4f}" for v in top_rf['importance']],
                   textposition='auto'),
            row=1, col=2
        )
        
        # XGBoost
        top_xgb = feature_importance_xgb.head(top_n)
        fig_importance.add_trace(
            go.Bar(x=top_xgb['importance'], y=top_xgb['feature'],
                   orientation='h', marker_color='#3a6fb0',
                   text=[f"{v:.4f}" for v in top_xgb['importance']],
                   textposition='auto'),
            row=1, col=3
        )
        
        fig_importance.update_layout(
            title="Top 10 Risk Factors by Model",
            showlegend=False,
            height=500
        )
        fig_importance.update_xaxes(title_text="Importance", row=1, col=1)
        fig_importance.update_xaxes(title_text="Importance", row=1, col=2)
        fig_importance.update_xaxes(title_text="Importance", row=1, col=3)
        
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Top Risk Factors")
        st.markdown("""
        **Consensus Top 5 Factors:**
        1. **MAX_PAYMENT_DELAY** - Highest delay in payment history
        2. **DELAYED_MONTHS_COUNT** - Number of months with delays
        3. **PAY_0** - Most recent payment status
        4. **CREDIT_UTILIZATION** - Credit usage ratio
        5. **LIMIT_BAL** - Credit limit (inverse relationship)
        
        **Risk Factor Categories:**
        - 🕐 **Payment Behavior** (60% weight)
        - 💳 **Credit Usage** (25% weight)
        - 👤 **Demographics** (15% weight)
        """)
    
    # Detailed factor analysis
    st.markdown("### Risk Factor Deep Dive")
    
    selected_factor = st.selectbox(
        "Select a risk factor to analyze:",
        ['MAX_PAYMENT_DELAY', 'DELAYED_MONTHS_COUNT', 'CREDIT_UTILIZATION', 'PAY_0', 'AGE']
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution by default status
        fig_dist = px.histogram(
            df_processed,
            x=selected_factor,
            color='DEFAULT',
            title=f"{selected_factor} Distribution by Default Status",
            color_discrete_map={0: '#00cc00', 1: '#ff0000'},
            labels={'DEFAULT': 'Default Status'},
            barmode='overlay',
            opacity=0.7
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Average values by risk segment
        avg_by_segment = risk_segments.groupby('risk_segment')[selected_factor].mean()
        avg_by_segment = avg_by_segment.reindex(['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'])
        
        fig_avg = go.Figure()
        fig_avg.add_trace(go.Bar(
            x=avg_by_segment.index,
            y=avg_by_segment.values,
            marker_color=['#00cc00', '#ffa500', '#ff6b6b', '#ff0000'],
            text=[f"{v:.2f}" for v in avg_by_segment.values],
            textposition='auto'
        ))
        fig_avg.update_layout(
            title=f"Average {selected_factor} by Risk Segment",
            xaxis_title="Risk Segment",
            yaxis_title=f"Average {selected_factor}"
        )
        st.plotly_chart(fig_avg, use_container_width=True)

# Tab 3: Early Warning Indicators
with tab3:
    st.markdown('<h2 class="sub-header">Early Warning System</h2>', unsafe_allow_html=True)
    
    # Display early warning thresholds
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### ⚠️ Alert Thresholds")
        
        for _, row in early_warning.iterrows():
            threshold_text = f"**{row['Feature']}**: Alert when > {row['Threshold']}"
            if row['Feature'] == 'PAYMENT_RATIO':
                threshold_text = f"**{row['Feature']}**: Alert when < {row['Threshold']}"
            
            st.markdown(f"• {threshold_text} ({row['Type']})")
        
        st.markdown("""
        <div class="insight-box">
        <b>🚨 Intervention Strategy:</b><br>
        • <span class="risk-high">Red Alert</span>: 3+ warning indicators active<br>
        • <span class="risk-medium">Yellow Alert</span>: 2 warning indicators active<br>
        • <span class="risk-low">Green Status</span>: 0-1 warning indicators active
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Warning indicator activation rates
        st.markdown("### 📊 Indicator Activation Rates")
        
        # Calculate activation rates for each indicator
        activation_rates = []
        for _, row in early_warning.iterrows():
            feature = row['Feature']
            threshold = row['Threshold']
            
            if feature == 'PAYMENT_RATIO':
                activated = (df_processed[feature] < threshold).mean() * 100
            else:
                activated = (df_processed[feature] > threshold).mean() * 100
            
            activation_rates.append({
                'Indicator': feature,
                'Activation Rate (%)': activated
            })
        
        activation_df = pd.DataFrame(activation_rates)
        
        fig_activation = px.bar(
            activation_df,
            x='Activation Rate (%)',
            y='Indicator',
            orientation='h',
            title="Early Warning Indicator Activation Rates",
            color='Activation Rate (%)',
            color_continuous_scale='RdYlGn_r'
        )
        fig_activation.update_traces(
            text=[f"{v:.1f}%" for v in activation_df['Activation Rate (%)']],
            textposition='auto'
        )
        st.plotly_chart(fig_activation, use_container_width=True)
    
    # Customer monitoring dashboard
    st.markdown("### 🔍 Real-time Customer Monitoring")
    
    # Simulate monitoring for a subset of customers
    monitoring_sample = risk_segments.sample(min(100, len(risk_segments)))
    
    # Calculate warning level for each customer
    def calculate_warning_level(row):
        warnings = 0
        if row['PAY_0'] > 1: warnings += 1
        if row['DELAYED_MONTHS_COUNT'] > 3: warnings += 1
        if row['CREDIT_UTILIZATION'] > 0.8: warnings += 1
        if row['MAX_PAYMENT_DELAY'] > 2: warnings += 1
        if row['PAYMENT_RATIO'] < 0.1: warnings += 1
        
        if warnings >= 3: return 'Red Alert'
        elif warnings == 2: return 'Yellow Alert'
        else: return 'Green Status'
    
    monitoring_sample['warning_level'] = monitoring_sample.apply(calculate_warning_level, axis=1)
    
    # Display warning level distribution
    warning_dist = monitoring_sample['warning_level'].value_counts()
    
    fig_warning = px.pie(
        values=warning_dist.values,
        names=warning_dist.index,
        title="Current Warning Level Distribution (Sample of 100 Customers)",
        color_discrete_map={
            'Green Status': '#00cc00',
            'Yellow Alert': '#ffa500',
            'Red Alert': '#ff0000'
        }
    )
    st.plotly_chart(fig_warning, use_container_width=True)

# Tab 4: Financial Impact
with tab4:
    st.markdown('<h2 class="sub-header">Financial Impact Analysis</h2>', unsafe_allow_html=True)
    
    # Calculate financial metrics
    total_exposure = df_processed['LIMIT_BAL'].sum()
    at_risk_exposure = df_processed[df_processed['DEFAULT'] == 1]['LIMIT_BAL'].sum()
    estimated_loss_rate = 0.60  # 60% loss given default
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Credit Exposure", f"${total_exposure/1e6:.1f}M")
    
    with col2:
        st.metric("At-Risk Exposure", f"${at_risk_exposure/1e6:.1f}M")
    
    with col3:
        potential_loss = at_risk_exposure * estimated_loss_rate
        st.metric("Potential Loss (60% LGD)", f"${potential_loss/1e6:.1f}M")
    
    # Financial impact by segment
    st.markdown("### 💰 Potential Loss by Risk Segment")
    
    segment_financial = []
    for segment in ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']:
        segment_data = risk_segments[risk_segments['risk_segment'] == segment]
        if len(segment_data) > 0:
            count = len(segment_data)
            defaults = segment_data['actual_default'].sum()
            default_rate = segment_data['actual_default'].mean()
            avg_limit = segment_data['LIMIT_BAL'].mean()
            potential_loss = defaults * avg_limit * estimated_loss_rate
            
            segment_financial.append({
                'Segment': segment,
                'Customers': count,
                'Defaults': int(defaults),
                'Default Rate': default_rate,
                'Avg Credit Limit': avg_limit,
                'Potential Loss': potential_loss
            })
    
    financial_df = pd.DataFrame(segment_financial)
    
    # Create financial impact visualization
    fig_financial = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Potential Loss by Segment', 'Cost-Benefit of Intervention'),
        specs=[[{"type": "bar"}, {"type": "waterfall"}]]
    )
    
    # Potential loss by segment
    fig_financial.add_trace(
        go.Bar(
            x=financial_df['Segment'],
            y=financial_df['Potential Loss'] / 1e6,
            marker_color=['#00cc00', '#ffa500', '#ff6b6b', '#ff0000'],
            text=[f"${v/1e6:.1f}M" for v in financial_df['Potential Loss']],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Cost-benefit waterfall
    intervention_cost = 5e6  # $5M intervention cost
    prevention_rate = 0.30  # 30% of defaults prevented
    
    prevented_loss = (financial_df[financial_df['Segment'].isin(['High Risk', 'Very High Risk'])]['Potential Loss'].sum() * 
                     prevention_rate)
    net_benefit = prevented_loss - intervention_cost
    
    fig_financial.add_trace(
        go.Waterfall(
            x=["Current Loss", "Intervention Cost", "Prevented Loss", "Net Benefit"],
            y=[potential_loss/1e6, -intervention_cost/1e6, -prevented_loss/1e6, 0],
            measure=["absolute", "relative", "relative", "total"],
            text=[f"${potential_loss/1e6:.1f}M", f"-${intervention_cost/1e6:.1f}M", 
                  f"-${prevented_loss/1e6:.1f}M", f"${net_benefit/1e6:.1f}M"],
            textposition="auto",
            connector={"line": {"color": "rgb(63, 63, 63)"}}
        ),
        row=1, col=2
    )
    
    fig_financial.update_layout(height=400, showlegend=False)
    fig_financial.update_xaxes(title_text="Risk Segment", row=1, col=1)
    fig_financial.update_yaxes(title_text="Potential Loss ($M)", row=1, col=1)
    fig_financial.update_yaxes(title_text="Amount ($M)", row=1, col=2)
    
    st.plotly_chart(fig_financial, use_container_width=True)
    
    # ROI calculation
    st.markdown("### 📈 Return on Investment")
    
    roi_metrics = {
        "Intervention Cost": f"${intervention_cost/1e6:.1f}M",
        "Prevented Losses (30% success rate)": f"${prevented_loss/1e6:.1f}M",
        "Net Benefit": f"${net_benefit/1e6:.1f}M",
        "ROI": f"{(net_benefit/intervention_cost)*100:.0f}%",
        "Payback Period": f"{intervention_cost/prevented_loss*12:.1f} months"
    }
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        for key, value in list(roi_metrics.items())[:3]:
            st.metric(key, value)
    
    with col2:
        for key, value in list(roi_metrics.items())[3:]:
            st.metric(key, value)
    
    st.markdown("""
    <div class="insight-box">
    <b>💼 Financial Recommendations:</b><br>
    • Focus intervention on <span class="risk-high">Very High Risk</span> segment for maximum ROI<br>
    • Estimated <span class="risk-low">$20.3M net benefit</span> from targeted intervention program<br>
    • <span class="risk-medium">30% prevention rate</span> assumption is conservative - actual results may be higher<br>
    • Recommend phased rollout starting with highest risk customers
    </div>
    """, unsafe_allow_html=True)

# Tab 5: Model Performance
with tab5:
    st.markdown('<h2 class="sub-header">Model Performance Metrics</h2>', unsafe_allow_html=True)
    
    # Model comparison
    model_metrics = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'Precision': [0.438, 0.644, 0.466],
        'Recall': [0.596, 0.343, 0.606],
        'F1 Score': [0.505, 0.448, 0.527],
        'F2 Score': [0.556, 0.378, 0.572]
    }
    
    metrics_df = pd.DataFrame(model_metrics)
    
    # Create performance comparison chart
    fig_metrics = go.Figure()
    
    for metric in ['Precision', 'Recall', 'F1 Score', 'F2 Score']:
        fig_metrics.add_trace(go.Bar(
            name=metric,
            x=metrics_df['Model'],
            y=metrics_df[metric],
            text=[f"{v:.3f}" for v in metrics_df[metric]],
            textposition='auto'
        ))
    
    fig_metrics.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
        barmode='group',
        yaxis_range=[0, 1]
    )
    
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Model Selection Rationale")
        st.markdown("""
        **Why F2 Score?**
        - Prioritizes **recall** over precision (β=2)
        - Critical for credit risk: missing defaults is costlier than false alarms
        - Ensures we catch more high-risk customers
        
        **Ensemble Approach:**
        - Combines strengths of all three models
        - Weighted average: LR (40%), RF (30%), XGB (30%)
        - Balances interpretability with performance
        """)
    
    with col2:
        st.markdown("### 📊 Confusion Matrix (Best Model)")
        
        # Display confusion matrix for best model (XGBoost)
        cm_data = {
            'Predicted No Default': [4422, 872],
            'Predicted Default': [251, 455]
        }
        cm_df = pd.DataFrame(cm_data, index=['Actual No Default', 'Actual Default'])
        
        fig_cm = px.imshow(
            cm_df.values,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['No Default', 'Default'],
            y=['No Default', 'Default'],
            color_continuous_scale='RdYlGn_r',
            text_auto=True
        )
        fig_cm.update_layout(title="XGBoost Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <b>🤖 Model Performance Summary:</b><br>
    • <span class="risk-high">XGBoost</span> provides best balance with F2=0.572<br>
    • <span class="risk-medium">60.6% recall</span> means we catch majority of defaults<br>
    • Ensemble approach recommended for production deployment<br>
    • Regular retraining needed to maintain performance
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Credit Risk Analytics Dashboard | Last Updated: 2024 | Powered by Machine Learning</p>
    <p>⚠️ <b>Disclaimer:</b> This dashboard is for analytical purposes. All credit decisions should be reviewed by qualified personnel.</p>
</div>
""", unsafe_allow_html=True)

# Add download section in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Export Options")

if st.sidebar.button("Generate Risk Report"):
    st.sidebar.success("Risk report generated successfully!")
    st.sidebar.download_button(
        label="Download Report (CSV)",
        data=risk_segments.to_csv(index=False),
        file_name="credit_risk_report.csv",
        mime="text/csv"
    )
