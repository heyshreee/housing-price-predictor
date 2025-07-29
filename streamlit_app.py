import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from enhanced_model import HousingPricePredictor
import os

# Page configuration
st.set_page_config(
    page_title="🏠 Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .prediction-result {
        font-size: 2rem;
        color: #2e8b57;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background-color: #f0fff0;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Load the sample housing data."""
    try:
        return pd.read_csv('housing_data.csv')
    except FileNotFoundError:
        # Create sample data if file doesn't exist
        sample_data = {
            'Rooms': [3, 4, 2, 5, 3, 4, 6, 2, 5, 3],
            'Area (sqft)': [1200, 1500, 800, 2000, 1000, 1400, 2500, 750, 1800, 1100],
            'Age': [20, 15, 30, 10, 25, 18, 5, 35, 12, 28],
            'Price (in $1000s)': [200, 250, 150, 300, 180, 230, 400, 120, 280, 170]
        }
        df = pd.DataFrame(sample_data)
        df.to_csv('housing_data.csv', index=False)
        return df

@st.cache_resource
def initialize_predictor():
    """Initialize and train the predictor model."""
    predictor = HousingPricePredictor()
    
    # Load data
    predictor.load_and_explore_data()
    
    # Prepare data
    X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = predictor.prepare_data()
    
    # Train models (simplified for web app)
    predictor.train_multiple_models(X_train, y_train)
    
    return predictor, X_test_orig, y_test

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 Housing Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced ML-powered real estate price estimation")
    
    # Initialize predictor
    with st.spinner("🔄 Initializing AI models..."):
        predictor, X_test_orig, y_test = initialize_predictor()
    
    # Sidebar for navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox("Choose a page:", [
        "🎯 Price Prediction", 
        "📊 Data Analysis", 
        "🤖 Model Performance",
        "📈 Market Insights"
    ])
    
    if page == "🎯 Price Prediction":
        prediction_page(predictor)
    elif page == "📊 Data Analysis":
        data_analysis_page()
    elif page == "🤖 Model Performance":
        model_performance_page(predictor, X_test_orig, y_test)
    elif page == "📈 Market Insights":
        market_insights_page()

def prediction_page(predictor):
    st.header("🎯 Predict House Price")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🏡 House Specifications")
        
        # Input fields
        rooms = st.slider("Number of Rooms", min_value=1, max_value=10, value=3, step=1)
        area = st.slider("Area (Square Feet)", min_value=500, max_value=5000, value=1500, step=50)
        age = st.slider("Age of House (Years)", min_value=0, max_value=50, value=15, step=1)
        
        # Predict button
        if st.button("🔮 Predict Price", type="primary"):
            with st.spinner("🧠 AI is analyzing..."):
                try:
                    predicted_price = predictor.predict_new_house(rooms, area, age)
                    
                    # Display result
                    st.markdown(f"""
                    <div class="prediction-result">
                        💰 Predicted Price: ${predicted_price:.1f}k
                        <br>
                        <small>(${predicted_price*1000:,.0f})</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Price per square foot
                    price_per_sqft = (predicted_price * 1000) / area
                    st.metric("Price per Square Foot", f"${price_per_sqft:.2f}")
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
    
    with col2:
        st.subheader("📋 House Summary")
        
        # Create a summary card
        summary_data = {
            "🏠 Rooms": rooms,
            "📐 Area": f"{area:,} sqft",
            "📅 Age": f"{age} years",
            "🏷️ Category": get_house_category(rooms, area, age)
        }
        
        for key, value in summary_data.items():
            st.info(f"**{key}**: {value}")
        
        # Comparison with similar houses
        st.subheader("🔍 Similar Houses")
        df = load_sample_data()
        
        # Find similar houses
        similar_houses = df[
            (abs(df['Rooms'] - rooms) <= 1) & 
            (abs(df['Area (sqft)'] - area) <= 300)
        ]
        
        if not similar_houses.empty:
            st.dataframe(similar_houses, use_container_width=True)
        else:
            st.info("No similar houses found in database")

def data_analysis_page():
    st.header("📊 Housing Market Data Analysis")
    
    # Load data
    df = load_sample_data()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Houses", len(df))
    with col2:
        st.metric("Avg Price", f"${df['Price (in $1000s)'].mean():.0f}k")
    with col3:
        st.metric("Avg Area", f"{df['Area (sqft)'].mean():.0f} sqft")
    with col4:
        st.metric("Avg Age", f"{df['Age'].mean():.0f} years")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Price vs Area scatter plot
        fig = px.scatter(df, x='Area (sqft)', y='Price (in $1000s)', 
                        size='Rooms', color='Age',
                        title="Price vs Area (sized by Rooms, colored by Age)",
                        hover_data=['Rooms'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Price distribution
        fig = px.histogram(df, x='Price (in $1000s)', 
                          title="Price Distribution",
                          nbins=10)
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔗 Feature Correlations")
    corr_matrix = df.corr()
    fig = px.imshow(corr_matrix, 
                    text_auto=True, 
                    aspect="auto",
                    title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)
    
    # Raw data
    st.subheader("📋 Raw Data")
    st.dataframe(df, use_container_width=True)

def model_performance_page(predictor, X_test_orig, y_test):
    st.header("🤖 Model Performance Analysis")
    
    # Model comparison
    st.subheader("🏆 Model Comparison")
    
    # Create performance metrics
    results = []
    for name, model_info in predictor.models.items():
        model = model_info['model']
        
        # Make predictions
        X_test_scaled = predictor.scaler.transform(X_test_orig)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        })
    
    results_df = pd.DataFrame(results).sort_values('RMSE')
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🥇 Best Model", results_df.iloc[0]['Model'])
    with col2:
        st.metric("📉 Best RMSE", f"{results_df.iloc[0]['RMSE']:.2f}")
    with col3:
        st.metric("📈 Best R²", f"{results_df.iloc[0]['R²']:.3f}")
    
    # Performance chart
    fig = px.bar(results_df, x='Model', y='RMSE', 
                 title="Model Performance Comparison (Lower RMSE is Better)",
                 color='RMSE', color_continuous_scale='RdYlBu_r')
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results table
    st.subheader("📊 Detailed Results")
    st.dataframe(results_df, use_container_width=True)
    
    # Prediction vs Actual plot
    best_model = predictor.models[predictor.best_model]['model']
    X_test_scaled = predictor.scaler.transform(X_test_orig)
    y_pred_best = best_model.predict(X_test_scaled)
    
    fig = px.scatter(x=y_test, y=y_pred_best,
                     title=f"Actual vs Predicted Prices ({predictor.best_model})",
                     labels={'x': 'Actual Price ($1000s)', 'y': 'Predicted Price ($1000s)'})
    # Add perfect prediction line
    min_val, max_val = min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())
    fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                  line=dict(color="red", width=2, dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

def market_insights_page():
    st.header("📈 Market Insights & Trends")
    
    df = load_sample_data()
    
    # Price insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💡 Key Insights")
        
        # Calculate insights
        price_per_sqft = (df['Price (in $1000s)'] * 1000) / df['Area (sqft)']
        avg_price_per_sqft = price_per_sqft.mean()
        
        insights = [
            f"🏠 Average price per sq ft: ${avg_price_per_sqft:.2f}",
            f"📊 Most common room count: {df['Rooms'].mode().iloc[0]} rooms",
            f"🏗️ Newest house: {df['Age'].min()} years old",
            f"🏚️ Oldest house: {df['Age'].max()} years old",
            f"💰 Price range: ${df['Price (in $1000s)'].min():.0f}k - ${df['Price (in $1000s)'].max():.0f}k"
        ]
        
        for insight in insights:
            st.info(insight)
    
    with col2:
        st.subheader("📊 Market Segments")
        
        # Create price segments
        df_copy = df.copy()
        df_copy['Price_Segment'] = pd.cut(df_copy['Price (in $1000s)'], 
                                         bins=[0, 200, 300, float('inf')], 
                                         labels=['Budget', 'Mid-range', 'Luxury'])
        
        segment_counts = df_copy['Price_Segment'].value_counts()
        fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                     title="Market Segmentation")
        st.plotly_chart(fig, use_container_width=True)
    
    # Investment recommendations
    st.subheader("💼 Investment Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("""
        **🟢 Buy Recommendation**
        - Houses > 2000 sqft
        - Age < 15 years
        - Good price/sqft ratio
        """)
    
    with col2:
        st.warning("""
        **🟡 Hold Recommendation**
        - Medium size (1000-2000 sqft)
        - Age 15-25 years
        - Average market price
        """)
    
    with col3:
        st.error("""
        **🔴 Avoid Recommendation**
        - Age > 30 years
        - Very small area < 800 sqft
        - High price/sqft ratio
        """)

def get_house_category(rooms, area, age):
    """Categorize house based on specifications."""
    if rooms >= 5 and area >= 2000 and age <= 10:
        return "🏰 Luxury"
    elif rooms >= 4 and area >= 1500:
        return "🏠 Premium"
    elif rooms >= 3 and area >= 1000:
        return "🏡 Standard"
    else:
        return "🏢 Compact"

if __name__ == "__main__":
    main()