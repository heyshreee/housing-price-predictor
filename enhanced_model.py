#!/usr/bin/env python3
"""
Enhanced Housing Price Predictor
A comprehensive machine learning model with multiple algorithms,
feature engineering, and proper validation techniques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

class HousingPricePredictor:
    """Enhanced Housing Price Predictor with multiple ML algorithms."""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        self.best_model = None
        self.best_score = float('inf')
        self.feature_names = []
        
    def load_and_explore_data(self, filename='housing_data.csv'):
        """Load and explore the housing dataset."""
        print("🏠 LOADING AND EXPLORING HOUSING DATA")
        print("=" * 50)
        
        # Load data
        self.df = pd.read_csv(filename)
        print(f"Dataset shape: {self.df.shape}")
        print("\n📊 First 5 rows:")
        print(self.df.head())
        
        print("\n📈 Dataset Statistics:")
        print(self.df.describe())
        
        print("\n🔍 Data Types and Missing Values:")
        print(self.df.info())
        print(f"Missing values:\n{self.df.isnull().sum()}")
        
        return self.df
    
    def create_enhanced_features(self):
        """Create additional features for better prediction."""
        print("\n🔧 FEATURE ENGINEERING")
        print("=" * 30)
        
        # Create a copy for feature engineering
        df_enhanced = self.df.copy()
        
        # Original features
        original_features = ['Rooms', 'Area (sqft)', 'Age']
        
        # Feature engineering
        df_enhanced['Price_per_sqft'] = df_enhanced['Price (in $1000s)'] * 1000 / df_enhanced['Area (sqft)']
        df_enhanced['Rooms_per_sqft'] = df_enhanced['Rooms'] / df_enhanced['Area (sqft)'] * 1000
        df_enhanced['Age_category'] = pd.cut(df_enhanced['Age'], 
                                           bins=[0, 10, 20, 50], 
                                           labels=['New', 'Medium', 'Old'])
        df_enhanced['Size_category'] = pd.cut(df_enhanced['Area (sqft)'], 
                                            bins=[0, 1000, 1500, 3000], 
                                            labels=['Small', 'Medium', 'Large'])
        
        # One-hot encode categorical features
        df_encoded = pd.get_dummies(df_enhanced, columns=['Age_category', 'Size_category'])
        
        # Remove target from features
        feature_cols = [col for col in df_encoded.columns if col != 'Price (in $1000s)']
        self.feature_names = feature_cols
        
        print(f"Enhanced features created: {len(feature_cols)} total features")
        print(f"New features: {[col for col in feature_cols if col not in original_features]}")
        
        return df_encoded[feature_cols], df_encoded['Price (in $1000s)']
    
    def prepare_data(self):
        """Prepare data for training with proper scaling."""
        print("\n📋 DATA PREPARATION")
        print("=" * 25)
        
        # Get enhanced features
        X, y = self.create_enhanced_features()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Scale features (NOT the target variable)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
    
    def train_multiple_models(self, X_train, y_train):
        """Train multiple ML models for comparison."""
        print("\n🤖 TRAINING MULTIPLE MODELS")
        print("=" * 35)
        
        # Initialize models
        models_config = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Support Vector Regression': SVR(kernel='rbf', C=100, gamma=0.1)
        }
        
        # Train and evaluate each model
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, 
                                      cv=3, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Store model and score
            self.models[name] = {
                'model': model,
                'cv_rmse': cv_rmse,
                'cv_std': cv_scores.std()
            }
            
            print(f"  Cross-validation RMSE: {cv_rmse:.2f} (+/- {cv_scores.std():.2f})")
            
            # Track best model
            if cv_rmse < self.best_score:
                self.best_score = cv_rmse
                self.best_model = name
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models on test set."""
        print(f"\n📊 MODEL EVALUATION RESULTS")
        print("=" * 40)
        
        results = []
        
        for name, model_info in self.models.items():
            model = model_info['model']
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                'Model': name,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'CV_RMSE': model_info['cv_rmse']
            })
            
            print(f"\n{name}:")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  R²: {r2:.3f}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('RMSE')
        
        print(f"\n🏆 BEST MODEL: {self.best_model}")
        print(f"\n📋 COMPLETE RESULTS SUMMARY:")
        print(results_df.to_string(index=False, float_format='%.3f'))
        
        return results_df
    
    def feature_importance_analysis(self):
        """Analyze feature importance for tree-based models."""
        print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 40)
        
        # Get Random Forest model
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']['model']
            
            # Feature importance
            importance = rf_model.feature_importances_
            feature_imp_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            print(feature_imp_df.to_string(index=False, float_format='%.4f'))
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_imp_df.head(10), x='Importance', y='Feature')
            plt.title('Top 10 Feature Importance (Random Forest)')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return feature_imp_df
    
    def make_predictions_examples(self, X_test_original, y_test):
        """Make example predictions and show results."""
        print(f"\n🎯 EXAMPLE PREDICTIONS")
        print("=" * 30)
        
        best_model_obj = self.models[self.best_model]['model']
        
        # Scale the test data for prediction
        X_test_scaled = self.scaler.transform(X_test_original)
        predictions = best_model_obj.predict(X_test_scaled)
        
        # Create examples DataFrame
        examples = pd.DataFrame({
            'Rooms': X_test_original['Rooms'].values,
            'Area (sqft)': X_test_original['Area (sqft)'].values,
            'Age': X_test_original['Age'].values,
            'Actual Price ($1000s)': y_test.values,
            'Predicted Price ($1000s)': predictions,
            'Difference ($1000s)': predictions - y_test.values,
            'Error %': ((predictions - y_test.values) / y_test.values * 100)
        })
        
        print("Sample Predictions:")
        print(examples.to_string(index=False, float_format='%.1f'))
        
        return examples
    
    def predict_new_house(self, rooms, area, age):
        """Predict price for a new house."""
        print(f"\n🏡 NEW HOUSE PREDICTION")
        print("=" * 30)
        print(f"House specifications:")
        print(f"  Rooms: {rooms}")
        print(f"  Area: {area} sqft")
        print(f"  Age: {age} years")
        
        # Create feature vector matching training format
        # We need to reconstruct all the engineered features
        sample_df = pd.DataFrame({
            'Rooms': [rooms],
            'Area (sqft)': [area],
            'Age': [age],
            'Price (in $1000s)': [0]  # Dummy value
        })
        
        # Apply same feature engineering
        sample_df['Price_per_sqft'] = 0  # Will be ignored
        sample_df['Rooms_per_sqft'] = sample_df['Rooms'] / sample_df['Area (sqft)'] * 1000
        sample_df['Age_category'] = pd.cut(sample_df['Age'], 
                                         bins=[0, 10, 20, 50], 
                                         labels=['New', 'Medium', 'Old'])
        sample_df['Size_category'] = pd.cut(sample_df['Area (sqft)'], 
                                          bins=[0, 1000, 1500, 3000], 
                                          labels=['Small', 'Medium', 'Large'])
        
        # One-hot encode
        sample_encoded = pd.get_dummies(sample_df, columns=['Age_category', 'Size_category'])
        
        # Ensure all columns match training data
        for col in self.feature_names:
            if col not in sample_encoded.columns:
                sample_encoded[col] = 0
        
        # Select and order features
        X_new = sample_encoded[self.feature_names]
        
        # Scale features
        X_new_scaled = self.scaler.transform(X_new)
        
        # Predict with best model
        best_model_obj = self.models[self.best_model]['model']
        predicted_price = best_model_obj.predict(X_new_scaled)[0]
        
        print(f"\n💰 Predicted Price: ${predicted_price:.1f}k (${predicted_price*1000:,.0f})")
        
        # Get predictions from all models
        print(f"\nPredictions from all models:")
        for name, model_info in self.models.items():
            model = model_info['model']
            pred = model.predict(X_new_scaled)[0]
            print(f"  {name}: ${pred:.1f}k")
        
        return predicted_price
    
    def save_best_model(self):
        """Save the best performing model."""
        best_model_obj = self.models[self.best_model]['model']
        model_filename = f'best_model_{self.best_model.lower().replace(" ", "_")}.pkl'
        
        # Save model and scaler
        joblib.dump({
            'model': best_model_obj,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_name': self.best_model
        }, model_filename)
        
        print(f"\n💾 Best model saved as: {model_filename}")
        return model_filename


def main():
    """Main execution function with comprehensive examples."""
    print("🏠 ENHANCED HOUSING PRICE PREDICTOR")
    print("=" * 50)
    print("Advanced ML pipeline with multiple algorithms and feature engineering")
    print()
    
    # Initialize predictor
    predictor = HousingPricePredictor()
    
    # Load and explore data
    predictor.load_and_explore_data()
    
    # Prepare data
    X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = predictor.prepare_data()
    
    # Train multiple models
    predictor.train_multiple_models(X_train, y_train)
    
    # Evaluate models
    results = predictor.evaluate_models(X_test, y_test)
    
    # Feature importance analysis
    predictor.feature_importance_analysis()
    
    # Example predictions
    examples = predictor.make_predictions_examples(X_test_orig, y_test)
    
    # Predict new houses
    print("\n" + "="*60)
    print("🏡 PREDICTING NEW HOUSE PRICES")
    print("="*60)
    
    # Example 1: Luxury house
    predictor.predict_new_house(rooms=6, area=2800, age=5)
    
    print("\n" + "-"*40)
    
    # Example 2: Modest house
    predictor.predict_new_house(rooms=3, area=1200, age=25)
    
    print("\n" + "-"*40)
    
    # Example 3: Small apartment
    predictor.predict_new_house(rooms=2, area=900, age=15)
    
    # Save best model
    predictor.save_best_model()
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"Best performing model: {predictor.best_model}")
    print(f"Model performance: {predictor.best_score:.2f} RMSE")


if __name__ == "__main__":
    main()