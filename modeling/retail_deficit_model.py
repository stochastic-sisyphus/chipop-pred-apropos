#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RetailDeficitModel:
    """
    Model for predicting retail deficits in Chicago ZIP codes
    based on housing development and demographic factors.
    """
    
    def __init__(self, data_dir='data', output_dir='modeling'):
        """Initialize model with data paths"""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Input files
        self.permits_file = self.data_dir / 'building_permits.csv'
        self.licenses_file = self.data_dir / 'business_licenses.csv'
        self.merged_file = Path('../output/merged_dataset.csv')
        
        # Output files
        self.model_file = self.output_dir / 'retail_deficit_model.pkl'
        self.predictions_file = self.output_dir / 'retail_deficit_predictions.csv'
        
        # Data frames
        self.merged_data = None
        self.features_df = None
        self.target = None
        self.model = None
        self.importance_df = None
        
        # Load and prepare data
        self.load_data()
        self.prepare_features()
    
    def load_data(self):
        """Load all required datasets"""
        try:
            # Try to load merged data if available
            if self.merged_file.exists():
                logger.info(f"Loading merged dataset from {self.merged_file}")
                self.merged_data = pd.read_csv(self.merged_file)
            else:
                # Load permits data
                if self.permits_file.exists():
                    logger.info(f"Loading building permits from {self.permits_file}")
                    permits_df = pd.read_csv(self.permits_file)
                    
                    # Convert columns if needed
                    if 'issue_date' in permits_df.columns:
                        permits_df['year'] = pd.to_datetime(permits_df['issue_date']).dt.year
                    
                    # Handle zip_code differently in building permits
                    if 'contact_1_zipcode' in permits_df.columns:
                        permits_df['zip_code'] = permits_df['contact_1_zipcode'].astype(str).str[:5]
                        permits_df['zip_code'] = permits_df['zip_code'].apply(
                            lambda x: x if x.isdigit() and len(x) == 5 else ''
                        )
                        permits_df = permits_df[permits_df['zip_code'] != '']
                
                # Load licenses data
                if self.licenses_file.exists():
                    logger.info(f"Loading business licenses from {self.licenses_file}")
                    licenses_df = pd.read_csv(self.licenses_file)
                    
                    # Convert columns if needed
                    if 'license_start_date' in licenses_df.columns:
                        licenses_df['year'] = pd.to_datetime(licenses_df['license_start_date']).dt.year
                    
                    # Clean up zip codes
                    if 'zip_code' in licenses_df.columns:
                        licenses_df['zip_code'] = licenses_df['zip_code'].astype(str).str[:5]
                        licenses_df['zip_code'] = licenses_df['zip_code'].apply(
                            lambda x: x if x.isdigit() and len(x) == 5 else ''
                        )
                        licenses_df = licenses_df[licenses_df['zip_code'] != '']
                    
                # Create merged dataset by processing the raw data
                logger.info("Creating merged dataset from raw data files")
                self.process_raw_data(permits_df, licenses_df)
                
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
    
    def process_raw_data(self, permits_df, licenses_df):
        """Process raw data files to create a merged dataset for modeling"""
        try:
            # Filter housing permits
            housing_permits = permits_df[
                permits_df['permit_type'].str.contains('PERMIT - NEW CONSTRUCTION', case=False, na=False) &
                permits_df['work_description'].str.contains('RESIDENTIAL|APARTMENT|CONDO|MULTI', case=False, na=False)
            ]
            
            # Filter multi-family permits
            multi_family_permits = housing_permits[
                housing_permits['work_description'].str.contains('APARTMENT|CONDO|MULTI|UNIT', case=False, na=False) &
                ~housing_permits['work_description'].str.contains('SINGLE FAMILY|ONE FAMILY', case=False, na=False)
            ]
            
            # Filter retail licenses
            retail_licenses = licenses_df[
                licenses_df['license_description'].str.contains(
                    'RETAIL|FOOD|GROCERY|RESTAURANT|SHOP|STORE', 
                    case=False, na=False
                )
            ]
            
            # Group by ZIP code and year for housing permits
            housing_by_zip_year = housing_permits.groupby(['zip_code', 'year']).size().reset_index(name='housing_permits')
            
            # Group by ZIP code and year for multi-family permits
            multi_family_by_zip_year = multi_family_permits.groupby(['zip_code', 'year']).size().reset_index(name='multi_family_permits')
            
            # Group by ZIP code and year for retail licenses
            retail_by_zip_year = retail_licenses.groupby(['zip_code', 'year']).size().reset_index(name='retail_licenses')
            
            # Merge the dataframes
            merged = housing_by_zip_year.merge(
                multi_family_by_zip_year, on=['zip_code', 'year'], how='left'
            ).merge(
                retail_by_zip_year, on=['zip_code', 'year'], how='left'
            )
            
            # Fill NaN values
            merged = merged.fillna(0)
            
            # Calculate housing-to-retail ratio
            merged['housing_retail_ratio'] = np.where(
                merged['retail_licenses'] > 0,
                merged['housing_permits'] / merged['retail_licenses'],
                merged['housing_permits']  # If no retail, use housing permits as the ratio
            )
            
            # Calculate multi-family-to-retail ratio
            merged['multi_family_retail_ratio'] = np.where(
                merged['retail_licenses'] > 0,
                merged['multi_family_permits'] / merged['retail_licenses'],
                merged['multi_family_permits']  # If no retail, use multi-family permits as the ratio
            )
            
            # Set the merged data
            self.merged_data = merged
            
        except Exception as e:
            logger.error(f"Error processing raw data: {str(e)}")
    
    def prepare_features(self):
        """Prepare features for modeling"""
        if self.merged_data is None:
            logger.error("No data available for feature preparation")
            return
        
        try:
            # Calculate aggregated features for each ZIP code
            zip_features = self.merged_data.groupby('zip_code').agg({
                'housing_permits': ['mean', 'sum', 'std'],
                'multi_family_permits': ['mean', 'sum', 'std'],
                'retail_licenses': ['mean', 'sum', 'std'],
                'housing_retail_ratio': ['mean', 'max'],
                'multi_family_retail_ratio': ['mean', 'max']
            })
            
            # Flatten the column names
            zip_features.columns = ['_'.join(col).strip() for col in zip_features.columns.values]
            
            # Reset index to get zip_code as a column
            zip_features = zip_features.reset_index()
            
            # Handle missing values (for std columns)
            zip_features = zip_features.fillna(0)
            
            # Calculate additional features
            # Growth in housing permits (comparing first half vs second half)
            first_half = self.merged_data[self.merged_data['year'] <= 2020]
            second_half = self.merged_data[self.merged_data['year'] > 2020]
            
            first_half_housing = first_half.groupby('zip_code')['housing_permits'].sum().reset_index()
            first_half_housing.columns = ['zip_code', 'first_half_housing']
            
            second_half_housing = second_half.groupby('zip_code')['housing_permits'].sum().reset_index()
            second_half_housing.columns = ['zip_code', 'second_half_housing']
            
            # Merge with zip_features
            zip_features = zip_features.merge(first_half_housing, on='zip_code', how='left')
            zip_features = zip_features.merge(second_half_housing, on='zip_code', how='left')
            
            # Fill NaN values
            zip_features = zip_features.fillna(0)
            
            # Calculate growth percentages
            zip_features['housing_growth_pct'] = np.where(
                zip_features['first_half_housing'] > 0,
                (zip_features['second_half_housing'] - zip_features['first_half_housing']) / zip_features['first_half_housing'] * 100,
                0
            )
            
            # Set the target variable: retail deficit score
            # Higher value means more retail deficit relative to housing
            zip_features['retail_deficit_score'] = zip_features['housing_permits_sum'] / (zip_features['retail_licenses_sum'] + 1)
            
            # Store features and target
            self.target = zip_features['retail_deficit_score']
            self.features_df = zip_features.drop(['retail_deficit_score'], axis=1)
            
            logger.info(f"Prepared features for {len(self.features_df)} ZIP codes")
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
    
    def train_model(self):
        """Train a random forest model to predict retail deficits"""
        if self.features_df is None or self.target is None:
            logger.error("Features or target not available for model training")
            return
        
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                self.features_df, self.target, test_size=0.25, random_state=42
            )
            
            # Scale the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.drop('zip_code', axis=1))
            X_test_scaled = scaler.transform(X_test.drop('zip_code', axis=1))
            
            # Train the model
            logger.info("Training Random Forest Regressor model")
            self.model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate the model
            y_pred = self.model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"Model evaluation - MSE: {mse:.4f}, R²: {r2:.4f}")
            
            # Get feature importances
            feature_names = X_train.columns.tolist()
            feature_names.remove('zip_code')
            
            importances = self.model.feature_importances_
            self.importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Save feature importances
            importance_file = self.output_dir / 'retail_deficit_feature_importance.csv'
            self.importance_df.to_csv(importance_file, index=False)
            logger.info(f"Feature importances saved to {importance_file}")
            
            # Save the model
            joblib.dump(self.model, self.model_file)
            logger.info(f"Model saved to {self.model_file}")
            
            # Visualize feature importances
            self._visualize_feature_importance()
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return None
    
    def predict_retail_deficits(self):
        """Make predictions for all ZIP codes"""
        if self.model is None:
            logger.warning("No trained model available. Training now...")
            self.train_model()
            
        if self.model is None:
            logger.error("Failed to train a model for predictions")
            return
        
        try:
            # Scale the features (excluding zip_code)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(self.features_df.drop('zip_code', axis=1))
            
            # Make predictions
            predictions = self.model.predict(features_scaled)
            
            # Create a dataframe with ZIP codes and predictions
            results = pd.DataFrame({
                'zip_code': self.features_df['zip_code'],
                'retail_deficit_score': predictions,
                'housing_permits_sum': self.features_df['housing_permits_sum'],
                'retail_licenses_sum': self.features_df['retail_licenses_sum'],
                'housing_retail_ratio_mean': self.features_df['housing_retail_ratio_mean'],
                'housing_growth_pct': self.features_df['housing_growth_pct']
            })
            
            # Add a deficit level category
            results['deficit_level'] = pd.qcut(
                results['retail_deficit_score'], 
                q=5, 
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            )
            
            # Sort by retail deficit score
            results = results.sort_values('retail_deficit_score', ascending=False)
            
            # Save the results
            results.to_csv(self.predictions_file, index=False)
            logger.info(f"Predictions saved to {self.predictions_file}")
            
            # Visualize the results
            self._visualize_predictions(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return None
    
    def _visualize_feature_importance(self):
        """Visualize the importance of each feature in the model"""
        if self.importance_df is None:
            logger.warning("No feature importance data available for visualization")
            return
        
        try:
            plt.figure(figsize=(12, 8))
            
            # Get top 10 features
            top_features = self.importance_df.head(10)
            
            # Create barplot
            sns.barplot(
                x='importance', 
                y='feature',
                data=top_features,
                palette='viridis'
            )
            
            plt.title('Top 10 Features for Predicting Retail Deficits', fontsize=16)
            plt.xlabel('Importance', fontsize=14)
            plt.ylabel('Feature', fontsize=14)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(self.output_dir / 'retail_deficit_feature_importance.png', dpi=300)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error visualizing feature importance: {str(e)}")
    
    def _visualize_predictions(self, results):
        """Visualize the retail deficit predictions"""
        if results is None or results.empty:
            logger.warning("No prediction results available for visualization")
            return
        
        try:
            # Top 15 ZIP codes with highest retail deficit
            plt.figure(figsize=(14, 8))
            
            top_deficit = results.head(15)
            
            ax = sns.barplot(
                x='zip_code',
                y='retail_deficit_score',
                hue='deficit_level',
                data=top_deficit,
                palette='rocket'
            )
            
            plt.title('Top 15 ZIP Codes with Highest Retail Deficits', fontsize=16)
            plt.xlabel('ZIP Code', fontsize=14)
            plt.ylabel('Retail Deficit Score', fontsize=14)
            plt.xticks(rotation=45)
            
            # Add data labels
            for i, row in enumerate(top_deficit.itertuples()):
                ax.text(
                    i, row.retail_deficit_score + 0.1, 
                    f"{row.retail_deficit_score:.1f}", 
                    ha='center', va='bottom',
                    fontweight='bold'
                )
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'top_retail_deficit_areas.png', dpi=300)
            plt.close()
            
            # Housing permits vs. retail licenses scatter plot
            plt.figure(figsize=(14, 8))
            
            ax = sns.scatterplot(
                x='housing_permits_sum',
                y='retail_licenses_sum',
                hue='deficit_level',
                size='housing_growth_pct',
                sizes=(20, 500),
                palette='rocket',
                data=results
            )
            
            # Add ZIP code labels for top deficit areas
            for i, row in enumerate(top_deficit.itertuples()):
                ax.text(
                    row.housing_permits_sum + 2, 
                    row.retail_licenses_sum + 2, 
                    row.zip_code, 
                    fontsize=9
                )
            
            plt.title('Housing Permits vs. Retail Licenses by ZIP Code', fontsize=16)
            plt.xlabel('Total Housing Permits', fontsize=14)
            plt.ylabel('Total Retail Licenses', fontsize=14)
            
            # Add reference line for a "balanced" ratio
            max_val = max(
                results['housing_permits_sum'].max(),
                results['retail_licenses_sum'].max()
            )
            plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='1:1 Ratio')
            plt.legend(title='Deficit Level')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'housing_vs_retail_scatter.png', dpi=300)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error visualizing predictions: {str(e)}")

def main():
    """Main function to train model and make predictions"""
    model = RetailDeficitModel()
    model.train_model()
    model.predict_retail_deficits()
    
    logger.info("Retail deficit modeling completed")

if __name__ == "__main__":
    main() 