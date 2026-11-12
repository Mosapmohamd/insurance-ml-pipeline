#!/usr/bin/env python
import pandas as pd
import numpy as np
import joblib
import os
import argparse
from sklearn.pipeline import Pipeline
import sys

def load_model(model_path='classification_model.pkl'):
    """
    Load the trained classification model
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded model
    """
    try:
        print(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def prepare_input_data(input_file, required_columns=None):
    """
    Load and prepare input data for prediction
    
    Args:
        input_file: Path to the input CSV file
        required_columns: List of columns that should be present in the data
        
    Returns:
        Prepared dataframe ready for prediction
    """
    try:
        print(f"Reading input data from: {input_file}")
        # Determine file type and read accordingly
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(input_file)
        else:
            raise ValueError("Unsupported file format. Please provide CSV or Excel file.")
        
        print(f"Input data shape: {df.shape}")
        
        # Verify required columns if specified
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    except Exception as e:
        print(f"Error preparing input data: {e}")
        sys.exit(1)

def make_predictions(model, data):
    """
    Use the model to make predictions on input data
    
    Args:
        model: Trained classification model
        data: Input dataframe
        
    Returns:
        DataFrame with original data and predictions
    """
    try:
        print("Making predictions...")
        
        # Get predictions
        y_pred = model.predict(data)
        
        # Get prediction probabilities if available
        try:
            y_pred_proba = model.predict_proba(data)
            # Create a dataframe with the probability for each class
            class_names = model.classes_
            proba_columns = [f"probability_class_{cls}" for cls in class_names]
            proba_df = pd.DataFrame(y_pred_proba, columns=proba_columns)
            
            # Add a column for the highest probability
            result_df = pd.DataFrame({
                'predicted_class': y_pred,
                'confidence': np.max(y_pred_proba, axis=1)
            })
            
            # Combine with probability columns
            result_df = pd.concat([result_df, proba_df], axis=1)
            
        except (AttributeError, NotImplementedError):
            # Model doesn't support predict_proba
            result_df = pd.DataFrame({'predicted_class': y_pred})
        
        print(f"Made {len(y_pred)} predictions")
        return result_df
    except Exception as e:
        print(f"Error making predictions: {e}")
        sys.exit(1)

def save_predictions(predictions, original_data=None, output_file='predictions.csv'):
    """
    Save predictions to file, optionally combining with original data
    
    Args:
        predictions: DataFrame with predictions
        original_data: Original input data (optional)
        output_file: Path to save the predictions
    """
    try:
        if original_data is not None:
            # Combine original data with predictions
            result_df = pd.concat([original_data.reset_index(drop=True), 
                                  predictions.reset_index(drop=True)], axis=1)
        else:
            result_df = predictions
            
        # Save to file
        result_df.to_csv(output_file, index=False)
        print(f"Predictions saved to: {output_file}")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        sys.exit(1)

def main():
    """
    Main function to run the inference pipeline
    """
    parser = argparse.ArgumentParser(description='Make predictions using a trained classification model')
    parser.add_argument('--input', '-i', required=True, help='Path to input data file (CSV or Excel)')
    parser.add_argument('--model', '-m', default='classification_model.pkl', 
                        help='Path to trained model file (default: classification_model.pkl)')
    parser.add_argument('--output', '-o', default='predictions.csv', 
                        help='Path to save predictions (default: predictions.csv)')
    parser.add_argument('--include-original', '-inc', action='store_true', 
                        help='Include original data in the output file')
    
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.model)
    
    # Get model feature names (if available)
    required_features = None
    try:
        # Attempt to get feature names from the model
        if hasattr(model, 'feature_names_in_'):
            required_features = model.feature_names_in_.tolist()
        # For pipeline models
        elif isinstance(model, Pipeline) and hasattr(model[-1], 'feature_names_in_'):
            required_features = model[-1].feature_names_in_.tolist()
    except (AttributeError, IndexError):
        print("Could not automatically determine required features from model")
    
    # Prepare input data
    input_data = prepare_input_data(args.input, required_features)
    
    # Make predictions
    predictions = make_predictions(model, input_data)
    
    # Save results
    if args.include_original:
        save_predictions(predictions, input_data, args.output)
    else:
        save_predictions(predictions, output_file=args.output)
    
    print("Inference completed successfully!")

if __name__ == "__main__":
    main()