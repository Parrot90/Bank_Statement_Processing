import os
import pandas as pd
import numpy as np
import joblib
import shap
import logging
import base64
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sentence_transformers import SentenceTransformer
import json



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Configuration ---
MODEL_DIR = 'ml_models/trained_models_prediction'
SENTENCE_TRANSFORMER_MODEL_PATH = os.path.join(MODEL_DIR, 'sentence_transformer_model')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
XGB_MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_model.pkl')
CATEGORY_MAPPINGS_PATH = os.path.join(MODEL_DIR, 'category_mappings.pkl')

# Global variables to store loaded models and mappings
model_st = None
scaler = None
xgb_model = None
int_to_category = None
category_to_int = None # Not strictly needed for inference, but good to load if saved

def load_inference_components():
    """Loads all necessary models and mappings for inference."""
    global model_st, scaler, xgb_model, int_to_category, category_to_int

    print("--- Loading Inference Components ---")
    try:
        # Load SentenceTransformer model
        model_st = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_PATH)
        print(f"SentenceTransformer model loaded from {SENTENCE_TRANSFORMER_MODEL_PATH}")

        # Load MinMaxScaler
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print(f"MinMaxScaler loaded from {SCALER_PATH}")

        # Load XGBoost model
        with open(XGB_MODEL_PATH, 'rb') as f:
            xgb_model = pickle.load(f)
        print(f"XGBoost model loaded from {XGB_MODEL_PATH}")

        # Load category mappings
        with open(CATEGORY_MAPPINGS_PATH, 'rb') as f:
            mappings = pickle.load(f)
            category_to_int = mappings['category_to_int']
            int_to_category = mappings['int_to_category']
        print(f"Category mappings loaded from {CATEGORY_MAPPINGS_PATH}")

        print("All inference components loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Required model file not found: {e}.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        exit()

def predict_transaction_category(description: str, money_in: float, money_out: float) -> str:
    """
    Predicts the category of a single transaction.

    Args:
        description (str): The description of the transaction.
        money_in (float): The money-in amount.
        money_out (float): The money-out amount.

    Returns:
        str: The predicted category.
    """
    if not all([model_st, scaler, xgb_model, int_to_category]):
        print("Error: Inference components not loaded. Please call load_inference_components() first.")
        return "Prediction Error: Models not loaded"

    # 1. Generate embedding for the description
    new_description_embedding = model_st.encode([description], show_progress_bar=False)

    # 2. Scale numerical features
    # Ensure input to scaler.transform is 2D array (1 sample, 2 features)
    new_numerical_features_scaled = scaler.transform(np.array([[money_in, money_out]]))

    # 3. Combine all features
    new_combined_features = np.hstack((new_description_embedding, new_numerical_features_scaled))

    # 4. Get probability estimates for all classes
    probabilities = xgb_model.predict_proba(new_combined_features)[0] # Get probabilities for the single sample 
    
    # 5. Find the index of the highest probability (predicted class) 
    predicted_category_encoded = np.argmax(probabilities) 
    
    # 6. Get the confidence score for the predicted class 
    confidence_score = float(probabilities[predicted_category_encoded])
    #print(f"confidence_score: {confidence_score}  ")
    
    # 7. Decode the predicted category back to its original name 
    predicted_category = int_to_category[predicted_category_encoded] 


    return { "predicted_category": predicted_category, "confidence":round(confidence_score,2)}
