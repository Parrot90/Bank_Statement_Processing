import joblib
import numpy as np
import pandas as pd
import spacy
import re
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from typing import Optional
import os


import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


class CategoryPredictionService:
    def __init__(self):
        # Set up paths
        CURRENT_DIR = Path(__file__).resolve().parent
        APP_DIR = CURRENT_DIR.parent

        MODEL_DIR = APP_DIR / "ml_models"
        DATA_DIR = APP_DIR / "data"


        # Load saved components
        model_path = MODEL_DIR / "knn_recommendation_model.pkl"
        scaler_path = MODEL_DIR / "recommendation_scaler.pkl"
        #encoder_path = MODEL_DIR / "supplier_encoder3.pkl"
        embedder_path = os.path.join(MODEL_DIR , "sentence_model_recommendation")
        #embedder_path = MODEL_DIR / "sentence_model_recommendation"
        csv_path = DATA_DIR / "final_categorized_data (1).csv"
        
        



        # Load model and scaler
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        self.embedder = SentenceTransformer(embedder_path)

        # Load spaCy NER model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
    # Download the model if not present
                from spacy.cli import download
                download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")



        # Load supplier frequency from CSV
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                df['Date'] = pd.to_datetime(df['Date'])
                df['amount'] = df[['Money_In', 'Money_Out']].max(axis=1)
                df['is_money_out'] = df['Money_Out'].apply(lambda x: 1 if x > 0 else 0)

                # Compute supplier frequency
                supplier_counts = df['Supplier'].value_counts()
                self.supplier_frequencies = supplier_counts.to_dict()

                # Fit encoder from CSV if not already loaded
                if self.supplier_encoder is None:
                    self.supplier_encoder = LabelEncoder()
                    self.supplier_encoder.fit(df['Supplier'].unique())
            except Exception as e:
                print(f"WARNING: Failed to load/process CSV: {e}")

        # if self.supplier_encoder is None:
        #     raise RuntimeError("No valid LabelEncoder available from .pkl or CSV.")

    # --------------- SUPPLIER EXTRACTORS ------------------

    def extract_supplier_ner(self, text: str) -> Optional[str]:
        """Extract supplier using SpaCy NER"""
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == "ORG":
                print(f"[NER] Extracted: {ent.text.strip()} (Label: {ent.label_})")
                return ent.text.strip()
        return None

    def extract_supplier_regex(self, text: str) -> Optional[str]:
        """Fallback regex-based extraction"""
        match = re.search(r'([A-Z][A-Z\s\*&\.]+(?:LTD|LLP|COM|EXPRESS|PAYPAL|RETAIL|LIMITED))', text.upper())
        if match:
            print(f"[Regex] Extracted: {match.group(1).strip()}")
            return match.group(1).strip()
        return None

    def extract_supplier(self, text: str) -> str:
        """Main supplier extractor with fallback"""
        supplier = self.extract_supplier_ner(text)
        if not supplier:
            print(f"[NER] No entity found. Fallback to regex.")
            supplier = self.extract_supplier_regex(text)
        return supplier or "UNKNOWN"

    # --------------- PREDICTION ------------------

    def preprocess(self, date_str: str, amount: float, description: str):
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        day, month, year = dt.day, dt.month, dt.year

        # Extract supplier from description
        supplier = self.extract_supplier(description)

        # Encode supplier
        try:
            supplier_encoded = self.supplier_encoder.transform([supplier])[0]
        except ValueError:
            print(f"[Encode] Unknown supplier: {supplier}, using 0")
            supplier_encoded = 0

        # Frequency and flags
        supplier_frequency = self.supplier_frequencies.get(supplier, 1)
        is_money_out = 1 if amount > 0 else 0

        raw_features = np.array([[
            day, month, year,
            supplier_encoded,
            supplier_frequency,
            amount,
            is_money_out
        ]])

        return self.scaler.transform(raw_features)

    def recommend_category(self,  description: str , money_in: float, money_out: float) -> str:
    # print("recommmendation service")
        try:
            # Preprocess
            description = description or ""
            is_money_out = int(money_out > 0)

            # Feature extraction
            desc_vector = self.embedder.encode([description])[0]
            numeric_features = np.array([[money_in, money_out, is_money_out]])
            combined = np.hstack([desc_vector, numeric_features[0].flatten().tolist()])
            combined_scaled = self.scaler.transform([combined])

            # Predict
            predicted_category = self.model.predict(combined_scaled)[0]
            #print(f"predicted_category: {predicted_category}")
            return predicted_category

        except Exception as e:
            return f"Prediction failed: {str(e)}"
        
        


