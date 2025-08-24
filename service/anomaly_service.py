import pandas as pd
import os
import csv
import re
import spacy
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

COLUMN_MAPPINGS = {
    "date": ["transaction date", "date", "posted date", "date of transaction", "txn date"],
    "narration": ["narration", "details", "description", "transaction details", "remarks", "particulars", "transaction"],
    "money_in": ["deposit", "money in", "in", "credit", "paid in", "DepositAmt."],
    "money_out": ["withdrawal", "money out", "out", "debit", "paid out", "WithdrawalAmt."]
}

def handle_duplicate_columns(df):
    if df.columns.duplicated().any():
        print("Warning: Duplicate column names found:", df.columns[df.columns.duplicated()].tolist())
        new_df = pd.DataFrame()
        seen_columns = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in seen_columns:
                if col_lower in [name.lower() for names in COLUMN_MAPPINGS.values() for name in names]:
                    if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[seen_columns[col_lower]]):
                        new_df[col_lower] = df[[col, seen_columns[col_lower]]].sum(axis=1)
                    else:
                        new_df[col_lower] = df[[col, seen_columns[col_lower]]].astype(str).agg(' '.join, axis=1)
                else:
                    suffix = 1
                    new_col = f"{col}_{suffix}"
                    while new_col in new_df.columns:
                        suffix += 1
                        new_col = f"{col}_{suffix}"
                    new_df[new_col] = df[col]
            else:
                new_df[col_lower] = df[col]
                seen_columns[col_lower] = col
        return new_df
    return df

def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise SystemExit("SpaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")

def clean_wrapped_lines(input_path: str, output_path: str) -> str:
    cleaned_rows = []
    with open(input_path, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        for row in reader:
            cleaned_row = [cell.replace('\n', ' ') for cell in row]
            cleaned_rows.append(cleaned_row)
    with open(output_path, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(cleaned_rows)
    return output_path

def handle_multiline(input_path: str, output_path: str) -> str:
    df = pd.read_csv(input_path).dropna(how='all').reset_index(drop=True)
    normalized_columns = {col.lower().strip(): col for col in df.columns}
    
    date_col = next((normalized_columns[col] for col in normalized_columns if col in [name.lower() for name in COLUMN_MAPPINGS["date"]]), None)
    narration_col = next((normalized_columns[col] for col in normalized_columns if col in [name.lower() for name in COLUMN_MAPPINGS["narration"]]), None)
    
    if not date_col or not narration_col:
        raise ValueError(f"Missing required columns. Found Date: {date_col}, Narration: {narration_col}")

    date_patterns = [
        r'\b\d{1,2}[-/|]\d{1,2}[-/|]\d{2,4}\b',
        r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
        r'\b\d{1,2}[-/](?:[A-Za-z]{3,9})[-/]\d{2,4}\b',
        r'\b\d{1,2}\s+(?:[A-Za-z]{3,9})\s+\d{2,4}\b',
        r'\b(?:[A-Za-z]{3,9})\s+\d{2,4}\b',
        r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+TO\s+\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'
    ]

    def is_date(x):
        if pd.isna(x):
            return False
        return any(re.search(pattern, str(x)) for pattern in date_patterns)

    df['is_date'] = df[date_col].astype(str).apply(is_date)
    cleaned_rows, current_row = [], None
    for _, row in df.iterrows():
        if row['is_date']:
            if current_row:
                cleaned_rows.append(current_row)
            current_row = list(row[:-1])
        elif current_row is not None:
            narration_original_idx = list(df.columns).index(narration_col)
            for col_idx in range(len(row) - 1):
                val = str(row[col_idx]) if not pd.isna(row[col_idx]) else ''
                if val.strip():
                    if col_idx == narration_original_idx:
                        current_row[col_idx] = f"{current_row[col_idx]} {val}".strip() if current_row[col_idx] else val
                    else:
                        current_row[col_idx] = f"{current_row[col_idx]} {val}".strip() if current_row[col_idx] else val
    if current_row:
        cleaned_rows.append(current_row)

    final_df = pd.DataFrame(cleaned_rows, columns=df.columns[:-1])
    final_df.to_csv(output_path, index=False)
    return output_path

def extract_suppliers(input_path: str, output_path: str) -> str:
    nlp = load_nlp()
    def extract_supplier_ner(text: str) -> str | None:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "ORG":
                return ent.text.strip()
        return None

    def extract_supplier_regex(text: str) -> str | None:
        match = re.search(
            r'\b([A-Z][A-Z0-9\s\&\.\-]+\s*(?:LTD|LLP|PVT LTD|PVT.LTD|COM|EXPRESS|PAYPAL|RETAIL|LIMITED|SECURITIES|BANK|CORP|GROUP|INC|CO|SOLUTIONS|SERVICES|TECHNOLOGIES|FINANCE|GLOBAL|INVESTMENTS|CONSULTING|MEDIA|NETWORK|SYSTEMS|DIGITAL|INDUSTRIES|ENTERPRISES|TRADING|AGENCY|CONSULTANTS|ASSOCIATES|INTERNATIONAL|MARKETING|DEVELOPMENT|MANAGEMENT|CONSORTIUM|PARTNERS|OFFICE|ADVISORS|TRUST|FUND|CAPITAL|HOLDINGS|DISTRIBUTORS|SUPPLIERS|MANUFACTURING|CHEMISTRY|LOGISTICS|CONSTRUCTION|ENGINEERING|HEALTHCARE|EDUCATION|RESEARCH))\b',
            text.upper()
        )
        return match.group(1).strip() if match else None

    def extract_supplier(text: str) -> str:
        return extract_supplier_ner(text) or extract_supplier_regex(text) or "UNKNOWN"

    df = pd.read_csv(input_path)
    narration_col_orig = next((col for col in df.columns if col.lower().strip() in [name.lower() for name in COLUMN_MAPPINGS["narration"]]), None)
    if not narration_col_orig:
        raise ValueError("Narration column not found.")
    df.rename(columns={narration_col_orig: 'narration'}, inplace=True)
    df['narration'] = df['narration'].astype(str).str.strip()
    df['supplier_name'] = df['narration'].apply(extract_supplier)
    df.to_csv(output_path, index=False)
    return output_path

def preprocess(df: pd.DataFrame) -> pd.DataFrame | None:
    df_processed = df.copy()
    df_processed = handle_duplicate_columns(df_processed)
    normalized_columns = {col.lower().strip(): col for col in df_processed.columns}
    
    date_col_orig = next((col for col in df_processed.columns if col.lower().strip() in [name.lower() for name in COLUMN_MAPPINGS["date"]]), None)
    money_in_col_orig = next((col for col in df_processed.columns if col.lower().strip() in [name.lower() for name in COLUMN_MAPPINGS["money_in"]]), None)
    money_out_col_orig = next((col for col in df_processed.columns if col.lower().strip() in [name.lower() for name in COLUMN_MAPPINGS["money_out"]]), None)

    if not date_col_orig:
        raise ValueError("Required date column not found in the CSV.")
    if not money_in_col_orig and not money_out_col_orig:
        raise ValueError("Required money in/out columns not found.")

    currency_symbols = '[$£₹€¥₩₽฿₺₪₫₦₱₲₴₵₡₸₭₮₼₺₳₤₥₦₧₰₱₲₳₴₵₷₸₺₻₼₽₾]'
    
    if money_in_col_orig:
        df_processed['money_in'] = pd.to_numeric(
            df_processed[money_in_col_orig].astype(str).str.replace(',', '').str.replace(currency_symbols, '', regex=True),
            errors='coerce'
        ).fillna(0)
    else:
        df_processed['money_in'] = 0
    
    if money_out_col_orig:
        df_processed['money_out'] = pd.to_numeric(
            df_processed[money_out_col_orig].astype(str).str.replace(',', '').str.replace(currency_symbols, '', regex=True),
            errors='coerce'
        ).fillna(0)
    else:
        df_processed['money_out'] = 0

    df_processed['amount'] = df_processed.apply(lambda x: x['money_in'] if x['money_in'] > 0 else x['money_out'], axis=1)
    df_processed['is_credit'] = df_processed['money_in'].apply(lambda x: 1 if x > 0 else 0)

    df_processed['date_col_processed'] = pd.to_datetime(df_processed[date_col_orig], errors='coerce', dayfirst=True)
    df_processed.dropna(subset=['date_col_processed'], inplace=True)

    df_processed['year'] = df_processed['date_col_processed'].dt.year
    df_processed['month'] = df_processed['date_col_processed'].dt.month
    df_processed['day'] = df_processed['date_col_processed'].dt.day
    df_processed['DayOfWeek'] = df_processed['date_col_processed'].dt.dayofweek
    df_processed['is_weekend'] = df_processed['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

    if 'supplier_name' in df_processed.columns:
        supplier_counts = df_processed['supplier_name'].value_counts()
        df_processed['supplier_frq'] = df_processed['supplier_name'].map(supplier_counts).fillna(0).astype(int)
    else:
        raise ValueError("Supplier_name column not found after extraction.")

    df_processed.drop(columns=['date_col_processed'], inplace=True)
    # Before returning df_processed:
    df_processed['original_date'] = df_processed[date_col_orig]

    # Ensure description is available
    if 'description' not in df_processed.columns:
        df_processed['description'] = df_processed.get('narration', '')

    return df_processed

def predict_anomalies(df: pd.DataFrame) -> pd.DataFrame | None:
    try:
        scaler = joblib.load("ml_models/scaler.pkl")
        iso_forest = joblib.load("ml_models/isolation.pkl")
        confidence_score_scaler = joblib.load("ml_models/confidence_score_scaler.pkl")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model files missing: {e}")
    except Exception as e:
        raise Exception(f"Error loading model files: {e}")

    features = ['day', 'month', 'year', 'amount', 'is_weekend', 'supplier_frq', 'is_credit']
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features for prediction: {', '.join(missing_features)}")

    X_scaled = scaler.transform(df[features].fillna(0))
    df['anomaly_raw'] = iso_forest.predict(X_scaled)
    raw_decision_scores = iso_forest.decision_function(X_scaled)
    inverted_scores = -raw_decision_scores
    df['confidence_score'] = confidence_score_scaler.transform(inverted_scores.reshape(-1, 1))
    # Round to 2 decimal places
    df['confidence_score'] = df['confidence_score'].round(2)

    if 'money_out' in df.columns and not df[df['is_credit'] == 0].empty:
        debit_threshold = df[df['is_credit'] == 0]['amount'].quantile(0.95)
        df['Flag'] = df.apply(lambda row: 1 if row['is_credit'] == 0 and row['amount'] >= debit_threshold else 0, axis=1)
    else:
        df['Flag'] = 0
    
    df['is_anomaly'] = df['Flag'].apply(lambda x: 1 if x == 1 else 0)
    return df

def full_processing_pipeline(input_file_path: str) -> pd.DataFrame | None:
    temp_dir = "temp_processing_data"
    os.makedirs(temp_dir, exist_ok=True)
    paths = {
        "input": input_file_path,
        "cleaned": os.path.join(temp_dir, "cleaned.csv"),
        "multiline": os.path.join(temp_dir, "multiline.csv"),
        "suppliers": os.path.join(temp_dir, "suppliers.csv")
    }

    try:
        print("Cleaning wrapped lines...")
        clean_wrapped_lines(paths["input"], paths["cleaned"])
        print("Handling multiline entries...")
        handle_multiline(paths["cleaned"], paths["multiline"])
        print("Extracting supplier names...")
        extract_suppliers(paths["multiline"], paths["suppliers"])
        print("Preprocessing data...")
        df = pd.read_csv(paths["suppliers"])
        df = preprocess(df.copy())
        if df is None:
            print("Error: Preprocessing failed.")
            return None
        print("Predicting anomalies...")
        df = predict_anomalies(df.copy())
        if df is None:
            print("Error: Anomaly prediction failed.")
            return None
        return df
    except Exception as e:
        print(f"An error occurred during file processing: {e}")
        return None
    finally:
        for path_key in ["cleaned", "multiline", "suppliers"]:
            if os.path.exists(paths[path_key]):
                os.remove(paths[path_key])
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)