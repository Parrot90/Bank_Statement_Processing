import pandas as pd
import numpy as np
import re
import spacy

class DuplicateDetectionService:
    def __init__(self, file_data):
        self.nlp = spacy.load("en_core_web_sm")
        if isinstance(file_data, str):
            self.df = pd.read_csv(file_data)
        else:
            self.df = pd.read_csv(file_data)
        self.df = self._standardize_columns(self.df)
        self._clean_and_prepare_data()
        self._generate_transaction_ids()
        self.df["IsDuplicate"] = 0
        self.duplicate_pairs = []
        self._duplicates_detected = False
        required_columns = {"transaction_id", "date", "supplier", "money_out", "money_in"}
        if not required_columns.issubset(self.df.columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")

    def _extract_supplier(self, text):
        if not isinstance(text, str) or not text.strip():
            return "unknown"
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == "ORG":
                return ent.text.strip().lower()
        match = re.search(r'([A-Z][A-Z\s\*&\.]+(?:LTD|LLP|COM|EXPRESS|PAYPAL|RETAIL|LIMITED))', text.upper())
        if match:
            return match.group(1).strip().lower()
        return "unknown"

    def _generate_transaction_ids(self):
        self.df["transaction_id"] = ["T" + str(i+1).zfill(4) for i in range(len(self.df))]

    def _standardize_columns(self, df):
        column_mapping = {
            "date": ["Transaction Date", "transaction date", "txn date", "date"],
            "money_out": ["Withdrawal", "withdraw", "debit", "amount_out", "money_out", "Money Out","out"],
            "money_in": ["Deposit", "credit", "amount_in", "money_in", "Money In","in"],
            "supplier": ["supplier_name", "supplier", "merchant", "vendor","transaction"],
            "description": ["narration", "description", "remarks", "particulars","transaction"]
        }
        df_columns_lower = [col.strip().lower() for col in df.columns]
        for standard_col, possible_names in column_mapping.items():
            for possible_name in possible_names:
                if possible_name.strip().lower() in df_columns_lower:
                    idx = df_columns_lower.index(possible_name.strip().lower())
                    actual_col_name = df.columns[idx]
                    df.rename(columns={actual_col_name: standard_col}, inplace=True)
                    break
        if "description" not in df.columns:
            df["description"] = "N/A"
        if "supplier" not in df.columns:
            df["supplier"] = "unknown"
        return df

    def _clean_currency(self, value):
        if pd.isna(value):
            return 0.0
        value = str(value)
        value = re.sub(r"[^\d\.\-]", "", value.replace(",", "").strip())
        try:
            return float(value) if value else 0.0
        except ValueError:
            return 0.0

    def _clean_and_prepare_data(self):
        self.df["supplier"] = (
            self.df["supplier"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"nan": "unknown"})
            .fillna("unknown")
        )
        missing_supplier = self.df["supplier"].isin(["unknown"]).sum()
        if missing_supplier > 0:
            print(f"Extracting {missing_supplier} missing supplier(s) from description...")
            self.df["supplier"] = [
                s if s != "unknown" else self._extract_supplier(desc)
                for s, desc in zip(self.df["supplier"], self.df["description"])
            ]
        self.df["money_out"] = self.df["money_out"].apply(self._clean_currency)
        self.df["money_in"] = self.df["money_in"].apply(self._clean_currency)
        self.df["date"] = pd.to_datetime(self.df["date"], errors='coerce', dayfirst=True)
        print("Invalid dates:", self.df["date"].isna().sum())
        self.df["date"] = self.df["date"].fillna(pd.Timestamp("1900-01-01"))
        self.df["date"] = self.df["date"].dt.strftime('%d-%m-%Y')

    def rule_based_check(self):
        self.df["IsDuplicate"] = 0
        self.duplicate_pairs = []
        grouped = self.df.groupby(["supplier", "money_out", "money_in", "date"])
        for _, group in grouped:
            if len(group) > 1:
                tx_ids = list(group["transaction_id"])
                for i in range(len(tx_ids)):
                    for j in range(i + 1, len(tx_ids)):
                        self.duplicate_pairs.append((tx_ids[i], tx_ids[j]))
                        self.df.loc[self.df["transaction_id"] == tx_ids[j], "IsDuplicate"] = 1
        self._duplicates_detected = True

    def detect_duplicates(self):
        self.rule_based_check()
        return {
            "total_transactions": int(len(self.df)),
            "total_duplicates": int(self.df["IsDuplicate"].sum())
        }

    def get_duplicate_details(self):
        self.rule_based_check()
        details = []
        for _, t2 in self.duplicate_pairs:
            row2 = self.df[self.df["transaction_id"] == t2].iloc[0].copy()
            details.append(row2.to_dict())
        clean_details = pd.json_normalize(details)
        clean_details.replace([np.inf, -np.inf], np.nan, inplace=True)
        clean_details.fillna("N/A", inplace=True)
        return clean_details.to_dict(orient="records")