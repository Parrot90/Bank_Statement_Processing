import json
import os
from datetime import datetime
from typing import Dict, List
import pandas as pd

RULES_PATH = os.getenv("RULES_PATH", "app\\rules.json")
DATA_PATH = os.getenv("DATA_PATH", "app\\Clean_file_updated.csv")

def load_rules():
    """Load rules from JSON file"""
    try:
        if os.path.exists(RULES_PATH):
            with open(RULES_PATH, 'r') as f:
                return json.load(f)
        else:
            return []
    except Exception:
        return []

def match_rule(transaction: Dict, rule: Dict) -> bool:
    """Check if a transaction matches a given rule"""
    try:
        supplier = str(transaction["supplier"]).strip().lower()
        amount = float(transaction["amount"])
        date_val = transaction["date"]

        # Supplier match
        if "supplier" in rule and rule["supplier"]:
            rule_suppliers = rule["supplier"] if isinstance(rule["supplier"], list) else [rule["supplier"]]
            rule_suppliers = [s.strip().lower() for s in rule_suppliers]
            if not any(supplier_rule in supplier for supplier_rule in rule_suppliers):
                return False

        # Amount match
        if "min_amount" in rule and rule["min_amount"] is not None:
            if amount < float(rule["min_amount"]):
                return False
        if "max_amount" in rule and rule["max_amount"] is not None:
            if amount > float(rule["max_amount"]):
                return False

        # Date match
        if "from_date" in rule and "to_date" in rule:
            if rule["from_date"] and rule["to_date"]:
                try:
                    from_date = datetime.strptime(rule["from_date"], "%Y-%m-%d").date()
                    to_date = datetime.strptime(rule["to_date"], "%Y-%m-%d").date()
                    if not (from_date <= date_val <= to_date):
                        return False
                except:
                    return False

        return True
    except Exception:
        return False

def predict_categorie(input_data: Dict):
    """Predict category for a single transaction"""
    rules = load_rules()
    
    if not rules:
        return {
            "category": "Uncategorized",
            "rule_applied": "no_rules_loaded"
        }
    
    # Convert date if it's a string
    if isinstance(input_data["date"], str):
        try:
            input_data["date"] = datetime.strptime(input_data["date"], "%Y-%m-%d").date()
        except:
            return {
                "category": "Uncategorized",
                "rule_applied": "invalid_date"
            }

    # Apply rules
    for i, rule in enumerate(rules):
        if match_rule(input_data, rule):
            category = rule.get("category", "Uncategorized")
            if isinstance(category, list):
                category = category[0] if category else "Uncategorized"
            
            return {
                "category": category,
                "rule_applied": f"rule_{i+1}"
            }

    return {
        "category": "Uncategorized",
        "rule_applied": "no_match"
    }

def apply_rules_to_dataset():
    """Apply rules to entire dataset"""
    if not os.path.exists(DATA_PATH):
        return {"error": f"Dataset file not found at {DATA_PATH}"}
    
    try:
        df = pd.read_csv(DATA_PATH)
        df.columns = df.columns.str.lower()
        
        results = []
        for _, row in df.iterrows():
            try:
                transaction = {
                    "supplier": str(row.get("name", row.get("supplier", ""))).strip(),
                    "amount": float(row.get("amount", 0)),
                    "date": pd.to_datetime(row.get("date"), errors='coerce').date()
                }

                if pd.isnull(transaction["date"]) or transaction["amount"] <= 0:
                    continue

                result = predict_categorie(transaction)
                if result["rule_applied"] != "no_match":
                    row_data = row.to_dict()
                    row_data.update({
                        "predicted_category": result["category"],
                        "rule_applied": result["rule_applied"]
                    })
                    results.append(row_data)
            except:
                continue

        return results
    except Exception as e:
        return {"error": f"Error processing dataset: {e}"}