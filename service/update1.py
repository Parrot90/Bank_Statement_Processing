# update.py
import streamlit as st
import spacy
import json
import os
import pandas as pd
from datetime import datetime
import re
from typing import Optional , Dict

class TransactionCategorizer:
    def __init__(self):
        self.rules_file = "rules.json"
        self.matching_records_output_file = "matchingCategory.csv"
        self.nlp = self._load_spacy_model()
        self._initialize_session_state()
        self._load_rules_and_categories()

    def _load_spacy_model(self):
        try:
            if "spacy_nlp_model" not in st.session_state:
                st.session_state.spacy_nlp_model = spacy.load("en_core_web_sm")
            return st.session_state.spacy_nlp_model
        except OSError:
    # Download the model if not present
                from spacy.cli import download
                download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")

    def _initialize_session_state(self):
        if "rules_ui" not in st.session_state:
            st.session_state.rules_ui = []
        if "show_mapping_output" not in st.session_state:
            st.session_state.show_mapping_output = False
        if "converted_df" not in st.session_state:
            st.session_state.converted_df = pd.DataFrame()
        if "working_df" not in st.session_state:
            st.session_state.working_df = pd.DataFrame()
        if "detected_column_maps" not in st.session_state:
            st.session_state.detected_column_maps = {}
        if "mappings_displayed_and_auto_proceeded" not in st.session_state:
            st.session_state.mappings_displayed_and_auto_proceeded = False
        if "all_known_categories" not in st.session_state:
            st.session_state.all_known_categories = set()

    def _load_rules_and_categories(self):
        initial_data = {"rules": [], "all_categories": []}
        
        if os.path.exists(self.rules_file):
            try:
                with open(self.rules_file, "r") as f:
                    loaded_data = json.load(f)
                    if isinstance(loaded_data, dict):
                        initial_data["rules"] = loaded_data.get("rules", [])
                        initial_data["all_categories"] = loaded_data.get("all_categories", [])
                    else:
                        initial_data["rules"] = loaded_data
            except json.JSONDecodeError:
                st.warning(f"Could not decode {self.rules_file}. Starting with empty rules and categories.")

        if not st.session_state.rules_ui:
            st.session_state.rules_ui = initial_data["rules"].copy()
        
        st.session_state.all_known_categories.update(initial_data["all_categories"])
        for rule_item in initial_data["rules"]:
            if "assign_category" in rule_item and rule_item["assign_category"].strip() != "":
                st.session_state.all_known_categories.add(rule_item["assign_category"].strip())

    def extract_supplier_ner(self, text: str) -> Optional[str]:
        """Extract supplier using SpaCy NER"""
        if self.nlp is None:
            return None
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == "ORG":
                return ent.text.strip()
        return None

    def extract_supplier_regex(self, text: str) -> Optional[str]:
        """Fallback regex-based extraction"""
        text_upper = text.upper()
        match = re.search(
            r'([A-Z][A-Z\s\*\&\.\-\']+?(?:LTD|LLP|COM|EXPRESS|PAYPAL|RETAIL|LIMITED|PVT|INC|CORP|GROUP|BANK|FINANCE|SERVICES|SOLUTIONS|CO|C O|& CO|\bASSOC\b|\bCONSULT\b|LLC|GMBH|AG|SP Z O O|P.J.S.C|PLC|SARL|NV|BV|AS|AB)\b)|'
            r'\b(?:AMAZON|GOOGLE|APPLE|MICROSOFT|FACEBOOK|EBAY|PAYTM|PHONEPE|GAYLE|TATA|RELIANCE|ADIDAS|NIKE|ZARA|H&M|SWIGGY|ZOMATO|UBER|OLA|FLIPKART|MYNTRA|BIGBASKET|GROFERS|DOMINOS|PIZZA HUT|KFC|BURGER KING|MCDONALDS|STARBUCKS|COSTA COFFEE|AIRTEL|VODAFONE|JIO|IDEA|SBI|HDFC|ICICI|AXIS|YES BANK|PNB|BOB)\b',
            text_upper
        )
        if match:
            for g in match.groups():
                if g is not None:
                    return g.strip()
        return None

    def extract_supplier(self, text: str) -> str:
        """Main supplier extractor with fallback"""
        if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
            return "UNKNOWN"
        supplier = self.extract_supplier_ner(text)
        if not supplier:
            supplier = self.extract_supplier_regex(text)
        return supplier or "UNKNOWN"

    def _standardize_columns(self, df):
        """Standardize CSV columns to expected names"""
        column_mapping = {
            "date": [
                "TransactionDate", "transaction_date", "Txn Date", "txn_date", "date",
                "transaction date", "entry date", "posted date", "value date", "tran. date",
                "effective date", "date of transaction", "booking date", "execution date",
                "transaction_dt", "tran date", "tran dt", "tdate", "acct date", "business date"
            ],
            "money_out": ["Withdrawal", "withdraw","OUT","WithdrawalAmt.", "debit", "amount_out", "money_out", "money out","Debited", "Dr"],
            "money_in": ["Deposit", "credit","IN" ,"amount_in","DepositAmt.", "money_in", "money in","Credited", "Cr"],
            "supplier": ["supplier_name", "supplier","narration","merchant", "vendor", "name",
                         "description", "transaction details", "details", "particulars", "remarks", "transaction_description"],
            "category": ["category","Label","type"],
            "balance": ["balance","Amount" ,"running_balance", "account_balance"]
        }

        df_columns_lower = [col.strip().lower() for col in df.columns]
        original_columns = list(df.columns)
        detected_maps_for_display = {}

        for standard_col, possible_names in column_mapping.items():
            found = False
            for possible_name in possible_names:
                possible_name_clean = possible_name.strip().lower()
                if possible_name_clean in df_columns_lower:
                    idx = df_columns_lower.index(possible_name_clean)
                    actual_col_name = original_columns[idx]
                    df.rename(columns={actual_col_name: standard_col}, inplace=True)
                    detected_maps_for_display[standard_col] = actual_col_name
                    found = True
                    break
            if not found and standard_col not in ["category", "balance"]:
                st.warning(f"Could not map a column for '{standard_col}'. Please ensure your CSV has a recognized column for it.")
                detected_maps_for_display[standard_col] = "Not Detected"

        st.session_state.detected_column_maps = detected_maps_for_display
        return df

    def flexible_date_parser(self, date_series):
        """Parse dates flexibly using multiple common formats"""
        def parse_single_date(date_str):
            if pd.isna(date_str) or str(date_str).strip() == '' or str(date_str).strip().lower() in ['none', 'null', 'nan']:
                return None

            date_str = str(date_str).strip()
            date_formats = [
                "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d",
                "%m-%d-%Y", "%m/%d/%Y", "%d.%m.%Y", "%Y.%m.%d",
                "%d %m %Y", "%Y %m %d", "%m %d %Y",
                "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y",
                "%d-%b-%Y", "%d-%B-%Y", "%b-%d-%Y", "%B-%d-%Y",
                "%d/%b/%Y", "%d/%B/%Y", "%b/%d/%Y", "%B/%d/%Y",
                "%d.%b.%Y", "%d.%B.%Y", "%b.%d.%Y", "%B.%d.%Y",
                "%d-%m-%y", "%d/%m/%y", "%y-%m-%d", "%y/%m/%d",
                "%m-%d-%y", "%m/%d/%y", "%d.%m.%y", "%y.%m.%d",
                "%d %b %y", "%d %B %y", "%b %d, %y", "%B %d, %y",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%SZ",
                "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S",
                "%m-%d-%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
                "%Y%m%d", "%d%m%Y", "%m%d%Y",
            ]

            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    return parsed_date.date()
                except:
                    continue

            try:
                parsed_date = pd.to_datetime(date_str, infer_datetime_format=True, errors='coerce')
                if not pd.isna(parsed_date):
                    return parsed_date.date()
            except:
                pass

            try:
                date_num = float(date_str)
                if 1 <= date_num <= 2958465:
                    if date_num >= 60:
                        date_num -= 1
                    excel_epoch = datetime(1900, 1, 1)
                    parsed_date = excel_epoch + pd.Timedelta(days=date_num - 1)
                    return parsed_date.date()
            except:
                pass

            try:
                timestamp = float(date_str)
                if timestamp > 1e9:
                    parsed_date = datetime.fromtimestamp(timestamp)
                    return parsed_date.date()
                elif timestamp > 1e12:
                    parsed_date = datetime.fromtimestamp(timestamp / 1000)
                    return parsed_date.date()
            except:
                pass

            return None

        parsed_dates = date_series.apply(parse_single_date)
        return parsed_dates

    def convert_to_standard_format(self, df):
        """Convert DataFrame to standard format with required columns"""
        df = self._standardize_columns(df)

        required_columns = ["supplier", "date"]
        if not ("money_in" in df.columns or "money_out" in df.columns):
            st.error("Missing required columns: Need either 'money_in' or 'money_out' column")
            return None

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            return None

        if "supplier" in df.columns:
            df['original_description'] = df['supplier'].astype(str).copy() 
            with st.spinner("Extracting supplier names using NER and regex..."):
                df['supplier'] = df['supplier'].astype(str).apply(self.extract_supplier)
        else:
            st.warning("No 'supplier' or 'description' column found to apply advanced supplier extraction.")

        rename_dict = {
            "supplier": "Supplier",
            "date": "Date",
            "original_description": "Original Description"
        }

        if "money_in" in df.columns:
            rename_dict["money_in"] = "Money_In"
        if "money_out" in df.columns:
            rename_dict["money_out"] = "Money_Out"
        if "category" in df.columns:
            rename_dict["category"] = "Category"
        if "balance" in df.columns:
            rename_dict["balance"] = "Balance"

        df = df.rename(columns=rename_dict)

        if "Category" not in df.columns:
            df["Category"] = ""
        if "Money_In" not in df.columns:
            df["Money_In"] = 0
        if "Money_Out" not in df.columns:
            df["Money_Out"] = 0
        if "Balance" not in df.columns:
            df["Balance"] = 0
        if "Original Description" not in df.columns:
            df["Original Description"] = ""

        df["Date"] = self.flexible_date_parser(df["Date"])

        for col in ["Money_In", "Money_Out", "Balance"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        display_order = ["Date", "Supplier", "Original Description", "Money_In", "Money_Out", "Balance", "Category"]
        display_order_existing = [col for col in display_order if col in df.columns]
        other_cols = [col for col in df.columns if col not in display_order_existing]
        df = df[display_order_existing + sorted(other_cols)]

        return df

    def match_single_rule(self, row, rule):
        """Check if a single row matches a rule"""
        supplier_val = str(row.get("Supplier", "")).strip().lower()

        try:
            money_in_val = float(row.get("Money_In", 0))
            money_out_val = float(row.get("Money_Out", 0))
        except:
            money_in_val = 0
            money_out_val = 0

        try:
            date_val = pd.to_datetime(str(row.get("Date"))).date()
        except:
            date_val = None

        category_val = str(row.get("Category", "")).strip().lower()

        rule_suppliers = [s.strip().lower() for s in rule.get("supplier", [])]
        supplier_match = supplier_val in rule_suppliers if rule_suppliers else True

        transaction_type_match = True
        if "transaction_type" in rule:
            if rule["transaction_type"] == "Pay In (Money In)" and money_in_val == 0:
                transaction_type_match = False
            elif rule["transaction_type"] == "Pay Out (Money Out)" and money_out_val == 0:
                transaction_type_match = False

        amount_match = True
        relevant_amount = money_in_val if money_in_val > 0 else money_out_val

        if "min_amount" in rule and relevant_amount < rule["min_amount"]:
            amount_match = False
        if "max_amount" in rule and relevant_amount > rule["max_amount"]:
            amount_match = False

        date_match = True
        if date_val:
            if "from_date" in rule:
                from_d = datetime.strptime(rule["from_date"], "%Y-%m-%d").date()
                if date_val < from_d:
                    date_match = False
            if "to_date" in rule:
                to_d = datetime.strptime(rule["to_date"], "%Y-%m-%d").date()
                if date_val > to_d:
                    date_match = False

        rule_categories = [c.strip().lower() for c in rule.get("category", [])]
        category_match = category_val in rule_categories if rule_categories else True

        return supplier_match and amount_match and date_match and category_match and transaction_type_match

    def save_rules(self):
        """Save current rules to JSON file"""
        data_to_save = {
            "rules": st.session_state.rules_ui,
            "all_categories": list(st.session_state.all_known_categories)
        }
        with open(self.rules_file, "w") as f:
            json.dump(data_to_save, f, indent=2)

    def run(self):
        """Main method to run the Streamlit app"""
        st.set_page_config(page_title="Rule Based System-Transaction Categorization", layout="wide")
        st.title("Rule Based - Transaction Categorization")

        # === CSV Upload ===
        st.header("Upload CSV File")
        uploaded_file = st.file_uploader("Upload any CSV (we'll auto-detect columns)", type=["csv"])

        if uploaded_file:
            if st.session_state.get("last_uploaded_file_id") != uploaded_file.file_id:
                st.session_state.last_uploaded_file_id = uploaded_file.file_id
                raw_df = pd.read_csv(uploaded_file)
                
                with st.spinner("Detecting columns and converting to standard format..."):
                    df_converted = self.convert_to_standard_format(raw_df.copy())
                
                if df_converted is not None:
                    st.session_state.converted_df = df_converted
                    st.session_state.working_df = df_converted.copy()
                    st.session_state.show_mapping_output = True
                    st.session_state.mappings_displayed_and_auto_proceeded = False

                    if "Category" in df_converted.columns:
                        new_categories = df_converted["Category"].dropna().astype(str).str.strip().tolist()
                        st.session_state.all_known_categories.update(new_categories)
                    st.session_state.all_known_categories.discard("")

            if st.session_state.show_mapping_output and not st.session_state.mappings_displayed_and_auto_proceeded:
                st.header("Detected Column Mappings")
                st.write("The application has automatically mapped your CSV columns to the following standard categories:")

                mapping_data = []
                for standard_col, detected_col in st.session_state.detected_column_maps.items():
                    mapping_data.append({"Standard Column": standard_col.replace('_', ' ').title(), 
                                       "Detected Original Column": detected_col})
                
                st.dataframe(pd.DataFrame(mapping_data), hide_index=True, use_container_width=True)
                st.success("Column mappings detected successfully! Proceeding to rule creation...")

                st.session_state.show_mapping_output = False
                st.session_state.mappings_displayed_and_auto_proceeded = True
                st.rerun()

        df = st.session_state.working_df

        if not df.empty and not st.session_state.show_mapping_output:
            st.header("Current Dataset")
            display_df_initial = df.drop(columns=["Category"], errors='ignore') 
            st.dataframe(display_df_initial, use_container_width=True) 
            
            supplier_options = sorted(df["Supplier"].dropna().astype(str).unique().tolist())
            
            current_df_categories = set(df["Category"].dropna().astype(str).unique().tolist())
            current_df_categories.discard("")
            combined_categories = sorted(list(st.session_state.all_known_categories.union(current_df_categories)))

            # === Rule Creation ===
            st.header("Create Rule")

            st.subheader("Supplier Details")
            supplier_select = st.multiselect("Select Supplier(s) (optional)", supplier_options)
            suppliers = [s.strip().lower() for s in supplier_select]

            st.subheader("Transaction Type")
            transaction_type = st.radio("Select transaction type:", ["Pay In (Money In)", "Pay Out (Money Out)"], 
                                      key="transaction_type_radio")

            st.subheader(" Set Amount ")
            col1, col2 = st.columns(2)
            with col1:
                min_amount = st.number_input("Min Amount (optional)", step=1.0, key="min_amount")
            with col2:
                max_amount = st.number_input("Max Amount (optional)", step=1.0, key="max_amount")

            st.subheader("Date Filter")
            date_filter_enabled = st.checkbox("Enable Date Filter")
            from_date, to_date = None, None
            if date_filter_enabled:
                col3, col4 = st.columns(2)
                with col3:
                    from_date = st.date_input("From Date (optional)", key="from_date")
                with col4:
                    to_date = st.date_input("To Date (optional)", key="to_date")

            st.subheader("Category Details")
            all_category_options = ["Add New Category..."] + combined_categories
            
            selected_assign_category_option = st.selectbox(
                "Select or Add a Category to apply to matching records",
                options=all_category_options,
                key="assign_category_dropdown"
            )

            assign_category = None
            if selected_assign_category_option == "Add New Category...":
                new_manual_category = st.text_input("Enter new category name", key="manual_category_input")
                if new_manual_category.strip():
                    assign_category = new_manual_category.strip()
                    st.session_state.all_known_categories.add(assign_category)
            else:
                assign_category = selected_assign_category_option
                if assign_category.strip():
                    st.session_state.all_known_categories.add(assign_category)
            
            if st.button("Add Rule"):
                rule = {}
                if suppliers:
                    rule["supplier"] = suppliers
                if min_amount:
                    rule["min_amount"] = float(min_amount)
                if max_amount:
                    rule["max_amount"] = float(max_amount)
                if date_filter_enabled and from_date:
                    rule["from_date"] = str(from_date)
                if date_filter_enabled and to_date:
                    rule["to_date"] = str(to_date)
                if assign_category and assign_category != "Add New Category...":
                    rule["assign_category"] = assign_category
                
                if transaction_type: 
                    rule["transaction_type"] = transaction_type

                st.session_state.rules_ui.append(rule)
                self.save_rules()
                st.success("Rule added and saved!")
                st.rerun()

            # === View & Delete Rules ===
            if st.session_state.rules_ui:
                st.subheader("Current Rules")
                for idx, rule in enumerate(st.session_state.rules_ui):
                    rule_display = f"Rule {idx + 1}: "
                    if "assign_category" in rule:
                        rule_display += f"Category: {rule['assign_category']}, "
                    if "category" in rule and rule["category"]: 
                        rule_display += f"Filter Category: {rule['category']}, "
                    if "supplier" in rule:
                        rule_display += f"Supplier: {rule['supplier']}, "
                    if "transaction_type" in rule:
                        rule_display += f"Type: {rule['transaction_type']}, "
                    if "min_amount" in rule or "max_amount" in rule:
                        rule_display += f"Amount: {rule.get('min_amount', 0)}-{rule.get('max_amount', '∞')}, "

                    st.markdown(f"*{rule_display.rstrip(', ')}*")
                    if st.button(f"Delete Rule {idx + 1}", key=f"del_{idx}"):
                        st.session_state.rules_ui.pop(idx)
                        self.save_rules()
                        st.rerun()

            # === Individual Rule Application ===
            st.header("Apply Rule")

            if st.session_state.rules_ui and not df.empty:
                rule_labels = [f"Rule {i+1}: {r.get('assign_category', r.get('category', 'Filter'))}" 
                              for i, r in enumerate(st.session_state.rules_ui)]
                selected_rule_idx = st.selectbox("Select a rule to apply:", 
                                               options=range(len(st.session_state.rules_ui)), 
                                               format_func=lambda x: rule_labels[x])

                if st.button("Apply Selected Rule"):
                    selected_rule = st.session_state.rules_ui[selected_rule_idx]

                    matches = df.apply(lambda row: self.match_single_rule(row, selected_rule), axis=1)
                    matched_count = matches.sum()

                    if matched_count > 0:
                        if "assign_category" in selected_rule:
                            df.loc[matches, "Category"] = selected_rule["assign_category"]

                        st.session_state.working_df = df.copy()
                        st.success(f"Applied {rule_labels[selected_rule_idx]} to {matched_count} records!")

                        matched_records = df[matches]
                        st.subheader("✅ Matching Records ")
                        st.dataframe(matched_records, use_container_width=True)

                        try:
                            if os.path.exists(self.matching_records_output_file):
                                matched_records.to_csv(self.matching_records_output_file, 
                                                      mode='a', header=False, index=False)
                                st.success(f"Matching records appended to {self.matching_records_output_file}")
                            else:
                                matched_records.to_csv(self.matching_records_output_file, 
                                                     mode='w', header=True, index=False)
                                st.success(f"Matching records automatically saved to new file: {self.matching_records_output_file}")
                        except Exception as e:
                            st.error(f"Could not automatically save matching records to {self.matching_records_output_file}: {e}")

                    else:
                        st.warning("No records matched the selected rule.")

        # === Show Final Results ===
        if not df.empty:
            st.header("📊 Updated Dataset ")
            st.dataframe(df, use_container_width=True)
            
            categorized_count = len(df[df["Category"].str.strip() != ""])
            uncategorized_count = len(df[df["Category"].str.strip() == ""])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Categorized Records", categorized_count)
            with col3:
                st.metric("Uncategorized Records", uncategorized_count)
            
            csv_final = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Final Dataset", data=csv_final, 
                              file_name="final_categorized_data.csv", mime="text/csv")

        elif uploaded_file is None:
            st.info("Please upload a file to start categorizing records.")

def transaction_categorization_app():
    categorizer = TransactionCategorizer()
    categorizer.run()

if __name__ == "__main__":
    transaction_categorization_app()