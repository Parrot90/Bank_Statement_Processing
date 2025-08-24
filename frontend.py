import streamlit as st
import requests
import pandas as pd
import json
import tempfile
from PIL import Image
import os
from datetime import datetime, date
import time
from datetime import datetime, date
from dotenv import load_dotenv
load_dotenv()




st.set_page_config(page_title="🧾 Receipt & Invoice Parser", page_icon="🧾", layout="wide")

st.title("🧾 Smart Receipt/Invoice/Bank Statement Parser")



#BASE_URL = "hhttps://644db7a87e70.ngrok-free.app "
BASE_URL = os.getenv("BASE_URL" , "http://backend:8000" )  # Update with your FastAPI backend URL



# Tabs setup
tab1, tab2, tab3 , tab4 ,tab5 , tab6 , tab7 , tab8= st.tabs([
    "📑 Invoice & Receipt PDFs",
    "🖼️ Invoice & Receipt Images",
    "🏦 Bank Statement",
    "Rule Based Categoriser ",
    "💳 Recommendation System",
    "📊 Transaction Category Prediction",
    "Anamoly and Duplicate Detection",
    "Single Transaction anamoly and duplicate detection"
])

with tab1:
    st.header("Upload Invoice or Receipt PDF")

    doc_type = st.radio("Select document type:", ["Invoice", "Receipt"], horizontal=True, key="pdf_doc_type")
    pdf_file = st.file_uploader(
        f"Upload a {doc_type.lower()} file (PDF)",
        type=["pdf"],
        key=f"{doc_type.lower()}_pdf"
    )

    if pdf_file and st.button(f"🚀 Extract {doc_type}", key=f"extract_{doc_type.lower()}_btn"):
        with st.spinner(f"🔍 Analyzing {doc_type.lower()}..."):
            try:
                endpoint = f"{BASE_URL}/parse-invoice/" if doc_type == "Invoice" else f"{BASE_URL}parse-receipt/"
                response = requests.post(
                    endpoint,
                    files={"file": (pdf_file.name, pdf_file.getvalue(), pdf_file.type)},
                    timeout=900
                )

                if response.status_code == 200:
                    data = response.json()

                    if "error" in data:
                        st.warning(f"⚠️ {data['error']}")
                    else:
                        # Summary Table
                        st.markdown(f"### 📌 {doc_type} Summary")
                        if doc_type == "Invoice":
                            summary = {
                                "Supplier Name": data.get("supplier_name", ""),
                                "Invoice Date": data.get("invoice_date", ""),
                                "Due Date": data.get("due_date", ""),
                                "Currency": data.get("currency", ""),
                                "Total Amount": data.get("total_amount", ""),
                                "Tax Amount": data.get("tax_amount", "")
                            }
                        else:
                            summary = {
                                "Store Name": data.get("store_name", ""),
                                "Date": data.get("date", ""),
                                "Currency": data.get("currency", ""),
                                "Total Amount": data.get("total_amount", ""),
                                "Tax Details": data.get("tax_details", ""),
                                "Transaction Number": data.get("transaction_number", ""),
                                "Card Details": data.get("card_details", ""),
                                "Service Fee": data.get("service_fee", "")
                            }

                        st.table(pd.DataFrame(summary.items(), columns=["Field", "Value"]))

                        # Items Table
                        st.markdown(f"### 🛒 {doc_type} Items")
                        items = data.get("items", [])
                        if items:
                            st.dataframe(pd.DataFrame(items).replace("", "—"), use_container_width=True)
                        else:
                            st.info(f"No items found in the {doc_type.lower()}.")

                        # JSON Download
                        st.download_button(
                            label="💾 Download JSON",
                            data=json.dumps(data, indent=2),
                            file_name=f"{doc_type.lower()}_data.json",
                            mime="application/json"
                        )
                else:
                    st.error(f"❌ Server returned {response.status_code}: {response.text}")

            except Exception as e:
                st.error(f"⚠️ Exception occurred: {e}")

with tab2:
    st.header("Upload Invoice or Receipt Image")

    doc_type = st.radio("Choose the document type to process:", ["Invoice", "Receipt"], horizontal=True, key="image_doc_type")
    uploaded_file = st.file_uploader(
        f"Upload a {doc_type.lower()} image",
        type=["png", "jpg", "jpeg"],
        key=f"{doc_type.lower()}_image"
    )

    if uploaded_file:
        col1, col2 = st.columns([1, 2])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with col1:
            img = Image.open(tmp_path)
            rotated_img = img.rotate(-90, expand=True)
            st.image(rotated_img, caption=f"Uploaded {doc_type} (Rotated -90°)", use_container_width=True)

        with col2:
            st.markdown(f"#### 🧠 Extracted {doc_type} Information")
            result_tab1, result_tab2 = st.tabs(["Parsed Output", "Raw Output"])
            with st.spinner(f"Analyzing {doc_type.lower()} .."):
                try:
                    with open(tmp_path, "rb") as f:
                        files = {"file": (uploaded_file.name, f, uploaded_file.type)}
                        endpoint = f"{BASE_URL}/api/invoice" if doc_type == "Invoice" else f"{BASE_URL}/api/receipt"
                        response = requests.post(endpoint, files=files)
                    response.raise_for_status()
                    json_data = response.json()
                except requests.exceptions.HTTPError as http_err:
                    with result_tab1:
                        st.error(f"HTTP error processing {doc_type.lower()}: {str(http_err)}")
                        if doc_type == "Invoice":
                            st.table(pd.DataFrame.from_dict({
                                "Supplier Name": [""],
                                "Invoice Date": [""],
                                "Total Amount": [""],
                                "Tax Amount": [""],
                                "Due Date": [""],
                                "Currency": [""],
                            }, orient="index", columns=["Value"]))
                            st.table(pd.DataFrame(columns=["description", "quantity", "unit_price", "total_price"]))
                        else:
                            st.table(pd.DataFrame.from_dict({
                                "Store Name": [""],
                                "Date": [""],
                                "Currency": [""],
                                "Total Amount": [""],
                                "Tax Details": [""],
                                "Transaction Number": [""],
                                "Card Details": [""],
                                "Service Fee": [""],
                            }, orient="index", columns=["Value"]))
                            st.table(pd.DataFrame(columns=["name", "description", "price", "unit_price", "quantity", "discount", "total", "line_total"]))
                    with result_tab2:
                        st.error(f"Raw output error: {str(http_err)}")
                        st.code(str(http_err))
                    st.stop()
                except Exception as e:
                    with result_tab1:
                        st.error(f"Error processing {doc_type.lower()}: {str(e)}")
                        if doc_type == "Invoice":
                            st.table(pd.DataFrame.from_dict({
                                "Supplier Name": [""],
                                "Invoice Date": [""],
                                "Total Amount": [""],
                                "Tax Amount": [""],
                                "Due Date": [""],
                                "Currency": [""],
                            }, orient="index", columns=["Value"]))
                            st.table(pd.DataFrame(columns=["description", "quantity", "unit_price", "total_price"]))
                        else:
                            st.table(pd.DataFrame.from_dict({
                                "Store Name": [""],
                                "Date": [""],
                                "Currency": [""],
                                "Total Amount": [""],
                                "Tax Details": [""],
                                "Transaction Number": [""],
                                "Card Details": [""],
                                "Service Fee": [""],
                            }, orient="index", columns=["Value"]))
                            st.table(pd.DataFrame(columns=["name", "description", "price", "unit_price", "quantity", "discount", "total", "line_total"]))
                    with result_tab2:
                        st.error(f"Raw output error: {str(e)}")
                        st.code(str(e))
                    st.stop()
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

                with result_tab1:
                    st.subheader("🔹 Summary Fields")
                    if doc_type == "Invoice":
                        summary_fields = {
                            "Supplier Name": json_data.get("supplier_name", ""),
                            "Invoice Date": json_data.get("invoice_date", ""),
                            "Total Amount": json_data.get("total_amount", ""),
                            "Tax Amount": json_data.get("tax_amount", ""),
                            "Due Date": json_data.get("due_date", ""),
                            "Currency": json_data.get("currency", "")
                        }
                        item_fields = ["description", "quantity", "unit_price", "total_price"]
                    else:
                        summary_fields = {
                            "Store Name": json_data.get("store_name", ""),
                            "Date": json_data.get("date", ""),
                            "Currency": json_data.get("currency", ""),
                            "Total Amount": json_data.get("total_amount", ""),
                            "Tax Details": json_data.get("tax_details", ""),
                            "Transaction Number": json_data.get("transaction_number", ""),
                            "Card Details": json_data.get("card_details", ""),
                            "Service Fee": json_data.get("service_fee", "")
                        }
                        item_fields = ["name", "description", "price", "unit_price", "quantity", "discount", "total", "line_total"]

                    st.table(pd.DataFrame.from_dict(summary_fields, orient="index", columns=["Value"]))
                    st.subheader(f"🛒 {doc_type} Items")
                    items = json_data.get("items", [])
                    if items and isinstance(items, list) and len(items) > 0:
                        df_items = pd.DataFrame(items)
                        for col in item_fields:
                            if col not in df_items.columns:
                                df_items[col] = ""
                        df_items = df_items[item_fields]
                        st.table(df_items)
                    else:
                        st.info(f"No items found in {doc_type.lower()}.")
                        st.table(pd.DataFrame(columns=item_fields))

                with result_tab2:
                    st.subheader("📝 Raw JSON Output")
                    st.code(json.dumps(json_data, indent=2))

    else:
        st.info(f"Please upload a {doc_type.lower()} image to proceed.")

with tab3:
    st.header("Bank Statement Extractor")

    uploaded_file = st.file_uploader("📤 Upload Bank Statement", type=["pdf","png", "jpg", "jpeg"], key="bank_pdf_upload")

    if uploaded_file:
        st.info("📨 Uploading to server and extracting...")

        try:
            response = requests.post(
                f"{BASE_URL}/process-bank-statement/",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                timeout=600
            )

            if response.status_code != 200:
                st.error(f"❌ API Error {response.status_code}: {response.json().get('error', 'Unknown')}")
            else:
                result = response.json()
                
                # Top Summary
                st.subheader("📌 Account Summary")
                summary_fields = {
                    "Account Holder": result.get("account_holder_name", "–"),
                    "Account Number": result.get("account_number", "–"),
                    "Bank Name": result.get("bank_name", "–"),
                    "Statement Period": result.get("statement_period", "–"),
                    "Currency": result.get("currency", "–"),
                    "Opening Balance": result.get("opening_balance", "–"),
                    "Closing Balance": result.get("closing_balance", "–"),
                }

                st.table(pd.DataFrame(summary_fields.items(), columns=["Field", "Value"]))

                # Transactions Table
                st.subheader("📋 Transactions")
                transactions = result.get("transactions", [])
                if transactions:
                    df = pd.DataFrame(transactions)
                    df["money_in"] = df["money_in"].fillna("–")
                    df["money_out"] = df["money_out"].fillna("–")
                    df["balance"] = df["balance"].fillna("–")
                    st.dataframe(
                        df[["date", "description", "money_in", "money_out", "balance"]],
                        use_container_width=True
                    )

                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name="bank_statement_transactions.csv",
                        mime="text/csv"
                    )

                    
                else:
                    st.warning("⚠️ No transactions found.")

                
                    st.subheader("🧾 Raw JSON Output")
                    st.json(result, expanded=False)

        except Exception as e:
            st.error(f"⚠️ Failed to process file: {str(e)}")



    
with tab5:


    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .success-message {
            background-color: #d4edda;
            color: #155724;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #c3e6cb;
        }
        .error-message {
            background-color: #f8d7da;
            color: #721c24;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #f5c6cb;
        }
        .stButton > button {
            width: 100%;
        }
        .record-table th, .record-table td {
            padding: 0.3rem 0.7rem !important;
            font-size: 0.95rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # --- Initialize Session State ---
    def initialize_session_state():
        if 'transactions' not in st.session_state:
            st.session_state.transactions = []
        if 'transaction_counter' not in st.session_state:
            st.session_state.transaction_counter = 0
        if 'confirmed_transactions' not in st.session_state:
            st.session_state.confirmed_transactions = []

    # --- Category Prediction API Call ---
    def get_recommend_category( description , money_in , money_out, api_url=f"{BASE_URL}/recommend-category"):
        #print("get recommend category")
        try:
            response = requests.post(api_url, json={
        
                "description": description ,
                "money_in": money_in,
                "money_out": money_out
            })
            print(f"response: {response.json()}")
            if response.status_code == 200:
                return response.json().get("recommend_category", "")
            else:
                st.warning(f"Prediction API error: {response.status_code} - {response.text}")
                return ""
        except Exception as e:
            st.warning(f"Prediction failed: {e}")
            return ""

    # --- Submit All Confirmed Records ---
    def submit_records():
        if not st.session_state.confirmed_transactions:
            st.error("No records to submit!")
            return False

        for i, transaction in enumerate(st.session_state.confirmed_transactions):
            valid_amount = (
                (transaction['amount_type'] == "Money In" and transaction['money_in'] > 0) or
                (transaction['amount_type'] == "Money Out" and transaction['money_out'] > 0)
            )
            if not all([transaction['description'], valid_amount, transaction['category']]):
                st.error(f"Please fill in all required fields for record {i+1}")
                return False

        with st.spinner('Submitting records...'):
            try:
                time.sleep(2)  # simulate API call
                st.session_state.confirmed_transactions = []
                st.session_state.transactions = []
                st.session_state.transaction_counter = 0
                return True
            except Exception as e:
                st.error(f"Error submitting records: {str(e)}")
                return False

    # --- Main UI Function ---
    def main():
        initialize_session_state()

        st.markdown('<h1 class="main-header">Category Recommendation System</h1>', unsafe_allow_html=True)
        st.markdown("---")

        # Add New Record Button
        col = st.columns([1, 2, 1])[1]
        with col:
            if st.button("➕ Add New Record", key="add_blank_row_btn", use_container_width=True):
                st.session_state.transaction_counter += 1
                st.session_state.transactions.append({
                    'id': st.session_state.transaction_counter,
                    'date': date.today(),
                    'description': '',
                    'amount_type': '',
                    'money_in': 0.0,
                    'money_out': 0.0,
                    'category': '',
                    'predicted_category_value': '',
                    'prediction_input_hash': ''
                })

        st.markdown("---")
        st.markdown("Enter Transaction Records")

        # Editable Grid
        if st.session_state.transactions:
            to_remove = []
            for idx, transaction in enumerate(st.session_state.transactions):
                cols = st.columns([1.5, 2, 2, 2, 0.8])
                with cols[0]:
                    transaction['date'] = st.date_input("Date*", value=transaction['date'], key=f"date_{transaction['id']}")
                with cols[1]:
                    transaction['amount_type'] = st.radio(
                        "Amount Type*", ["Money In", "Money Out"], key=f"amount_type_{transaction['id']}", horizontal=True
                    )
                    if transaction['amount_type'] == "Money In":
                        transaction['money_in'] = st.number_input(
                            "Money In*", value=transaction.get('money_in', 0.0),
                            min_value=0.0, step=0.01, format="%.2f", key=f"money_in_{transaction['id']}"
                        )
                        transaction['money_out'] = 0.0
                    elif transaction['amount_type'] == "Money Out":
                        transaction['money_out'] = st.number_input(
                            "Money Out*", value=transaction.get('money_out', 0.0),
                            min_value=0.0, step=0.01, format="%.2f", key=f"money_out_{transaction['id']}"
                        )
                        transaction['money_in'] = 0.0

                with cols[2]:
                    transaction['description'] = st.text_input(
                        "Description*", value=transaction['description'], key=f"description_{transaction['id']}",
                        placeholder="e.g. Office supplies"
                    )

                with cols[3]:
                    amount_value = transaction['money_in'] if transaction['amount_type'] == "Money In" else transaction['money_out']
                    current_input_data = f"{transaction['date']}-{amount_value}-{transaction['description']}"
                    current_input_hash = hash(current_input_data)
                    #print(transaction['predicted_category_value'])
                    print(f"predicted_category: {transaction['predicted_category_value']}")

                    if (transaction['predicted_category_value'] == '' or
                        transaction['prediction_input_hash'] != current_input_hash):
                        
                        
                        if transaction['date'] and amount_value > 0 and len(transaction['description'].strip()) >0:
                            recommend_category = get_recommend_category(
                                #date=transaction['date'].strftime("%Y-%m-%d"),
                                description=transaction['description'],
                                money_in= transaction['money_in'],
                                money_out=transaction['money_out']
                                
                            )
                            transaction['predicted_category_value'] = recommend_category
                            transaction['prediction_input_hash'] = current_input_hash
                        else:
                            transaction['predicted_category_value'] = ''
                            transaction['prediction_input_hash'] = ''

                    category_options = ['']
                    if transaction['predicted_category_value']:
                        category_options.append(transaction['predicted_category_value'])
                    category_options.append('Other - Enter Manually')

                    if transaction['category'] and transaction['category'] not in category_options and transaction['category'] != 'Other - Enter Manually':
                        category_options.insert(1, transaction['category'])

                    default_index = 0
                    if transaction['category'] in category_options:
                        default_index = category_options.index(transaction['category'])

                    selected_category = st.selectbox(
                        "Category*", options=category_options, index=default_index,
                        key=f"category_select_{transaction['id']}"
                    )

                    if selected_category == 'Other - Enter Manually':
                        transaction['category'] = st.text_input(
                            "Enter Category Name", value=transaction.get('category', ''),
                            key=f"manual_category_{transaction['id']}"
                        )
                    else:
                        transaction['category'] = selected_category

                with cols[4]:
                    st.markdown("<br>", unsafe_allow_html=True)
                    valid_amount = (
                        (transaction['amount_type'] == "Money In" and transaction['money_in'] > 0) or
                        (transaction['amount_type'] == "Money Out" and transaction['money_out'] > 0)
                    )
                    add_disabled = not (
                        transaction['description'] and valid_amount and transaction['category']
                    )
                    if st.button("Add", key=f"add_{transaction['id']}", disabled=add_disabled, use_container_width=True):
                        st.session_state.confirmed_transactions.append(transaction.copy())
                        to_remove.append(transaction['id'])

            # Remove added rows
            if to_remove:
                st.session_state.transactions = [
                    t for t in st.session_state.transactions if t['id'] not in to_remove
                ]
                st.rerun()
                return

        # Confirmed Records Table
        st.markdown("---")
        st.markdown("### ✅ Confirmed Records")

        if st.session_state.confirmed_transactions:
            df = pd.DataFrame([{
                'Date': t['date'],
                'Amount Type': t['amount_type'],
                'Money In': t['money_in'],
                'Money Out': t['money_out'],
                'Description': t['description'],
                'Category': t['category']
            } for t in st.session_state.confirmed_transactions])
            st.dataframe(df, use_container_width=True, height=220)

            total_amount = sum(
                t['money_in'] if t['amount_type'] == "Money In" else -t['money_out']
                for t in st.session_state.confirmed_transactions
            )
            st.markdown(f"<div style='text-align:right; font-size:1.1rem;'><b>Total Amount: ${total_amount:.2f}</b></div>", unsafe_allow_html=True)
            st.markdown("---")

            col1, col2, col3 = st.columns([1, 2, 1])
            all_valid = all([
                t['description'] and t['category'] and (
                    (t['amount_type'] == "Money In" and t['money_in'] > 0) or
                    (t['amount_type'] == "Money Out" and t['money_out'] > 0)
                )
                for t in st.session_state.confirmed_transactions
            ])

            with col2:
                if st.button("🚀 Submit All Records", key="submit_all", type="primary", disabled=not all_valid):
                    if submit_records():
                        st.success("All records submitted successfully!")
                        st.rerun()
        else:
            st.info("No confirmed records yet. Add and confirm a record above.")

    if __name__ == "__main__":
        main()


with tab6:
    st.title("Transaction Category Prediction")

# --- Initialize Session State ---
    if "single_result" not in st.session_state:
       st.session_state.single_result = None

# --- Single Transaction Input Section ---
    with st.form("transaction_form"):
       col1, col2, col3, col4 = st.columns([2, 3, 2, 2])
    
       with col1:
        date = st.text_input("Date (Required)", placeholder="e.g., 2025-07-10")

       with col2:
        description = st.text_input("Description (Required)", placeholder="e.g., Amazon Order #1234")

       with col3:
        flow_type = st.selectbox("Type", ["Money In", "Money Out"])

       with col4:
        amount = st.number_input("Amount", min_value=0.00, step=0.01, format="%.2f")

       submitted = st.form_submit_button("Predict")
       if submitted:
        if not date.strip() or not description.strip():
            st.error("Date and Description are required.")
        elif amount <= 0:
            st.error("Amount must be greater than 0.")
        else:

            transaction = {               
                "Description": description.strip(),
                "Money_In": amount if flow_type == "Money In" else 0.0,
                "Money_Out": amount if flow_type == "Money Out" else 0.0              
                
            }

            try:
                response = requests.post(f"{BASE_URL}/predict", json=transaction)
                if response.status_code == 200:
                    st.session_state.single_result = response.json()
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# --- Display Prediction Results ---
    if st.session_state.single_result:
       result = st.session_state.single_result
       st.subheader("Prediction Result")
       result_df = pd.DataFrame([{
        "Predicted Category": result["predicted_category"],
        "Confidence": f"{result['confidence']:.2%}"
        }])
       st.table(result_df)

       json_result = json.dumps(result, indent=2)
    

with tab7:
    

    st.set_page_config(page_title="Duplicate & Anomaly Detection Dashboard", layout="wide")
    st.title("🧠 Duplicate & Anomaly Detection Dashboard")

    # File uploader
    uploaded_file = st.file_uploader("📤 Upload CSV File", type=["csv"])

    if uploaded_file:
        # Check if a new file has been uploaded
        if 'uploaded_file_name' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.uploaded_file_name = uploaded_file.name
            # Clear previous supplier list to ensure new CSV data is used
            if 'unique_suppliers' in st.session_state:
                del st.session_state.unique_suppliers
            with st.spinner("Processing your file..."):
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_input_path = tmp_file.name

                # Upload to FastAPI backend
                try:
                    response = requests.post(
                        f"{BASE_URL}/upload",
                        files={"file": (uploaded_file.name, open(temp_input_path, "rb"))}
                    )
                    if response.status_code == 200:
                        st.success("File uploaded and processed for duplicate and anomaly detection.")
                        st.session_state.file_uploaded = True
                    else:
                        st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
                        st.session_state.file_uploaded = False
                except Exception as e:
                    st.error(f"Upload failed: {e}")
                    st.session_state.file_uploaded = False
                finally:
                    if os.path.exists(temp_input_path):
                        os.remove(temp_input_path)
        else:
            st.session_state.file_uploaded = True
    else:
        st.session_state.file_uploaded = False
        # Clear supplier list if no file is uploaded
        if 'unique_suppliers' in st.session_state:
            del st.session_state.unique_suppliers

    def display_table(data, title, columns_to_show):
        if data:
            df = pd.DataFrame(data)
            df.columns = [col.lower().replace(" ", "_") for col in df.columns]
            # Remove duplicate columns
            df = df.loc[:, ~df.columns.duplicated()]
            df = df.astype(str)  # Convert to string to avoid ArrowTypeError
            st.subheader(title)
            available_columns = [col for col in columns_to_show if col in df.columns]
            if not available_columns:
                st.warning(f"No valid columns available for {title}. Available columns: {df.columns.tolist()}")
                return
            st.dataframe(df[available_columns])
        else:
            st.info(f"No {title.lower()} found.")

    if st.session_state.get("file_uploaded", False):
        # All Transactions Section
        st.header("📋 All Transactions")
        try:
            res = requests.get(f"{BASE_URL}/transactions")
            res.raise_for_status()
            data = res.json()
            st.markdown(f"**🧾 Total Transactions: {data.get('total_transactions', 0)}**")
            columns_to_show = ['date', 'supplier', 'description', 'money_in', 'money_out']
            display_table(data.get("transactions", []), "All Transactions", columns_to_show)
        except Exception as e:
            st.error(f"Failed to load transactions: {e}")

        # Duplicates Section
        st.header("🔁 Detected Duplicates")
        try:
            res = requests.get(f"{BASE_URL}/duplicates")
            res.raise_for_status()
            data = res.json()
            st.markdown(f"**🔁 Total Duplicates: {data.get('total_duplicates', 0)}**")
            columns_to_show = ['date', 'supplier', 'description', 'money_in', 'money_out']
            display_table(data.get("duplicates_found", []), "Duplicate Transactions", columns_to_show)
        except Exception as e:
            st.error(f"Failed to load duplicates: {e}")

        # Anomalies Section
        st.header("🚨 Detected Anomalies")
        try:
            res = requests.get(f"{BASE_URL}/anomalies")
            res.raise_for_status()
            data = res.json()
            st.markdown(f"**🚨 Total Anomalies: {data.get('total_anomalies', 0)}**")
            columns_to_show = ['date', 'supplier_name', 'description', 'money_in', 'money_out', 'confidence_score']
            display_table(data.get("anomalies", []), "Anomalous Transactions", columns_to_show)
        except Exception as e:
            st.error(f"Failed to load anomalies: {e}")
# Check Single Transaction Section
with tab8:
    st.header("🔍 Check Single Transaction for Anomaly")
    if st.session_state.get("file_uploaded", False):
        try:
            if 'unique_suppliers' not in st.session_state:
                res = requests.get(f"{BASE_URL}/transactions")
                res.raise_for_status()
                data = res.json()
                transactions = pd.DataFrame(data.get("transactions", []))
                st.session_state.unique_suppliers = sorted(transactions.get('supplier', pd.Series([])).dropna().unique())

            unique_suppliers = st.session_state.unique_suppliers

            if not unique_suppliers:
                st.warning("No supplier names available.")
            else:
                with st.form(key="single_transaction_form"):
                    supplier = st.selectbox("Select Supplier", unique_suppliers)
                    date_input = st.text_input("Transaction Date (e.g., DD-MM-YYYY)", help="Enter date in DD-MM-YYYY format.")
                    money_in = st.number_input("Money In", min_value=0.0, step=0.01, format="%.2f")
                    money_out = st.number_input("Money Out", min_value=0.0, step=0.01, format="%.2f")
                    submit_button = st.form_submit_button("Check Anomaly")

                    if submit_button:
                        try:
                            parsed_date = pd.to_datetime(date_input, errors='raise', dayfirst=True)
                            payload = {
                                "date": parsed_date.strftime("%d-%m-%Y"),
                                "supplier_name": supplier,
                                "money_in": money_in,
                                "money_out": money_out
                            }
                            res = requests.post(f"{BASE_URL}/predict_single_transaction_anomaly", json=payload)
                            res.raise_for_status()
                            result = res.json()
                            st.write("### Prediction Result:")
                            st.write(f"- **Is Anomaly**: {'Yes' if result['is_anomaly'] else 'No'}")
                            st.write(f"- **Confidence Score**: {result['confidence_score']:.4f}")
                        except ValueError:
                            st.error("Invalid date format. Please use DD-MM-YYYY.")
                        except Exception as e:
                            st.error(f"Error checking anomaly: {e}")
        except Exception as e:
            st.error(f"Failed to load supplier data: {e}")
    else:
        st.info("Please upload a CSV file first.")