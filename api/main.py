from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
import io
from service.extraction import process_invoice_document, process_receipt_document
from service.image_processor import ImageProcessor , INVOICE_PROMPT, RECEIPT_PROMPT
from service.recommendation_service import CategoryPredictionService
from service.rules_based import load_rules, match_rule,apply_rules_to_dataset  , predict_categorie
from models.schema import PredictionResponse, PredictionRequest  , TransactionInput 
from typing import Optional
from starlette.concurrency import run_in_threadpool
from service.anomaly_service import full_processing_pipeline, preprocess, predict_anomalies, COLUMN_MAPPINGS
from service.duplicate_service import DuplicateDetectionService
import numpy as np
from models.schema import Transaction
import joblib
import pandas as pd
import tempfile
import os
import logging
import shutil
import json
import re
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from service.transaction_scorer import (
    load_inference_components,
    predict_transaction_category,
    
)
from typing import List, Dict



from service.bank_statement import (
    classify_document_with_gemini,
    process_pdf_for_extraction_with_gemini,
    extract_clean_json,
)

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

recommendation_service = CategoryPredictionService()
dup_service = None
anomaly_df = None
supplier_freq_map = None



@app.post("/process-bank-statement/")
async def process_bank_document(file: UploadFile = File(...)):
    
    try:
        # Save uploaded file temporarily
        filename = file.filename or ""
        ext = os.path.splitext(filename)[-1].lower()
        if ext not in [".pdf", ".png", ".jpg", ".jpeg"]:
            return JSONResponse(status_code=400, content={"error": "Unsupported file type."})

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Step 1: Classification
        '''doc_type = classify_document_with_gemini(tmp_path)
        if doc_type != "bank_statement":
            os.remove(tmp_path)
            return JSONResponse(status_code=400, content={"error": "Document is not a bank statement."})'''

        # Step 2: Extraction using Gemini
        extraction_prompt = """
You are an expert financial data extractor. Your task is to process raw visual information from a UK bank statement and output a clean, valid JSON object containing specific financial information and transaction details.

Output Requirements:
Strictly output a single, well-formed JSON object. No other text, markdown formatting (like triple backticks for code blocks), comments, or explanations should be included in your final response.

JSON Structure:
{
  "account_holder_name": "string",
  "account_number": "string",
  "bank_name": "string",
  "statement_period": "string",
  "opening_balance": "string",
  "closing_balance": "string",
  "currency": "string",
  "transactions": [
    {
      "date": "string",
      "description": "string",
      "money_in": "string|null",
      "money_out": "string|null",
      "balance": "string|null"
    }
  ]
}

note - All dates must be returned in day–month–year (DD–MM–YYYY) format.
"""
        llm_output = process_pdf_for_extraction_with_gemini(tmp_path, extraction_prompt)
        structured_data = extract_clean_json(llm_output)

        os.remove(tmp_path)
        return JSONResponse(content=structured_data)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    

logger = logging.getLogger("gemma3")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)



@app.post("/parse-invoice/")
async def parse_invoice(file: UploadFile = File(...)):
    try:
        if not file.filename:
            return JSONResponse(content={"error": "No filename provided."}, status_code=400)
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = process_invoice_document(pdf_path=file_path)
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/parse-receipt/")
async def parse_receipt(file: UploadFile = File(...)):
    try:
        if not file.filename:
            return JSONResponse(content={"error": "No filename provided."}, status_code=400)
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = process_receipt_document(pdf_path=file_path)
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    


GEMINI_API_KEY = "AIzaSyAVa8rPSAh8WFQXGbX8sUmSQNGY17vuD4Q"
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

@app.post("/api/invoice")
async def process_invoice(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        logger.error("Invalid file type: %s", file.content_type)
        raise HTTPException(status_code=400, detail="File must be an image (png, jpg, jpeg)")

    temp_path = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            content = await file.read()
            if not content:
                logger.error("Empty file uploaded")
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            tmp.write(content)
            temp_path = tmp.name
        logger.info("Temporary file created: %s", temp_path)

        # Process image with Gemini API
        processor = ImageProcessor(prompt=INVOICE_PROMPT)
        logger.info("Validating image as invoice")
        result = processor.analyze_image(temp_path, expected_type="invoice")
        logger.info("Image processing completed")

        # Ensure all required fields are present
        required_fields = {
            "supplier_name": "",
            "invoice_date": "",
            "total_amount": "",
            "tax_amount": "",
            "due_date": None,
            "currency": None,
            "items": []
        }
        for field in required_fields:
            if field not in result:
                logger.warning("Missing field %s in result, setting default", field)
                result[field] = required_fields[field]

        # Ensure items have all required subfields
        item_fields = ["description", "quantity", "unit_price", "total_price"]
        for item in result.get("items", []):
            for field in item_fields:
                if field not in item:
                    logger.warning("Missing item field %s, setting to None", field)
                    item[field] = None

        return result
    except ValueError as ve:
        logger.error("Validation error: %s", str(ve))
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(ve)}")
    except Exception as e:
        logger.error("Unexpected error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing invoice: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.info("Temporary file deleted: %s", temp_path)
            except Exception as e:
                logger.error("Error deleting temporary file: %s", str(e))

@app.post("/api/receipt")
async def process_receipt(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        logger.error("Invalid file type: %s", file.content_type)
        raise HTTPException(status_code=400, detail="File must be an image (png, jpg, jpeg)")

    temp_path = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            content = await file.read()
            if not content:
                logger.error("Empty file uploaded")
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            tmp.write(content)
            temp_path = tmp.name
        logger.info("Temporary file created: %s", temp_path)

        # Process image with Gemini API
        processor = ImageProcessor(prompt=RECEIPT_PROMPT)
        logger.info("Validating image as receipt")
        result = processor.analyze_image(temp_path, expected_type="receipt")
        logger.info("Image processing completed")

        # Ensure all required fields are present
        required_fields = {
            "store_name": "",
            "date": "",
            "currency": "",
            "total_amount": "",
            "tax_details": "",
            "transaction_number": "",
            "card_details": "",
            "service_fee": "",
            "items": []
        }
        for field in required_fields:
            if field not in result:
                logger.warning("Missing field %s in result, setting default", field)
                result[field] = required_fields[field]

        # Ensure items have all required subfields
        item_fields = ["name", "description", "price", "unit_price", "quantity", "discount", "total", "line_total"]
        for item in result.get("items", []):
            for field in item_fields:
                if field not in item:
                    logger.warning("Missing item field %s, setting to empty string", field)
                    item[field] = ""

        return result
    except ValueError as ve:
        logger.error("Validation error: %s", str(ve))
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(ve)}")
    except Exception as e:
        logger.error("Unexpected error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing receipt: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.info("Temporary file deleted: %s", temp_path)
            except Exception as e:
                logger.error("Error deleting temporary file: %s", str(e))

@app.options("/api/invoice")
async def options_invoice():
    return {}

@app.options("/api/receipt")
async def options_receipt():
    return {}


@app.post("/recommend-category", response_model=PredictionResponse)
def recommend_category(request: PredictionRequest):
    #print("calling service")
    try:
        category = recommendation_service.recommend_category(
            
            description=request.description,
            money_in=request.money_in,
            money_out=request.money_out
        )
        print(f"Predicted category: {category}")
        if category.startswith("Prediction failed"):
            raise HTTPException(status_code=400, detail=category)
        return {"recommend_category": category}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rules")
def get_category_by_rule(input_data: TransactionInput):
    """Predict category for a single transaction"""
    try:
        # Convert to dict - compatible with both Pydantic v1 and v2
        try:
            transaction_dict = input_data.model_dump()
        except AttributeError:
            transaction_dict = input_data.dict()
        
        result = predict_categorie(transaction_dict)
        
        # Return only category and rule_applied
        return {
            "category": result["category"],
            "rule_applied": result["rule_applied"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        

@app.get("/getCategoryByRule")
def get_all_predictions():
    """Get all matched transactions from dataset"""
    try:
        results = apply_rules_to_dataset()
        
        if isinstance(results, dict) and "error" in results:
            raise HTTPException(status_code=404, detail=results["error"])
        
        return {
            "total_matched": len(results),
            "matched_transactions": results,
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/predict")
async def predict(transaction: Transaction):
    try:
        result = predict_transaction_category(transaction.Description, transaction.Money_In, transaction.Money_Out)
        #print(f"Predicted category: {result}")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Initialize models on startup
load_inference_components()



@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global dup_service, anomaly_df, supplier_freq_map
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    # Save uploaded CSV to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        temp_input_path = tmp_file.name

    try:
        # Validate CSV columns
        df = pd.read_csv(temp_input_path)
        df_columns_lower = [col.lower().strip() for col in df.columns]
        required_cols = ["date", "money_in", "money_out"]
        matched_cols = [
            any(col in df_columns_lower for col in [name.lower() for name in COLUMN_MAPPINGS[req_col]])
            for req_col in required_cols
        ]
        if not all(matched_cols):
            missing = [req_col for req_col, matched in zip(required_cols, matched_cols) if not matched]
            raise HTTPException(status_code=400, detail=f"CSV missing required columns: {missing}")

        # Process for duplicate detection
        dup_service = await run_in_threadpool(DuplicateDetectionService, temp_input_path)
        print("Duplicate service initialized. Columns:", dup_service.df.columns.tolist())

        # Clean up any CSV files generated by duplicate service
        for file in ["all_transactions.csv", "all_duplicates.csv"]:
            if os.path.exists(file):
                os.remove(file)
                print(f"Removed {file}")

        # Process for anomaly detection using original CSV
        anomaly_df = await run_in_threadpool(full_processing_pipeline, temp_input_path)
        if anomaly_df is None:
            raise HTTPException(status_code=500, detail="Failed to process CSV for anomaly detection.")
        supplier_freq_map = anomaly_df['supplier_name'].value_counts().to_dict()
        print("Anomaly detection completed. Columns:", anomaly_df.columns.tolist())
        
        return {"message": "File uploaded and processed for duplicate and anomaly detection."}
    except Exception as e:
        import traceback
        print("Upload error traceback:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    finally:
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)

@app.get("/transactions")
async def get_all_transactions():
    if dup_service is None:
        raise HTTPException(status_code=400, detail="Please upload a CSV file first.")
    try:
        transactions = dup_service.df.copy()
        print("Transactions fetched. Columns:", transactions.columns.tolist())
        print("Sample data:", transactions.head(2).to_dict())
        transactions.replace([np.inf, -np.inf], np.nan, inplace=True)
        transactions.fillna(0, inplace=True)
        return {
            "transactions": transactions.to_dict(orient="records"),
            "total_transactions": len(transactions),
            "total_duplicates": int(transactions["IsDuplicate"].sum())
        }
    except Exception as e:
        print(f"Error in /transactions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch transactions: {str(e)}")

@app.get("/duplicates")
async def get_duplicates():
    if dup_service is None:
        raise HTTPException(status_code=400, detail="Please upload a CSV file first.")
    try:
        dup_service.detect_duplicates()
        duplicates = dup_service.get_duplicate_details()
        print("Duplicates fetched:", len(duplicates))
        return {"duplicates_found": duplicates, "total_duplicates": len(duplicates)}
    except Exception as e:
        print(f"Error in /duplicates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch duplicates: {str(e)}")

@app.get("/anomalies")
async def get_anomalies():
    if anomaly_df is None:
        raise HTTPException(status_code=400, detail="Please upload a CSV file first.")
    try:
        anomalies_df = anomaly_df[anomaly_df['is_anomaly'] == 1].fillna('')
        
        # Ensure 'original_date' and 'description' columns are present
        if 'original_date' in anomalies_df.columns:
            anomalies_df = anomalies_df.rename(columns={'original_date': 'date'})
        if 'description' not in anomalies_df.columns:
            anomalies_df['description'] = ''  # Fallback if missing
        
        required_columns = [
            'date',             # From 'original_date'
            'supplier_name',
            'description',      # Ensure narration/description exists
            'money_in',
            'money_out',
            'confidence_score'
        ]
        
        # Filter to required columns
        anomalies_filtered = anomalies_df[required_columns].to_dict(orient='records')
        print("Anomalies fetched:", len(anomalies_filtered))
        
        return {
            "anomalies": anomalies_filtered,
            "total_anomalies": len(anomalies_filtered)
        }
    except Exception as e:
        print(f"Error in /anomalies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch anomalies: {str(e)}")


@app.post("/predict_single_transaction_anomaly")
async def predict_single_transaction_anomaly(transaction: TransactionInput):
    if anomaly_df is None or supplier_freq_map is None:
        raise HTTPException(status_code=400, detail="Please upload a CSV file first.")
    try:
        parsed_date = pd.to_datetime(transaction.date, errors='raise', dayfirst=True)
        supplier_frq = supplier_freq_map.get(transaction.supplier_name, 1)
        temp_data = [{
            COLUMN_MAPPINGS["date"][0]: parsed_date.strftime("%Y-%m-%d"),
            COLUMN_MAPPINGS["narration"][0]: transaction.supplier_name,
            COLUMN_MAPPINGS["money_in"][0]: transaction.money_in,
            COLUMN_MAPPINGS["money_out"][0]: transaction.money_out,
            'supplier_name': transaction.supplier_name,
            'supplier_frq': supplier_frq
        }]
        temp_df = pd.DataFrame(temp_data)
        temp_df_preprocessed = preprocess(temp_df.copy())
        if temp_df_preprocessed is None or temp_df_preprocessed.empty:
            raise HTTPException(status_code=500, detail="Error preprocessing transaction.")
        temp_df_preprocessed['amount'] = temp_df_preprocessed.apply(lambda x: x['money_in'] if x['money_in'] > 0 else x['money_out'], axis=1)
        temp_df_preprocessed['is_credit'] = temp_df_preprocessed['money_in'].apply(lambda x: 1 if x > 0 else 0)
        result_single = predict_anomalies(temp_df_preprocessed)
        if result_single is None or result_single.empty:
            raise HTTPException(status_code=500, detail="Prediction failed.")
        return {
            "is_anomaly": bool(result_single['is_anomaly'].iloc[0]),
            "confidence_score": float(result_single['confidence_score'].iloc[0])
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Please use DD-MM-YYYY.")
    except Exception as e:
        print(f"Error in /predict_single_transaction_anomaly: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error predicting anomaly: {str(e)}")
