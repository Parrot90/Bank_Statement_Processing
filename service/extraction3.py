# service.py
import json
from typing import Dict
from PyPDF2 import PdfReader
import pdfplumber
from PIL import Image
import pytesseract
import ollama


def extract_text_from_invoice_pdf(pdf_path: str) -> str:
    """
    Attempts to extract text from a PDF.
    Tries PyPDF2 first for selectable text, then pdfplumber for better layout handling.
    If both fail or return empty, suggests OCR.
    """
    text_content = ""

    # Try PyPDF2 for basic text extraction
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                extracted_page_text = page.extract_text()
                if extracted_page_text:
                    text_content += extracted_page_text + "\n"
        if text_content.strip():
            print(f"Successfully extracted text from '{pdf_path}' using PyPDF2.")
            return text_content.strip()
    except Exception as e:
        print(f"PyPDF2 failed for '{pdf_path}': {e}")

    # If PyPDF2 didn't get much, try pdfplumber (better for complex layouts)
    text_content = "" # Reset
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_page_text = page.extract_text()
                if extracted_page_text:
                    text_content += extracted_page_text + "\n"
        if text_content.strip():
            print(f"Successfully extracted text from '{pdf_path}' using pdfplumber.")
            return text_content.strip()
    except Exception as e:
        print(f"pdfplumber failed for '{pdf_path}': {e}")

    # If neither found text, it's likely a scanned PDF or a complex image-based PDF
    print(f"Could not extract sufficient text from '{pdf_path}' using standard methods.")
    print("This PDF might be image-based (scanned). Consider using OCR.")
    print("You will need to convert PDF pages to images and then use `extract_text_from_image`.")
    return ""



def extract_text_from_scanned_invoice_pdf_with_ocr(pdf_path: str) -> str:
    """
    Extracts text from a scanned PDF (image-based) using pytesseract (OCR).
    Requires Poppler (for `pdf2image`) and Tesseract installed on your system.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("`pdf2image` not installed. Please install it: `pip install pdf2image`")
        print("Also, `Poppler` (a dependency of pdf2image) needs to be installed on your system.")
        print("On macOS with Homebrew: `brew install poppler`")
        return ""

    print(f"Attempting OCR on '{pdf_path}' (this may take a while for large PDFs)...")
    text_content = ""
    try:
        # Convert PDF pages to images
        images = convert_from_path(pdf_path) # You might add dpi=300 for better quality

        for i, image in enumerate(images):
            print(f"  Processing page {i+1} with OCR...")
            page_text = pytesseract.image_to_string(image)
            text_content += page_text + "\n"

        if text_content.strip():
            print(f"Successfully extracted text from '{pdf_path}' using OCR.")
            return text_content.strip()
        else:
            print(f"OCR extracted no text from '{pdf_path}'. The image quality might be too low.")
            return ""
    except Exception as e:
        print(f"Error during OCR extraction from '{pdf_path}': {e}")
        return ""
    
def detect_document_type(text: str, model_name: str = "gemma3:12b") -> str:
    """
    Detects whether the input text is from an invoice, a receipt, or unknown.
    Returns: 'invoice', 'receipt', or 'unknown'
    """
    try:
        system_prompt = """
You are a document classification AI.
Given the raw OCR or extracted text of a document, identify if it is an 'invoice', 'receipt', or something else.

RULES:
- Return only one of the following lowercase words: 'invoice', 'receipt', or 'unknown'
- Do not include any other text or explanation.
- If the text contains total amounts, line items, supplier name, invoice number, etc. → it's likely an invoice.
- If it contains transaction number, store address, card details, total + tax, etc. → it's likely a receipt.
- If it contains account numbers, opening/closing balances, transaction summaries, or IFSC codes → it's likely a bank statement. In this case, return 'unknown'.
- Otherwise, return 'unknown'.
"""

        user_prompt = f"""
Classify the following document text:
###
{text}
###
Return only: 'invoice', 'receipt', or 'unknown'
"""

        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.1},
            stream=False
        )

        classification = response["message"]["content"].strip().lower()
        if classification in ["invoice", "receipt"]:
            return classification
        return "unknown"

    except Exception as e:
        print(f"⚠️ Document type detection failed: {e}")
        return "unknown"

def clean_llm_json(text: str) -> str:
    """
    Attempts to extract the valid JSON substring from the model's output.
    You can also use a regex fallback if necessary.
    """
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return text[start:end]
    except ValueError:
        return text  # fallback to raw if brackets not found

def extract_structured_data_with_gemma(receipt_text: str, model_name: str = "gemma3:12b", debug: bool = False) -> Dict:
    if not receipt_text.strip():
        raise ValueError("No receipt text provided for extraction.")

    system_prompt = """ From the raw text of a SINGLE invoice, return a CLEAN, STRICT JSON object with these fields:

Top-level Fields:
- supplier_name   (string) : Name of the company issuing the invoice.
- invoice_date    (string) : Date of invoice in yyyy-mm-dd format.
- total_amount    (string) : Grand total billed (numbers only, no currency symbols).
- tax_amount      (string) : Total tax/VAT/GST applied (numbers only).
- due_date        (string|null) : Date when payment is due, or null if not found.
- currency        (string|null) : Currency in ISO 4217 format (e.g., "GBP", "USD").
- items           (array of objects) : Line items in the invoice, each with:

Each `item` object must include:
  - description   (string)
  - quantity      (string|null)
  - unit_price    (string|null)
  - total_price   (string|null)

GUIDELINES:
- Automatically determine invoice **type** (e.g., restaurant, travel agent like Agoda, service contract).
- For **restaurant receipts**, extract:
    - `quantity` as the number of units ordered (e.g., 2 × Biryani)
    - `unit_price` and `total_price` from item-level gross value
- For **travel agent invoices**, extract:
    - `quantity` as number of nights, rooms, or persons (if specified)
    - `unit_price` as price per room per night or per person
    - `description` should include hotel and booking details
- For **contract or service-based invoices**, extract:
    - `description` as task or service performed
    - `quantity` and `unit_price` if available, otherwise set to null

RULES:
- Output STRICTLY VALID JSON. No markdown, explanations, formatting, or extra text.
- Use `null` for any missing, ambiguous, or unparseable values.
- Accept and normalize flexible date formats (e.g., "9 Sept 2014", "18/08/14").
- Extract currency from symbols (e.g., £ → "GBP", $ → "USD", € → "EUR"), or set to null if unknown.
- Use layout and line structure to infer correct line items.
- Skip non-item lines like "Subtotal", "Discount", "Tip", etc.
-Date should be in the same format as shown in file .
- If no valid line items are found, return an empty array for `"items"`."""


    user_prompt = f"""
Extract structured data from the following content:
###
{receipt_text}
###
Return only the JSON object.
"""

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": 0.1,
                "top_p": 0.9,
                "num_ctx": 8192
            },
            stream=False
        )

        raw_output = response["message"]["content"].strip()
        json_text = clean_llm_json(raw_output)

        #if debug:
            #print("🔍 Raw output:\n", raw_output)
           # print("✅ Extracted JSON:\n", json_text)

        return json.loads(json_text)

    except json.JSONDecodeError as e:
        print("❌ JSON decoding failed:", e)
        if debug:
            print("🔍 Raw model output:\n", raw_output)
        return {}

    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")
        return {}
    


def extract_text_from_receipt_pdf(pdf_path: str) -> str:
    """
    Attempts to extract text from a PDF.
    Tries PyPDF2 first for selectable text, then pdfplumber for better layout handling.
    If both fail or return empty, suggests OCR.
    """
    text_content = ""

    # Try PyPDF2 for basic text extraction
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                extracted_page_text = page.extract_text()
                if extracted_page_text:
                    text_content += extracted_page_text + "\n"
        if text_content.strip():
            print(f"Successfully extracted text from '{pdf_path}' using PyPDF2.")
            return text_content.strip()
    except Exception as e:
        print(f"PyPDF2 failed for '{pdf_path}': {e}")

    # If PyPDF2 didn't get much, try pdfplumber (better for complex layouts)
    text_content = "" # Reset
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_page_text = page.extract_text()
                if extracted_page_text:
                    text_content += extracted_page_text + "\n"
        if text_content.strip():
            print(f"Successfully extracted text from '{pdf_path}' using pdfplumber.")
            return text_content.strip()
    except Exception as e:
        print(f"pdfplumber failed for '{pdf_path}': {e}")

    # If neither found text, it's likely a scanned PDF or a complex image-based PDF
    print(f"Could not extract sufficient text from '{pdf_path}' using standard methods.")
    print("This PDF might be image-based (scanned). Consider using OCR.")
    print("You will need to convert PDF pages to images and then use `extract_text_from_image`.")
    return ""



def extract_text_from_scanned_receipt_pdf_with_ocr(pdf_path: str) -> str:
    """
    Extracts text from a scanned PDF (image-based) using pytesseract (OCR).
    Requires Poppler (for `pdf2image`) and Tesseract installed on your system.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("`pdf2image` not installed. Please install it: `pip install pdf2image`")
        print("Also, `Poppler` (a dependency of pdf2image) needs to be installed on your system.")
        print("On macOS with Homebrew: `brew install poppler`")
        return ""

    print(f"Attempting OCR on '{pdf_path}' (this may take a while for large PDFs)...")
    text_content = ""
    try:
        # Convert PDF pages to images
        images = convert_from_path(pdf_path) # You might add dpi=300 for better quality

        for i, image in enumerate(images):
            print(f"  Processing page {i+1} with OCR...")
            page_text = pytesseract.image_to_string(image)
            text_content += page_text + "\n"

        if text_content.strip():
            print(f"Successfully extracted text from '{pdf_path}' using OCR.")
            return text_content.strip()
        else:
            print(f"OCR extracted no text from '{pdf_path}'. The image quality might be too low.")
            return ""
    except Exception as e:
        print(f"Error during OCR extraction from '{pdf_path}': {e}")
        return ""



def extract_receipts_data_with_gemma(receipt_text: str, model_name: str = "gemma3:12b", debug: bool = False) -> Dict:
    if not receipt_text.strip():
        raise ValueError("No receipt text provided for extraction.")

    system_prompt = """
You are an expert at extracting structured information from receipts. Parse the following fields and return a valid JSON only:

{
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

Instructions:
•⁠  ⁠Do NOT return anything outside the JSON object.
•⁠  ⁠If a field is not found, return an empty string.
•⁠  ⁠Ensure JSON is parsable and contains double quotes.
•⁠  ⁠For items, extract all lines including discounts, totals, offers etc. Each item should include fields: name, description, price, unit_price, quantity, discount, total, line_total (use empty strings if not found).
•⁠  ⁠Use the exact field names and order as shown above.
•⁠  ⁠If a value is missing, use an empty string ("") or an empty list for items.
•⁠  ⁠Only return the JSON object, nothing else.
•⁠  ⁠Ensure the JSON is valid and parsable.
.  Date should be in same fromat as seen in the file

"""
    user_prompt = f"""
Extract structured data from the following content:
###
{receipt_text}
###
Return only the JSON object.
"""

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": 0.1,
                "top_p": 0.9,
                "num_ctx": 8192
            },
            stream=False
        )

        raw_output = response["message"]["content"].strip()
        json_text = clean_llm_json(raw_output)

        '''if debug:
            print("🔍 Raw output:\n", raw_output)
            print("✅ Extracted JSON:\n", json_text)'''

        return json.loads(json_text)

    except json.JSONDecodeError as e:
        print("❌ JSON decoding failed:", e)
        if debug:
            print("🔍 Raw model output:\n", raw_output)
        return {}

    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")
        return {}
    

 



def process_invoice_document(pdf_path: str, model_name: str = "gemma3:12b", debug: bool = False) -> Dict:
    """
    Extracts and processes a document, confirming it's an invoice before extraction.
    Returns structured data as a dictionary or an error dict.
    """
    # Step 1: Extract text (with fallback to OCR)
    text = extract_text_from_invoice_pdf(pdf_path)
    if not text:
        text = extract_text_from_scanned_invoice_pdf_with_ocr(pdf_path)
    if not text:
        return {"error": "❌ Could not extract text from PDF."}

    # Step 2: Detect document type
    doc_type = detect_document_type(text, model_name=model_name)
    if debug:
        print(f"📄 Detected Document Type: {doc_type}")

    # Step 3: Extract structured data if it's an invoice
    if doc_type == "invoice":
        data = extract_structured_data_with_gemma(text, model_name=model_name, debug=debug)
        return data
    else:
        return {"error": "❌ Document is not an invoice. Kindly provide Proper Invoice."}
    

def process_receipt_document(pdf_path: str, model_name: str = "gemma3:12b", debug: bool = False) -> Dict:
    """
    Extracts and processes a document, confirming it's an invoice before extraction.
    Returns structured data as a dictionary or an error dict.
    """
    # Step 1: Extract text (with fallback to OCR)
    text = extract_text_from_receipt_pdf(pdf_path)
    if not text:
        text = extract_text_from_scanned_receipt_pdf_with_ocr(pdf_path)
    if not text:
        return {"error": "❌ Could not extract text from PDF."}

    # Step 2: Detect document type
    doc_type = detect_document_type(text, model_name=model_name)
    if debug:
        print(f"📄 Detected Document Type: {doc_type}")

    # Step 3: Extract structured data if it's an invoice
    if doc_type == "receipt":
        data = extract_receipts_data_with_gemma(text, model_name=model_name, debug=debug)
        return data
    else:
        return {"error": "❌ Document is not an Receipt . kindly upload proper Receipt."}



