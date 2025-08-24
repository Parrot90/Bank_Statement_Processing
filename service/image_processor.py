import base64
import os
import json
import google.generativeai as genai
import os
from dotenv import load_dotenv
import filetype

load_dotenv()


# --- Gemini Vision wrapper ---

class ImageProcessor:
    def __init__(self, prompt=None):
        # Get API key from environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")
        
        # Configure GenAI
        genai.configure(api_key=api_key)

        # Initialize model
        self.model = genai.GenerativeModel("gemma-3-27b-it")
        self.prompt = prompt or "What is this?"


   
        
    def get_mime_type(self, path):
    # Read the first few bytes to guess type
      with open(path, "rb") as f:
        kind = filetype.guess(f.read(262))  # 262 bytes is enough for most formats

      if kind and kind.mime.startswith("image/"):
        return kind.mime

    # Fallback based on file extension
      ext = os.path.splitext(path)[1].lower()
      return {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg'
    }.get(ext, 'image/jpeg')

    def encode_image(self, path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def classify_image(self, image_path):
        mime = self.get_mime_type(image_path)
        with open(image_path, "rb") as f:
            image_data = f.read()
        classification_prompt = """
        Analyze the image and determine if it is an invoice, a receipt, or neither. Return a JSON object with a single key "document_type" and one of the following values:
        - "invoice": For documents with supplier details, due dates, or structured billing (e.g., business invoices, hotel bookings).
        - "receipt": For documents with store names, transaction numbers, or itemized purchases (e.g., restaurant or retail receipts).
        - "other": For images that are neither invoices nor receipts (e.g., photos of people, landscapes, or random objects).
        Output only the JSON object, nothing else.
        {
          "document_type": "invoice|receipt|other"
        }
        """
        content = [
            {"mime_type": mime, "data": image_data},
            {"text": classification_prompt}
        ]
        try:
            response = self.model.generate_content(content)
            output = response.text
            start_idx = output.find("{")
            end_idx = output.rfind("}") + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No valid JSON found in classification output")
            json_data = json.loads(output[start_idx:end_idx])
            return json_data.get("document_type", "other")
        except Exception as e:
            raise ValueError(f"Classification error: {str(e)}")

    def analyze_image(self, image_path, expected_type):
        # Validate document type
        document_type = self.classify_image(image_path)
        if document_type == "other":
            raise ValueError("Image is neither an invoice nor a receipt")
        if expected_type == "invoice" and document_type != "invoice":
            raise ValueError(f"Image is a {document_type}, but an invoice was expected")
        if expected_type == "receipt" and document_type != "receipt":
            raise ValueError(f"Image is a {document_type}, but a receipt was expected")

        # Set the appropriate prompt based on expected type
        prompt = INVOICE_PROMPT if expected_type == "invoice" else RECEIPT_PROMPT
        mime = self.get_mime_type(image_path)
        with open(image_path, "rb") as f:
            image_data = f.read()
        content = [
            {"mime_type": mime, "data": image_data},
            {"text": prompt}
        ]
        try:
            response = self.model.generate_content(content)
            output = response.text
            start_idx = output.find("{")
            end_idx = output.rfind("}") + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No valid JSON found in output")
            json_str = output[start_idx:end_idx]
            return json.loads(json_str)
        except Exception as e:
            raise ValueError(f"Gemini API error: {str(e)}")

# --- Prompt Templates ---
INVOICE_PROMPT = """
From the raw text of a SINGLE invoice, return a CLEAN, STRICT JSON object with these fields:

{
  "supplier_name": "",
  "invoice_date": "",
  "total_amount": "",
  "tax_amount": "",
  "due_date": null,
  "currency": null,
  "items": []
}

Top-level Fields:
- supplier_name   (string) : Name of the company issuing the invoice.
- invoice_date    (string) : Date of invoice in yyyy-mm-dd format.
- total_amount    (string) : Grand total billed (numbers only, no currency symbols).
- tax_amount      (string) : Total tax/VAT/GST applied (numbers only).
- due_date        (string|null) : Date when payment is due, or null if not found.
- currency        (string|null) : Currency in ISO 4217 format (e.g., "GBP", "USD").
- items           (array of objects) : Line items in the invoice, each with:

Each item object must include:
  - description   (string)
  - quantity      (string|null)
  - unit_price    (string|null)
  - total_price   (string|null)

GUIDELINES:
- Automatically determine invoice *type* (e.g., restaurant, travel agent like Agoda, service contract).
- For *restaurant receipts*, extract:
    - quantity as the number of units ordered (e.g., 2 × Biryani)
    - unit_price and total_price from item-level gross value
- For *travel agent invoices*, extract:
    - quantity as number of nights, rooms, or persons (if specified)
    - unit_price as price per room per night or per person
    - description should include hotel and booking details
- For *contract or service-based invoices*, extract:
    - description as task or service performed
    - quantity and unit_price if available, otherwise set to null

RULES:
- Output STRICTLY VALID JSON. No markdown, explanations, formatting, or extra text.
- Use null for any missing, ambiguous, or unparseable values.
- Date should be same as in shown in file 
- Extract currency from symbols (e.g., £ → "GBP", $ → "USD", € → "EUR"), or set to null if unknown.
- Use layout and line structure to infer correct line items.
- Skip non-item lines like "Subtotal", "Discount", "Tip", etc.
- If no valid line items are found, return an empty array for "items".
"""

RECEIPT_PROMPT = """
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
- Do NOT return anything outside the JSON object.
- All dates must be returned in day–month–year (DD–MM–YYYY) format.
- If a field is not found, return an empty string.
- Ensure JSON is parsable and contains double quotes.
- For items, extract all lines including discounts, totals, offers etc. Each item should include fields: name, description, price, unit_price, quantity, discount, total, line_total (use empty strings if not found).
- Use the exact field names and order as shown above.
- If a value is missing, use an empty string ("") or an empty list for items.
- Only return the JSON object, nothing else.
- Ensure the JSON is valid and parsable.
"""