import os
import shutil
from PIL import Image
import google.generativeai as genai
import json
import re
import pytesseract
from pdf2image import convert_from_path
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

# Configure Google API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("GEMINI_API_KEY not set in environment.")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-pro")

def convert_pdf_to_images(pdf_path):
    """
    Converts each page of a PDF into PNG images. Returns a list of image file paths.
    """
    try:
        output_folder = tempfile.mkdtemp(prefix="pdf_images_")
        images = convert_from_path(pdf_path ,  dpi=300)
        image_paths = []

        for i, img in enumerate(images):
            img_path = os.path.join(output_folder, f"page_{i + 1}.png")
            img.convert("RGB").save(img_path, "PNG")
            image_paths.append(img_path)

        return image_paths
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []

def convert_to_images(file_path):
    """
    Handles both PDF and image inputs. Returns list of image paths.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return convert_pdf_to_images(file_path)

    elif ext in [".png", ".jpg", ".jpeg"]:
        try:
            output_folder = tempfile.mkdtemp(prefix="image_input_")
            img = Image.open(file_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_path = os.path.join(output_folder, f"image_1{ext}")
            img.save(img_path)
            return [img_path]
        except Exception as e:
            print(f"[convert_to_images] Error: {e}")
            return []
    else:
        print(f"[convert_to_images] Unsupported file type: {ext}")
        return []

    

def extract_clean_json(llm_output: str) -> dict:
    """
    Parses a JSON object from the LLM's output, handling possible markdown or noise.
    """
    try:
        clean = llm_output.strip()
        # Remove markdown code fences if present
        clean = clean.removeprefix("```json").removesuffix("```").strip()
        clean = clean.removeprefix("```").removesuffix("```").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        # Fallback: extract {...} or [...] from the text
        match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', clean)
        if match:
            return json.loads(match.group(0))
        else:
            raise ValueError(f"Failed to parse JSON from output: {llm_output}")

def process_pdf_for_extraction_with_gemini(file_path: str, prompt: str) -> str:
    """
    Converts a PDF or image to images, uploads them to Gemini, and queries with prompt.
    Returns the raw text response.
    """
    image_paths = convert_to_images(file_path)
    if not image_paths:
        return "Error: No images extracted from the input file."

    inputs = [prompt]
    uploaded_files = []
    try:
        for img_path in image_paths:
            print(f"Uploading {img_path}...")
            uploaded_file = genai.upload_file(img_path)
            uploaded_files.append(uploaded_file)
            inputs.append(uploaded_file)
        response = model.generate_content(
            inputs,
            generation_config={"max_output_tokens": 50000, "temperature": 0.7}
        )
        return response.text
    except Exception as e:
        print(f"Error during Gemini query: {e}")
        return f"Error: {e}"
    finally:
        # Clean up uploaded files and local images
        for uf in uploaded_files:
            try:
                genai.delete_file(uf.name)
            except Exception:
                pass
        for path in image_paths:
            try:
                shutil.rmtree(os.path.dirname(path))
            except Exception:
                pass

def classify_document_with_gemini(file_path: str) -> str:
    """
    Classifies the document as one of 'bank_statement', 'invoice', 'receipt', or 'unknown'.
    Uses OCR keyword search and falls back to the Gemini model.
    """
    image_paths = convert_to_images(file_path)
    if not image_paths:
        return "error: No images extracted for classification."

    # OCR-based keyword detection (for bank statements)
    try:
        full_text = ""
        for img_path in image_paths:
            text = pytesseract.image_to_string(Image.open(img_path))
            full_text += text.lower() + " "
        bank_keywords = ["account number", "transaction date", "narration", "debit", "credit", "available balance", "ifsc"]
        matches = sum(kw in full_text for kw in bank_keywords)
        if matches >= 3:
            return "bank_statement"
    except Exception:
        # If OCR fails, just proceed to model-based classification
        pass

    # Model-based classification prompt
    prompt = (
         "Classify the type of document from the images provided. "
        "Respond with only one of: 'bank_statement', 'invoice', 'receipt', or 'unknown'."
    )
    inputs = [prompt]
    uploaded_files = []
    try:
        for img_path in image_paths:
            print(f"Uploading {img_path} for classification...")
            uf = model.upload_file(img_path)
            uploaded_files.append(uf)
            inputs.append(uf)
        response = model.generate_content(inputs, generation_config={"temperature": 0.3, "max_output_tokens": 50})
        prediction = response.text.strip().lower()
        # Interpret the model output flexibly
        if "bank" in prediction:
            return "bank_statement"
        elif "invoice" in prediction:
            return "invoice"
        elif "receipt" in prediction:
            return "receipt"
        else:
            return "unknown"
    except Exception as e:
        print(f"Classification error: {e}")
        return "error: Classification failed."
    finally:
        for uf in uploaded_files:
            try:
                genai.delete_file(uf.name)
            except Exception:
                pass
        for path in image_paths:
            try:
                shutil.rmtree(os.path.dirname(path))
            except Exception:
                pass

# Example usage:
if __name__ == "__main__":
    file_path = "data/Screenshot 2025-07-16 at 12.40.57 PM.png"  # or .png/.jpg image
    print(f"Classifying document: {file_path}")
    doc_type = classify_document_with_gemini(file_path)
    print(f"Detected document type: {doc_type}")

    if doc_type == "bank_statement":
        extraction_prompt = """
You are an expert financial data extractor. Your task is to process raw visual information from a UK bank statement and output a clean, valid JSON object containing specific financial information and transaction details.

Output Requirements:
Strictly output a single, well-formed JSON object. No other text or markdown. The JSON structure must be exactly as below:

```json
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


```"""
        raw_output = process_pdf_for_extraction_with_gemini(file_path, extraction_prompt)
        print("Raw LLM output:", raw_output[:100])
        try:
            data = extract_clean_json(raw_output)
            print("Parsed JSON:", json.dumps(data, indent=2))
        except ValueError as e:
            print(e)