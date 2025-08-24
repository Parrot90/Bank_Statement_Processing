Kreative Time Box Project
=========================

A modular, ML-powered document intelligence system for extracting, validating, and matching financial data from receipts, invoices, and bank transactions. Designed for real-world business workflows with fraud detection, expense categorization, and human-in-the-loop feedback.

Tech Stack Overview
-------------------

Programming Language:
• Python

Document & Image Handling:
• Camelot – PDF table extraction
• OpenCV – Image preprocessing
• Pillow – Image manipulation
• pdfplumber – Text and layout extraction from PDFs
• PyPDF2 – PDF parsing and merging

Deep Learning & Vision APIs:
• Gemma 3 12B
• LLama
• ollama server
• Gemini API – Vision-based text extraction

API Development:
• FastAPI – RESTful endpoints (/upload, /extract, /validate)
• SwaggerUI – Interactive API documentation
• n8n – Workflow automation
• Python – Core logic for OCR and ML integration

Dashboard & Visualization:
• Pandas – Data manipulation
• matplotlib / seaborn – Visual analytics
• Streamlit – Interactive dashboards

Source Control:
• Git – Version control

Core Modules
------------

1. Backend & APIs
• FastAPI endpoints for document upload, extraction, and validation
• SwaggerUI for API testing
• n8n for backend orchestration
• Python for OCR logic and ML integration

2. Fraud & Duplicate Detection
• Rule-Based System – Flags suspicious patterns (e.g., same amount/supplier)

3. Expense Categorization & Learning
• Models: RandomForest, LogisticRegression, LSTM
• Feature Engineering: n-grams, amount ranges, supplier-based features
• Feedback System: Retraining based on user corrections
• Rule Engine: JSON-based overrides for model predictions

4. Transaction Matching
• ML Similarity Scoring – Binary classification of receipt-bank transaction pairs
• Threshold Logic – Match strength and confidence scoring
• Model Explainability – Category prediction insights
• Streamlit UI – For reviewing and validating matches

5. Frontend / UI
• Streamlit Dashboards – For transaction matching and feedback
• Custom UI Components – Built using standard frontend tools

Getting Started
---------------

1. Clone the repo:
   git clone https://github.com/your-org/kreative-time-box.git
   cd kreative-time-box

2. Install dependencies:
   pip install -r requirements.txt

3. Run API:
   uvicorn api.main:app --reload

4. Launch dashboard:
   streamlit run ui/dashboard.py
