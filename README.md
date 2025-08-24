Kreative Time Box Project Tech Stack
________________________________________
Programming Language:
•	Python
________________________________________
Document & Image Handling:
•	Camelot
•	OpenCV
•	Pillow
•	pdfplumber
•	pypdf2
Used for PDF table extraction, image processing, and handling document I/O operations.
________________________________________
Deep Learning/Layout Models/Preprocessing/CNN:
•	Gemma 3 12B
•	LLama
•	ollama server
•	Gemini API
API for Text Extraction Using advance Vision Models
________________________________________
API Development:
•	FastAPI
•	SwaggerUI
API & Response
________________________________________
Dashboard / Visualization:
•	Pandas
•	matplotlib
•	seaborn
•	streamlit

________________________________________
Source Control:
•	Git
________________________________________
2. Backend & APIs
•	FastAPI – Backend framework for building endpoints like /upload, /extract, /validate, etc.
•	Swagger – Used for API documentation.
•	Python – Core language for OCR logic, PDF handling, and ML integration.
•	n8n – Workflow automation tool for backend or API integration.
•	Git – Version control for source code.
________________________________________
3. Fraud & Duplicate Detection
•	Rule-Based System – Flags suspicious patterns like same amount/supplier.
________________________________________
4. Expense Categorization & Learning
•	RandomForest / LogisticRegression / LSTM – Used to classify transaction categories based on supplier, amount, and description.
•	Feature Engineering – Includes n-grams, amount ranges, and supplier-based feature
•	Feedback System – Enables retraining based on corrected predictions.
•	Rule Engine (JSON-based) – Allows overriding model predictions with user-defined rules.
________________________________________
5. Transaction Matching
•	ML Similarity Scoring – Binary classification model trained on receipt and bank transaction pairs.
•	Threshold Logic + Confidence Score – Assigns match strength and flags uncertain matches.
•	Model Explanability : Category Prediction
•	Streamlit 
________________________________________
6. Frontend/UI
•	Streamlit – Used for dashboard visualization in transaction matching.
•	Custom UI  – Developed using standard frontend tools to support user interaction.


