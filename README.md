# AI-Powered Fake Job Detection System

A student-project MVP built with Flask, pandas, scikit-learn, TF-IDF, and Logistic Regression. The website analyzes a job listing and shows a baseline classification, a demo risk indicator, and warning signals.

## Important limitations

- `data/sample_jobs.csv` contains **100 handcrafted demonstration rows** created for this project, not a verified real-world dataset.
- The model is a real trained scikit-learn pipeline, but performance on this small sample is **not evidence of real-world accuracy**. Similar phrasing between rows may make metrics look better than they would on unseen real listings.
- The displayed risk indicator is not a calibrated probability and is not proof that a job is fraudulent or safe.
- The app does not connect to live company registries, URL reputation services, or job boards.
- Do not enter passwords, OTPs, bank details, or private personal data.

## Features

- Responsive dark-themed web UI
- Job title, company, salary, website, and description inputs
- TF-IDF + Logistic Regression classifier
- Evaluation metrics printed when the model is trained
- Basic warning-signal explanations (heuristic checks, separate from ML model)
- Optional OCR upload for job screenshots/photos (PNG, JPG, JPEG, WEBP; up to 8 MB)
- Safe-use disclaimer and `/health` endpoint

## Requirements

- Python 3.10+ recommended
- Internet connection for installing packages
- VS Code (recommended, optional)

## Run on Windows

1. Extract the ZIP.
2. Open the extracted `fake_job_detection_project` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. (Optional but recommended) create a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt or run commands without activating the environment.
5. Install the Tesseract OCR application (the Python package alone is not enough). On Windows, install Tesseract OCR and ensure `tesseract.exe` is on PATH. The app also checks the common location `C:\Program Files\Tesseract-OCR\tesseract.exe`.
6. Install Python packages:

   ```powershell
   py -m pip install -r requirements.txt
   ```
7. Train the model and see evaluation metrics:

   ```powershell
   py train_model.py
   ```
8. Start the website:

   ```powershell
   py app.py
   ```
9. Open `http://127.0.0.1:5000` in your browser. Keep the terminal open while testing.

If `py` does not work, replace `py` with `python`.

## Project structure

```text
fake_job_detection_project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── sample_jobs.csv
├── models/
│   └── (model file generated after training)
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Dataset format for future improvement

The CSV currently uses these columns:

- `title`
- `company_profile`
- `description`
- `requirements`
- `benefits`
- `fraudulent` (`0` = genuine, `1` = fraudulent)

For a stronger academic result, replace or supplement the demo CSV with a properly licensed, real labelled dataset such as EMSCAD/Employment Scam Aegean Dataset, confirm its usage terms, remove duplicates and leakage, split by source/time where possible, and evaluate precision, recall, F1, confusion matrix, and false-positive/false-negative cases. Do not claim a target accuracy before testing.

## Next improvements

1. Add a real labelled dataset and reproducible cleaning notebook.
2. Add cross-validation and confusion-matrix visualization.
3. Store user-submitted reports in SQLite with privacy safeguards.
4. Add admin authentication and input validation before deployment.
5. Deploy only after security review; debug mode is for local development only.


## Using OCR image upload

1. Install the Tesseract OCR application for Windows and make sure `tesseract.exe` is on PATH (or installed at the common path checked by the app).
2. Install project Python requirements with `py -m pip install -r requirements.txt`.
3. Start the app with `py app.py`.
4. On the homepage, upload a clear screenshot/photo of a job posting (PNG/JPG/JPEG/WEBP, maximum 8 MB).
5. The app extracts the visible text with OCR and sends it to the existing TF-IDF + Logistic Regression model. The result panel shows the extracted text so it can be reviewed.

OCR can make mistakes on blurry, low-resolution, rotated, handwritten, or low-contrast images. Always review extracted text and verify jobs independently. This demo processes uploaded images in memory and does not save them to disk.
