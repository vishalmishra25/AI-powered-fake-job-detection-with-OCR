from pathlib import Path
from io import BytesIO
import pickle
import re

from flask import Flask, render_template, request, jsonify
from PIL import Image, UnidentifiedImageError
import pytesseract

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fake_job_model.pkl"

app = Flask(__name__)
# Limit uploads to 8 MB to keep this academic demo safe and responsive.
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# On Windows, use the common Tesseract installation path if it is not on PATH.
if not __import__("shutil").which("tesseract"):
    common_tesseract = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if common_tesseract.exists():
        pytesseract.pytesseract.tesseract_cmd = str(common_tesseract)


def ensure_model():
    """Train a baseline model on the bundled demo dataset if none exists."""
    if not MODEL_PATH.exists():
        from train_model import train_model
        train_model()


def extract_text_from_image(upload):
    """Extract job-posting text from a user-uploaded screenshot/photo."""
    filename = (upload.filename or "").strip()
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError("Please upload a PNG, JPG, JPEG, or WEBP image.")

    try:
        image = Image.open(upload.stream)
        image.verify()
        upload.stream.seek(0)
        image = Image.open(upload.stream).convert("RGB")
        extracted = pytesseract.image_to_string(image, config="--psm 6")
    except pytesseract.pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR is not installed or was not found. Install the Tesseract OCR app, "
            "then restart the Flask server. Setup instructions are in README.md."
        ) from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a readable image. Please choose a valid screenshot.") from exc

    # Normalize whitespace for more consistent model input and keyword checks.
    return re.sub(r"\s+", " ", extracted).strip()


def get_warning_signals(text, company, salary, website):
    combined = f"{text} {company} {salary} {website}".lower()
    checks = [
        (r"\b(registration fee|pay.*fee|upfront payment|security deposit|training fee)\b", "Requests an upfront fee or payment"),
        (r"\b(otp|one.time password|password|pin|bank details)\b", "Mentions sensitive credentials or financial details"),
        (r"\b(guaranteed job|no interview|earn .*daily|easy money|instant hiring)\b", "Uses unusually strong or unrealistic promises"),
        (r"\b(act now|urgent|immediately|limited time|today only)\b", "Uses pressure or urgency language"),
        (r"\b(whatsapp only|telegram only|message on telegram)\b", "Pushes communication to an informal channel"),
        (r"\b(no experience required.*high salary|work from home.*\$?\d{3,})\b", "Pairs minimal requirements with unusually high pay"),
    ]
    signals = [label for pattern, label in checks if re.search(pattern, combined)]
    if not company.strip():
        signals.append("Company name was not provided in the separate company field")
    if not website.strip():
        signals.append("Company website was not provided in the separate website field")
    return signals


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    form = {"job_title": "", "company": "", "salary": "", "website": "", "description": ""}
    if request.method == "POST":
        form = {key: request.form.get(key, "").strip() for key in form}
        ocr_text = ""
        image_upload = request.files.get("job_image")

        if image_upload and image_upload.filename:
            try:
                ocr_text = extract_text_from_image(image_upload)
                if not ocr_text:
                    result = {"error": "No readable text was found in the image. Try a clearer, well-lit screenshot or paste the job description."}
            except (ValueError, RuntimeError) as exc:
                result = {"error": str(exc)}

        if result is None:
            description_for_analysis = " ".join(part for part in [form["description"], ocr_text] if part)
            text = " ".join([form["job_title"], form["company"], form["salary"], form["website"], description_for_analysis])

            if len(description_for_analysis) < 15 and not form["job_title"]:
                result = {"error": "Please enter a job title or a job description (at least 15 characters), or upload a readable job screenshot."}
            else:
                ensure_model()
                with open(MODEL_PATH, "rb") as f:
                    model = pickle.load(f)
                prediction = int(model.predict([text])[0])
                probabilities = model.predict_proba([text])[0]
                classes = list(model.classes_)
                fake_index = classes.index(1) if 1 in classes else 1
                risk = round(float(probabilities[fake_index]) * 100)
                signals = get_warning_signals(text, form["company"], form["salary"], form["website"])
                # These rule-based signals are cautions, not explanations of the ML model.
                if signals:
                    risk = min(99, max(risk, min(95, 45 + 9 * len(signals))))
                label = "Potentially Suspicious" if prediction == 1 or risk >= 55 else "Lower Risk Indicators"
                result = {
                    "label": label,
                    "risk": risk,
                    "signals": signals or ["No obvious warning phrase was matched by the basic checks."],
                    "advice": ("Do not pay fees or share OTPs/passwords. Verify the employer using independently found official contact details." if label == "Potentially Suspicious" else "Still verify the company, job listing, and recruiter through official channels before sharing personal information."),
                    "model_note": "Baseline TF-IDF + Logistic Regression model trained on a small bundled demonstration dataset. This result is not proof that a job is genuine or fraudulent.",
                    "ocr_text": ocr_text,
                }
    return render_template("index.html", result=result, form=form)


@app.errorhandler(413)
def file_too_large(_error):
    result = {"error": "The uploaded image is too large. Please use an image smaller than 8 MB."}
    form = {"job_title": "", "company": "", "salary": "", "website": "", "description": ""}
    return render_template("index.html", result=result, form=form), 413


@app.route("/health")
def health():
    return jsonify({"status": "ok", "project": "AI-Powered Fake Job Detection System", "ocr": "enabled"})


if __name__ == "__main__":
    ensure_model()
    app.run(debug=True)
