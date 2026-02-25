"""
SpecterScan — Streamlit Application
====================================
All-in-one legal contract risk analysis.
Frontend + Backend + ML in a single deployable app.

Run:  streamlit run app.py
"""

import os
import io
import logging

import streamlit as st
import spacy
import joblib
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# ── Configuration ────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "legal_risk_classifier.pkl")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RISK_LABELS = {
    0: "Normal / Compliant",
    1: "Risky / Potential Issue",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("specterscan")


# ── Page Config ──────────────────────────────────
st.set_page_config(
    page_title="SpecterScan",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ══════════════════════════════════════════════════
# CUSTOM CSS — Replicates the React frontend design
# ══════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, .stApp {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        background-color: #f8fafc;
        color: #0f172a;
    }
    .block-container { padding-top: 2rem; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { display: none; }

    /* ── Upload View ── */
    .specter-header {
        text-align: center;
        margin-bottom: 2.5rem;
        animation: fadeInDown 0.6s ease-out;
    }
    .specter-logo {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .specter-logo-icon {
        width: 28px;
        height: 28px;
        background: linear-gradient(135deg, #2563eb, #60a5fa);
        border-radius: 6px;
        transform: rotate(45deg);
        display: inline-block;
    }
    .specter-logo h1 {
        font-size: 2rem;
        letter-spacing: -0.02em;
        color: #0f172a;
        margin: 0;
        padding: 0;
    }
    .specter-header p {
        color: #64748b;
        font-size: 1.125rem;
        margin-top: 0.25rem;
    }

    /* ── Drop Zone ── */
    .upload-zone {
        width: 100%;
        max-width: 640px;
        margin: 0 auto;
        border: 2px dashed #e2e8f0;
        border-radius: 16px;
        background: #ffffff;
        padding: 3rem 2rem;
        text-align: center;
        transition: all 0.3s;
        box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05);
        cursor: pointer;
    }
    .upload-zone:hover {
        border-color: #2563eb;
        background: #f0fdf4;
    }
    .upload-icon-circle {
        width: 64px;
        height: 64px;
        border-radius: 50%;
        background: #f1f5f9;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1.5rem;
        font-size: 1.75rem;
    }
    .upload-zone h3 {
        font-size: 1.25rem;
        margin-bottom: 0.5rem;
        color: #0f172a;
    }
    .upload-zone p {
        color: #64748b;
        margin-bottom: 1rem;
    }
    .supported-formats {
        font-size: 0.875rem;
        color: #94a3b8;
        background: #f1f5f9;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        display: inline-block;
    }

    /* ── File Ready State ── */
    .file-info {
        text-align: center;
        animation: scaleIn 0.4s ease-out;
    }
    .file-info h3 { color: #0f172a; margin-bottom: 0.25rem; }
    .file-size { font-size: 0.875rem; color: #64748b; }
    .ready-badge {
        margin-top: 1.25rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        color: #059669;
        background: #d1fae5;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-weight: 500;
        font-size: 0.875rem;
    }

    /* ── Analyze Button ── */
    .analyze-btn {
        display: block;
        width: 100%;
        max-width: 400px;
        margin: 2rem auto 0;
        background: #2563eb;
        color: white;
        border: none;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s;
        text-align: center;
    }
    .analyze-btn:hover {
        background: #1d4ed8;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99,102,241,0.3);
    }
    .analyze-btn:disabled, .analyze-btn.disabled {
        background: #cbd5e1;
        cursor: not-allowed;
        transform: none;
        box-shadow: none;
    }

    /* ── Results Header ── */
    .results-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #e2e8f0;
    }
    .results-header-left {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .results-header-left h2 {
        font-size: 1.25rem;
        color: #0f172a;
        margin: 0;
    }
    .results-filename {
        font-size: 0.875rem;
        color: #64748b;
    }
    .risk-score-badge {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        background: #fef2f2;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border: 1px solid #fecaca;
    }
    .risk-score-badge .label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #991b1b;
        font-weight: 600;
    }
    .risk-score-badge .value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #7f1d1d;
    }

    /* ── Column Titles ── */
    .column-title {
        padding: 1rem 1.25rem;
        font-size: 1rem;
        font-weight: 600;
        color: #0f172a;
        border-bottom: 1px solid #e2e8f0;
        background: #f8fafc;
        border-radius: 12px 12px 0 0;
        margin: 0;
    }

    /* ── Document Viewer ── */
    .doc-viewer {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05);
        overflow: hidden;
    }
    .doc-body {
        padding: 1.5rem;
        line-height: 1.8;
        font-size: 1rem;
        color: #334155;
        max-height: 70vh;
        overflow-y: auto;
    }
    .clause-normal {
        padding-right: 0.25rem;
    }
    .clause-risky {
        background: #fee2e2;
        border-bottom: 2px solid #ef4444;
        padding: 0.125rem 0.25rem;
        border-radius: 2px;
        font-weight: 500;
        cursor: default;
    }
    .clause-risky:hover { background: #fecaca; }
    .doc-footer {
        margin-top: 2rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e2e8f0;
        text-align: center;
        color: #64748b;
        font-size: 0.875rem;
        font-style: italic;
    }

    /* ── Clause Cards ── */
    .clauses-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05);
        overflow: hidden;
    }
    .clauses-list-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.85rem 1.25rem;
        border-bottom: 1px solid #e2e8f0;
        background: #ffffff;
    }
    .flagged-count {
        font-size: 0.875rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .clauses-list-body {
        padding: 1.25rem;
        max-height: 66vh;
        overflow-y: auto;
        background: #f8fafc;
        display: flex;
        flex-direction: column;
        gap: 1.25rem;
    }
    .clause-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #ef4444;
        border-radius: 12px;
        box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05);
        overflow: hidden;
        transition: all 0.2s;
    }
    .clause-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.85rem 1.15rem;
        border-bottom: 1px solid #e2e8f0;
        background: #fafaf9;
    }
    .risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        padding: 0.2rem 0.7rem;
        border-radius: 9999px;
        background: #fee2e2;
        color: #991b1b;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .clause-num {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
    }
    .clause-num-label {
        font-size: 0.6rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .clause-num-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #b91c1c;
        line-height: 1;
    }
    .card-body {
        padding: 1.15rem;
    }
    .card-body p {
        font-size: 0.95rem;
        line-height: 1.6;
        color: #0f172a;
        font-style: italic;
    }
    .card-footer {
        padding: 0.75rem 1.15rem;
        border-top: 1px dashed #e2e8f0;
        background: #fcfcfc;
        text-align: right;
    }
    .review-btn {
        font-size: 0.875rem;
        color: #2563eb;
        font-weight: 600;
        background: none;
        border: none;
        padding: 0.375rem 0.75rem;
        border-radius: 8px;
        cursor: pointer;
    }
    .review-btn:hover { background: #eff6ff; }

    /* ── Empty State ── */
    .empty-clauses {
        text-align: center;
        padding: 3rem 1rem;
        color: #64748b;
        font-size: 1rem;
    }

    /* ── Spinner ── */
    .analyzing-spinner {
        text-align: center;
        padding: 4rem 2rem;
    }
    .analyzing-spinner p {
        color: #64748b;
        font-size: 1.1rem;
        margin-top: 1rem;
    }

    /* ── Summary Stats Row ── */
    .stats-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .stat-card {
        flex: 1;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05);
    }
    .stat-card .stat-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .stat-card .stat-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
    }
    .stat-card.danger .stat-value { color: #dc2626; }
    .stat-card.safe .stat-value { color: #059669; }

    /* ── Animations ── */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.95); }
        to   { opacity: 1; transform: scale(1); }
    }

    /* ── Hide Streamlit defaults ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbar"] {display: none;}
    .stDeployButton {display: none;}

    /* ── Streamlit file uploader styling ── */
    [data-testid="stFileUploader"] {
        max-width: 640px;
        margin: 0 auto;
    }
    [data-testid="stFileUploader"] > div {
        border: 2px dashed #e2e8f0;
        border-radius: 16px;
        background: #ffffff;
        padding: 2rem;
        transition: all 0.3s;
    }
    [data-testid="stFileUploader"] > div:hover {
        border-color: #2563eb;
        background: #f0fdf4;
    }
    [data-testid="stFileUploader"] label {
        display: none;
    }

    /* ── Back button ── */
    .back-btn {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: none;
        border: none;
        color: #64748b;
        font-size: 0.95rem;
        font-weight: 500;
        cursor: pointer;
        padding: 0.5rem 0.75rem;
        border-radius: 50%;
        transition: all 0.2s;
    }
    .back-btn:hover {
        background: #f1f5f9;
        color: #0f172a;
    }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# MODEL LOADING (cached — runs only once)
# ══════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_models():
    """Load all ML models once and cache them across reruns."""
    logger.info("Loading ML models...")

    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("SentenceTransformer loaded.")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}\n"
            "Run 'python train_model.py' first to train the classifier."
        )
    classifier = joblib.load(MODEL_PATH)
    logger.info("Classifier loaded.")

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise OSError(
            "spaCy model 'en_core_web_sm' not found.\n"
            "Run: python -m spacy download en_core_web_sm"
        )
    logger.info("spaCy model loaded.")

    return embedder, classifier, nlp


# ══════════════════════════════════════════════════
# BACKEND LOGIC
# ══════════════════════════════════════════════════
def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1")


def segment_into_clauses(text: str, nlp) -> list[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) >= 5]


def analyze_document(file_bytes: bytes, filename: str, embedder, classifier, nlp) -> dict:
    """Run the full analysis pipeline — same logic as the FastAPI backend."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        raw_text = extract_text_from_pdf(file_bytes)
    elif ext == ".txt":
        raw_text = extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: '{ext}'. Upload .pdf or .txt.")

    if not raw_text or not raw_text.strip():
        raise ValueError("No readable text could be extracted from the file.")

    clauses = segment_into_clauses(raw_text, nlp)
    if not clauses:
        raise ValueError("No meaningful clauses could be extracted.")

    embeddings = embedder.encode(clauses, show_progress_bar=False)
    predictions = classifier.predict(embeddings)

    results = []
    for i, (clause_text, pred) in enumerate(zip(clauses, predictions)):
        label = int(pred)
        results.append({
            "clause_index": i + 1,
            "clause_text": clause_text,
            "risk_label": label,
            "risk_category": RISK_LABELS.get(label, "Unknown"),
        })

    return {
        "filename": filename,
        "total_clauses": len(results),
        "results": results,
    }


# ══════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════
def render_upload_view(embedder, classifier, nlp):
    """Render the upload page — replicates the React UploadView."""

    # Header
    st.markdown("""
    <div class="specter-header">
        <div class="specter-logo">
            <div class="specter-logo-icon"></div>
            <h1>SpecterScan</h1>
        </div>
        <p>Contract Risk Classification System</p>
    </div>
    """, unsafe_allow_html=True)

    # Upload zone
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Icon and text above the uploader
        st.markdown("""
        <div style="text-align:center; margin-bottom: 0.5rem;">
            <div class="upload-icon-circle">☁️</div>
            <h3 style="font-size:1.25rem; margin-bottom:0.5rem; color:#0f172a;">Upload Contract Document</h3>
            <p style="color:#64748b; margin-bottom:1rem;">Drag and drop your PDF or text file here, or click to browse</p>
            <span class="supported-formats">Supports .pdf, .txt</span>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload contract",
            type=["pdf", "txt"],
            label_visibility="collapsed",
            key="file_uploader",
        )

        if uploaded_file:
            file_size_mb = uploaded_file.size / 1024 / 1024
            st.markdown(f"""
            <div class="file-info">
                <h3>📄 {uploaded_file.name}</h3>
                <p class="file-size">{file_size_mb:.2f} MB</p>
                <div class="ready-badge">✅ Ready for analysis</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

        # Analyze button — runs analysis immediately on click
        if uploaded_file:
            if st.button("🔍  Analyze Document", use_container_width=True, type="primary"):
                with st.spinner("Analyzing your document…"):
                    try:
                        file_bytes = uploaded_file.getvalue()
                        data = analyze_document(
                            file_bytes, uploaded_file.name,
                            embedder, classifier, nlp,
                        )
                        st.session_state.results = data
                        st.session_state.view = "results"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
        else:
            st.button(
                "Analyze Document",
                use_container_width=True,
                disabled=True,
            )


def render_results_view(data: dict):
    """Render the results page — replicates the React ResultsView."""

    total = data["total_clauses"]
    risky = sum(1 for r in data["results"] if r["risk_label"] == 1)
    safe = total - risky
    risk_score = f"{risky / total:.2f}" if total > 0 else "0.00"

    # ── Header ──
    col_back, col_info, col_score = st.columns([0.5, 6, 2])
    with col_back:
        if st.button("← ", key="back_btn"):
            st.session_state.view = "upload"
            st.session_state.results = None
            st.rerun()
    with col_info:
        st.markdown(f"""
        <div>
            <h2 style="font-size:1.25rem; color:#0f172a; margin:0;">Analysis Results</h2>
            <span class="results-filename">{data['filename']}</span>
        </div>
        """, unsafe_allow_html=True)
    with col_score:
        st.markdown(f"""
        <div class="risk-score-badge">
            <span class="label">Total Risk Score</span>
            <span class="value">{risk_score}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 0.5rem 0 1.25rem; border:none; border-top:1px solid #e2e8f0;'>", unsafe_allow_html=True)

    # ── Summary Stats ──
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-label">Total Clauses</div>
            <div class="stat-value">{total}</div>
        </div>
        <div class="stat-card danger">
            <div class="stat-label">Risky Clauses</div>
            <div class="stat-value">{risky}</div>
        </div>
        <div class="stat-card safe">
            <div class="stat-label">Safe Clauses</div>
            <div class="stat-value">{safe}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Split Layout: Document | Flagged Clauses ──
    left_col, right_col = st.columns([3, 2])

    # LEFT — Document Viewer
    with left_col:
        doc_html = '<div class="doc-viewer"><div class="column-title">Document Content</div><div class="doc-body">'
        for clause in data["results"]:
            text = clause["clause_text"].replace("<", "&lt;").replace(">", "&gt;")
            if clause["risk_label"] == 1:
                doc_html += f'<span class="clause-risky">{text}</span> '
            else:
                doc_html += f'<span class="clause-normal">{text}</span> '
        doc_html += '<div class="doc-footer"><p>End of Document</p></div></div></div>'
        st.markdown(doc_html, unsafe_allow_html=True)

    # RIGHT — Flagged Clauses List
    with right_col:
        flagged = [c for c in data["results"] if c["risk_label"] == 1]
        cards_html = '<div class="clauses-panel">'
        cards_html += '<div class="column-title">Flagged Clauses List</div>'
        cards_html += f'<div class="clauses-list-header"><span class="flagged-count">{len(flagged)} Flagged Clauses</span></div>'
        cards_html += '<div class="clauses-list-body">'

        if not flagged:
            cards_html += '<div class="empty-clauses">🎉 No risky clauses detected!</div>'
        else:
            for clause in flagged:
                text = clause["clause_text"].replace("<", "&lt;").replace(">", "&gt;")
                cards_html += f"""
                <div class="clause-card">
                    <div class="card-header">
                        <div class="risk-badge">⚠️ {clause['risk_category']}</div>
                        <div class="clause-num">
                            <span class="clause-num-label">Clause</span>
                            <span class="clause-num-value">#{clause['clause_index']}</span>
                        </div>
                    </div>
                    <div class="card-body">
                        <p>"{text}"</p>
                    </div>
                    <div class="card-footer">
                        <span class="review-btn">Review Section</span>
                    </div>
                </div>
                """

        cards_html += '</div></div>'
        st.markdown(cards_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# MAIN APP ENTRY
# ══════════════════════════════════════════════════
def main():
    inject_css()

    # ── Session State ──
    if "view" not in st.session_state:
        st.session_state.view = "upload"
    if "results" not in st.session_state:
        st.session_state.results = None

    # ── Load Models (once) ──
    with st.spinner("Loading ML models… (first run only, please wait)"):
        try:
            embedder, classifier, nlp = load_models()
        except (FileNotFoundError, OSError) as e:
            st.error(str(e))
            st.stop()

    # ── Route Views ──
    if st.session_state.view == "upload":
        render_upload_view(embedder, classifier, nlp)

    elif st.session_state.view == "results" and st.session_state.results:
        render_results_view(st.session_state.results)


if __name__ == "__main__":
    main()
