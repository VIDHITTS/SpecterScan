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
import streamlit.components.v1 as components
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
def inject_css(theme="dark"):
    if theme == "dark":
        bg = "#0f172a"
        text = "#e2e8f0"
        text2 = "#f1f5f9"
        muted = "#94a3b8"
        card_bg = "rgba(255,255,255,0.04)"
        card_border = "rgba(255,255,255,0.08)"
        accent = "#6366f1"
        accent_light = "rgba(99,102,241,0.15)"
        danger_bg = "rgba(239,68,68,0.08)"
        danger_text = "#fca5a5"
        danger_val = "#f87171"
        safe_val = "#34d399"
        divider = "rgba(255,255,255,0.06)"
        uploader_bg = "rgba(255,255,255,0.04)"
        uploader_border = "rgba(255,255,255,0.12)"
        uploader_hover_border = "rgba(99,102,241,0.4)"
        uploader_hover_bg = "rgba(99,102,241,0.05)"
        uploader_btn_bg = "rgba(99,102,241,0.15)"
        uploader_btn_color = "#c7d2fe"
        uploader_btn_border = "rgba(99,102,241,0.3)"
        uploader_text = "#94a3b8"
        btn_disabled_bg = "rgba(255,255,255,0.06)"
        btn_disabled_color = "#64748b"
        scroll_thumb = "rgba(255,255,255,0.08)"
        scroll_hover = "rgba(255,255,255,0.15)"
        card_hover = "rgba(255,255,255,0.06)"
        header_dark_bg = "rgba(0,0,0,0.15)"
        toggle_color = "#94a3b8"
    else:
        bg = "#f8fafc"
        text = "#0f172a"
        text2 = "#1e293b"
        muted = "#64748b"
        card_bg = "#ffffff"
        card_border = "#e2e8f0"
        accent = "#6366f1"
        accent_light = "rgba(99,102,241,0.08)"
        danger_bg = "#fef2f2"
        danger_text = "#991b1b"
        danger_val = "#dc2626"
        safe_val = "#059669"
        divider = "#e2e8f0"
        uploader_bg = "#ffffff"
        uploader_border = "#e2e8f0"
        uploader_hover_border = "#6366f1"
        uploader_hover_bg = "#f5f3ff"
        uploader_btn_bg = "#6366f1"
        uploader_btn_color = "#ffffff"
        uploader_btn_border = "#6366f1"
        uploader_text = "#64748b"
        btn_disabled_bg = "#f1f5f9"
        btn_disabled_color = "#94a3b8"
        scroll_thumb = "#cbd5e1"
        scroll_hover = "#94a3b8"
        card_hover = "#f8fafc"
        header_dark_bg = "#f1f5f9"
        toggle_color = "#64748b"

    st.markdown(
        f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    html, body, .stApp {{ font-family: 'Outfit', sans-serif; background: {bg}; color: {text}; }}
    .block-container {{ padding: 2rem 3rem; max-width: 100% !important; }}
    [data-testid="stHeader"] {{ display: none; }}
    [data-testid="stSidebar"] {{ display: none; }}
    #MainMenu, footer, [data-testid="stToolbar"], .stDeployButton {{ display: none; }}

    .specter-header {{ text-align: center; margin-bottom: 2.5rem; animation: fadeInDown 0.8s cubic-bezier(0.16, 1, 0.3, 1); }}
    .specter-logo {{ display: flex; align-items: center; justify-content: center; gap: 1rem; margin-bottom: 0.5rem; }}
    .specter-logo h1 {{ font-size: 2.5rem; font-weight: 700; letter-spacing: -0.03em; color: {text2}; margin: 0; }}
    .specter-header p {{ color: {muted}; font-size: 1.1rem; font-weight: 300; margin-top: 0.5rem; }}

    .supported-formats {{ font-size: 0.8rem; color: {muted}; background: {card_bg}; padding: 0.3rem 0.8rem; border-radius: 20px; border: 1px solid {card_border}; }}

    .file-info {{ text-align: center; animation: scaleIn 0.5s; background: {accent_light}; padding: 1.5rem; border-radius: 16px; border: 1px solid {card_border}; margin-top: 1.5rem; }}
    .file-info h3 {{ color: {text2}; margin-bottom: 0.25rem; font-weight: 600; }}
    .file-size {{ font-size: 0.9rem; color: {muted}; }}
    .ready-badge {{ margin-top: 1rem; display: inline-flex; align-items: center; gap: 0.5rem; color: {safe_val}; background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.2); padding: 0.5rem 1.2rem; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }}

    .risk-score-badge {{ display: flex; flex-direction: column; align-items: flex-end; background: {danger_bg}; padding: 0.6rem 1.2rem; border-radius: 12px; border: 1px solid {card_border}; }}
    .risk-score-badge .label {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: {danger_text}; font-weight: 600; }}
    .risk-score-badge .value {{ font-size: 1.4rem; font-weight: 700; color: {danger_val}; }}
    .results-filename {{ font-size: 0.9rem; color: {muted}; }}

    .stats-row {{ display: flex; gap: 1rem; margin-bottom: 1.5rem; }}
    .stat-card {{ flex: 1; background: {card_bg}; border: 1px solid {card_border}; border-radius: 16px; padding: 1.25rem; transition: transform 0.3s ease; }}
    .stat-card:hover {{ transform: translateY(-2px); }}
    .stat-card .stat-label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: {muted}; font-weight: 600; margin-bottom: 0.4rem; }}
    .stat-card .stat-value {{ font-size: 1.8rem; font-weight: 700; color: {text2}; }}
    .stat-card.danger .stat-value {{ color: {danger_val}; }}
    .stat-card.safe .stat-value {{ color: {safe_val}; }}

    .doc-viewer, .clauses-panel {{ background: {card_bg}; border: 1px solid {card_border}; border-radius: 16px; overflow: hidden; }}
    .column-title {{ padding: 1rem 1.5rem; font-size: 1rem; font-weight: 600; color: {text}; background: {card_bg}; border-bottom: 1px solid {divider}; margin: 0; }}
    .doc-body {{ padding: 1.5rem; line-height: 1.8; font-size: 0.95rem; color: {text}; max-height: 65vh; overflow-y: auto; }}
    .doc-footer {{ padding: 1.5rem; border-top: 1px solid {divider}; text-align: center; color: {muted}; font-size: 0.85rem; font-style: italic; }}
    .clause-risky {{ background: rgba(239,68,68,0.12); color: {danger_text}; border-bottom: 2px solid rgba(239,68,68,0.4); padding: 0.125rem 0.25rem; border-radius: 3px; font-weight: 500; }}

    .clauses-list-header {{ display: flex; justify-content: space-between; align-items: center; padding: 0.75rem 1.5rem; border-bottom: 1px solid {divider}; }}
    .flagged-count {{ font-size: 0.85rem; font-weight: 600; color: {muted}; text-transform: uppercase; letter-spacing: 0.05em; }}
    .clauses-list-body {{ padding: 1rem; max-height: 60vh; overflow-y: auto; display: flex; flex-direction: column; gap: 0.75rem; }}
    .empty-clauses {{ text-align: center; padding: 3rem 1rem; color: {muted}; font-size: 1rem; }}

    .clause-card {{ background: {card_bg}; border: 1px solid {card_border}; border-left: 4px solid #ef4444; border-radius: 12px; overflow: hidden; transition: all 0.2s ease; }}
    .clause-card:hover {{ background: {card_hover}; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
    .card-header {{ display: flex; justify-content: space-between; align-items: center; padding: 0.75rem 1rem; border-bottom: 1px solid {divider}; background: {header_dark_bg}; }}
    .risk-badge {{ display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.25rem 0.75rem; border-radius: 20px; background: rgba(239,68,68,0.12); color: {danger_text}; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }}
    .clause-num {{ display: flex; flex-direction: column; align-items: flex-end; }}
    .clause-num-label {{ font-size: 0.6rem; color: {muted}; text-transform: uppercase; letter-spacing: 0.05em; }}
    .clause-num-value {{ font-size: 1.1rem; font-weight: 700; color: {danger_val}; line-height: 1; }}
    .card-body {{ padding: 0.85rem 1rem; }}
    .card-body p {{ font-size: 0.9rem; line-height: 1.6; color: {text}; margin: 0; word-wrap: break-word; overflow-wrap: break-word; }}

    /* File Uploader */
    [data-testid="stFileUploader"] > div {{ background: {uploader_bg} !important; border: 2px dashed {uploader_border} !important; border-radius: 16px !important; padding: 1.5rem !important; transition: all 0.3s ease !important; }}
    [data-testid="stFileUploader"] > div:hover {{ border-color: {uploader_hover_border} !important; background: {uploader_hover_bg} !important; }}
    [data-testid="stFileUploader"] label {{ display: none; }}
    [data-testid="stFileUploader"] button {{ background: {uploader_btn_bg} !important; color: {uploader_btn_color} !important; border: 1px solid {uploader_btn_border} !important; border-radius: 8px !important; }}
    [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] p {{ color: {uploader_text} !important; }}
    [data-testid="stFileUploadDropzone"] {{ background: {uploader_bg} !important; border-color: {uploader_border} !important; }}
    [data-testid="stFileUploadDropzone"]:hover {{ border-color: {uploader_hover_border} !important; background: {uploader_hover_bg} !important; }}
    [data-testid="stFileUploadDropzone"] span {{ color: {uploader_text} !important; }}
    [data-testid="stFileUploadDropzone"] svg {{ fill: {muted} !important; stroke: {muted} !important; }}

    /* Buttons */
    .stButton > button[kind="primary"] {{ background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; border: none !important; color: white !important; font-weight: 600 !important; border-radius: 10px !important; padding: 0.75rem 1.5rem !important; transition: all 0.3s ease !important; box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important; }}
    .stButton > button[kind="primary"]:hover {{ transform: translateY(-2px) !important; box-shadow: 0 8px 20px rgba(99,102,241,0.4) !important; }}
    .stButton > button:disabled {{ background: {btn_disabled_bg} !important; color: {btn_disabled_color} !important; border: 1px solid {card_border} !important; }}

    /* Theme toggle */
    .theme-toggle {{ position: fixed; top: 1rem; right: 1.5rem; z-index: 9999; }}
    .theme-toggle button {{ background: {card_bg} !important; border: 1px solid {card_border} !important; color: {toggle_color} !important; border-radius: 50% !important; width: 40px !important; height: 40px !important; padding: 0 !important; font-size: 1.2rem !important; cursor: pointer !important; transition: all 0.3s !important; min-height: 0 !important; }}
    .theme-toggle button:hover {{ background: {accent_light} !important; transform: scale(1.1) !important; }}

    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: {scroll_thumb}; border-radius: 10px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {scroll_hover}; }}
    @keyframes fadeInDown {{ from {{ opacity: 0; transform: translateY(-20px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    @keyframes scaleIn {{ from {{ opacity: 0; transform: scale(0.95); }} to {{ opacity: 1; transform: scale(1); }} }}
    </style>
    """,
        unsafe_allow_html=True,
    )


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


def analyze_document(
    file_bytes: bytes, filename: str, embedder, classifier, nlp
) -> dict:
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
        results.append(
            {
                "clause_index": i + 1,
                "clause_text": clause_text,
                "risk_label": label,
                "risk_category": RISK_LABELS.get(label, "Unknown"),
            }
        )

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
    st.markdown(
        """
    <div class="specter-header">
        <div class="specter-logo">
            <h1>SpecterScan</h1>
        </div>
        <p>Contract Risk Classification System</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Upload zone
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Icon and text above the uploader
        st.markdown(
            """
        <div style="text-align:center; margin-bottom: 0.5rem;">
            <h3 class="specter-header" style="font-size:1.25rem; margin-bottom:0.5rem;">Upload Contract Document</h3>
            <p style="margin-bottom:1rem;" class="results-filename">Drag and drop your PDF or text file here, or click to browse</p>
            <span class="supported-formats">Supports .pdf, .txt</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload contract",
            type=["pdf", "txt"],
            label_visibility="collapsed",
            key="file_uploader",
        )

        if uploaded_file:
            file_size_mb = uploaded_file.size / 1024 / 1024
            st.markdown(
                f"""
            <div class="file-info">
                <h3>📄 {uploaded_file.name}</h3>
                <p class="file-size">{file_size_mb:.2f} MB</p>
                <div class="ready-badge">✅ Ready for analysis</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

        # Analyze button — runs analysis immediately on click
        if uploaded_file:
            if st.button(
                "🔍  Analyze Document", use_container_width=True, type="primary"
            ):
                with st.spinner("Analyzing your document…"):
                    try:
                        file_bytes = uploaded_file.getvalue()
                        data = analyze_document(
                            file_bytes,
                            uploaded_file.name,
                            embedder,
                            classifier,
                            nlp,
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
        st.markdown(
            f"""
        <div>
            <h2 style="font-size:1.25rem; margin:0;">Analysis Results</h2>
            <span class="results-filename">{data['filename']}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col_score:
        st.markdown(
            f"""
        <div class="risk-score-badge">
            <span class="label">Total Risk Score</span>
            <span class="value">{risk_score}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<hr style='margin: 0.5rem 0 1.25rem; border:none; border-top:1px solid currentColor; opacity:0.15;'>",
        unsafe_allow_html=True,
    )

    # ── Summary Stats ──
    st.markdown(
        f"""
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
    """,
        unsafe_allow_html=True,
    )

    # ── Split Layout: Document | Flagged Clauses ──
    left_col, right_col = st.columns([3, 2])

    # LEFT — Document Viewer
    with left_col:
        # Build full HTML and render via components.html to avoid size limits
        theme = st.session_state.get("theme", "dark")
        if theme == "dark":
            doc_bg = "transparent"
            doc_text = "#cbd5e1"
            doc_title = "#e2e8f0"
            doc_card = "rgba(255,255,255,0.03)"
            doc_border = "rgba(255,255,255,0.08)"
            doc_divider = "rgba(255,255,255,0.06)"
            risky_text = "#fca5a5"
            doc_muted = "#64748b"
            doc_scroll = "rgba(255,255,255,0.08)"
        else:
            doc_bg = "transparent"
            doc_text = "#334155"
            doc_title = "#0f172a"
            doc_card = "#ffffff"
            doc_border = "#e2e8f0"
            doc_divider = "#e2e8f0"
            risky_text = "#991b1b"
            doc_muted = "#64748b"
            doc_scroll = "#cbd5e1"

        doc_html = '<div class="doc-body">'
        for clause in data["results"]:
            text = clause["clause_text"].replace("<", "&lt;").replace(">", "&gt;")
            text = " ".join(text.split())
            if clause["risk_label"] == 1:
                doc_html += f'<span class="clause-risky">{text}</span> '
            else:
                doc_html += f"{text} "
        doc_html += '</div><div class="doc-footer">End of Document</div>'

        full_doc = f"""
        <html><head>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
        body {{ font-family: 'Outfit', sans-serif; background: {doc_bg}; color: {doc_text}; margin: 0; padding: 0; }}
        .doc-viewer {{ background: {doc_card}; border: 1px solid {doc_border}; border-radius: 16px; overflow: hidden; }}
        .column-title {{ padding: 1rem 1.5rem; font-size: 1rem; font-weight: 600; color: {doc_title}; background: {doc_card}; border-bottom: 1px solid {doc_divider}; }}
        .doc-body {{ padding: 1.5rem; line-height: 1.9; font-size: 0.95rem; color: {doc_text}; }}
        .clause-risky {{ background: rgba(239,68,68,0.12); color: {risky_text}; border-bottom: 2px solid rgba(239,68,68,0.4); padding: 0.125rem 0.25rem; border-radius: 3px; font-weight: 500; }}
        .doc-footer {{ padding: 1rem 1.5rem; border-top: 1px solid {doc_divider}; text-align: center; color: {doc_muted}; font-size: 0.85rem; font-style: italic; }}
        ::-webkit-scrollbar {{ width: 6px; }} ::-webkit-scrollbar-track {{ background: transparent; }} ::-webkit-scrollbar-thumb {{ background: {doc_scroll}; border-radius: 10px; }}
        </style></head><body>
        <div class="doc-viewer"><div class="column-title">Document Content</div>{doc_html}</div>
        </body></html>"""
        components.html(full_doc, height=600, scrolling=True)

    # RIGHT — Flagged Clauses List
    with right_col:
        flagged = [c for c in data["results"] if c["risk_label"] == 1]

        # Render panel header
        st.markdown(
            f'<div class="clauses-panel">'
            f'<div class="column-title">Flagged Clauses List</div>'
            f'<div class="clauses-list-header"><span class="flagged-count">{len(flagged)} Flagged Clauses</span></div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        if not flagged:
            st.markdown(
                '<div class="empty-clauses">🎉 No risky clauses detected!</div>',
                unsafe_allow_html=True,
            )
        else:
            # Render each card INDIVIDUALLY to avoid Streamlit HTML size limits
            for clause in flagged:
                # Clean text: escape HTML, strip newlines, collapse whitespace
                text = clause["clause_text"].replace("<", "&lt;").replace(">", "&gt;")
                text = " ".join(
                    text.split()
                )  # collapse all whitespace/newlines into single spaces
                st.markdown(
                    f'<div class="clause-card">'
                    f'<div class="card-header">'
                    f'<div class="risk-badge">⚠️ {clause["risk_category"]}</div>'
                    f'<div class="clause-num">'
                    f'<span class="clause-num-label">Clause</span>'
                    f'<span class="clause-num-value">#{clause["clause_index"]}</span>'
                    f"</div></div>"
                    f'<div class="card-body"><p>"{text}"</p></div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════
# MAIN APP ENTRY
# ══════════════════════════════════════════════════
def main():
    # ── Session State ──
    if "view" not in st.session_state:
        st.session_state.view = "upload"
    if "results" not in st.session_state:
        st.session_state.results = None
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"

    # ── Inject CSS for current theme ──
    inject_css(st.session_state.theme)

    # ── Theme Toggle (top-right corner) ──
    toggle_col = st.columns([20, 1])[1]
    with toggle_col:
        icon = "🌙" if st.session_state.theme == "light" else "☀️"
        st.markdown('<div class="theme-toggle">', unsafe_allow_html=True)
        if st.button(icon, key="theme_toggle"):
            st.session_state.theme = (
                "light" if st.session_state.theme == "dark" else "dark"
            )
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

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
