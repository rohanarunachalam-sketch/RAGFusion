import os
import requests
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000"
).rstrip("/")


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Hybrid RAG Assistant",
    page_icon="🔎",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #666666;
        margin-bottom: 30px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        background-color: #fafafa;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">Hybrid RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions from your document using Vector Search + Knowledge Graph.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# BACKEND HEALTH CHECK
# =========================================================

def check_backend():

    try:

        response = requests.get(
            f"{BACKEND_URL}/health",
            timeout=10
        )

        if response.status_code == 200:
            return True

        return False

    except requests.RequestException:
        return False


# =========================================================
# PROCESS PDF
# =========================================================

def process_pdf(uploaded_file):

    try:

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf"
            )
        }

        response = requests.post(
            f"{BACKEND_URL}/process",
            files=files,
            timeout=600
        )

        if response.status_code == 200:

            data = response.json()

            if data.get("success"):

                return {
                    "success": True,
                    "message": data.get(
                        "message",
                        "PDF processed successfully."
                    ),
                    "filename": data.get(
                        "filename",
                        uploaded_file.name
                    )
                }

            return {
                "success": False,
                "message": data.get(
                    "message",
                    "PDF processing failed."
                )
            }

        try:
            error_data = response.json()
            error_message = error_data.get(
                "detail",
                "Unknown backend error."
            )
        except Exception:
            error_message = response.text

        return {
            "success": False,
            "message": error_message
        }

    except requests.Timeout:

        return {
            "success": False,
            "message": "Backend request timed out while processing the PDF."
        }

    except requests.RequestException as e:

        return {
            "success": False,
            "message": f"Could not connect to backend: {e}"
        }


# =========================================================
# ASK QUESTION
# =========================================================

def ask_backend(question):

    try:

        response = requests.post(
            f"{BACKEND_URL}/ask",
            json={
                "question": question
            },
            timeout=300
        )

        if response.status_code == 200:

            data = response.json()

            if data.get("success"):

                return {
                    "success": True,
                    "answer": data.get(
                        "answer",
                        ""
                    )
                }

            return {
                "success": False,
                "message": "Backend could not generate an answer."
            }

        try:
            error_data = response.json()
            error_message = error_data.get(
                "detail",
                "Unknown backend error."
            )
        except Exception:
            error_message = response.text

        return {
            "success": False,
            "message": error_message
        }

    except requests.Timeout:

        return {
            "success": False,
            "message": "Backend request timed out while generating the answer."
        }

    except requests.RequestException as e:

        return {
            "success": False,
            "message": f"Could not connect to backend: {e}"
        }


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Document")

    backend_status = check_backend()

    if backend_status:

        st.success("Backend connected")

    else:

        st.error("Backend unavailable")

    st.caption(f"Backend: {BACKEND_URL}")

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    st.divider()

    if uploaded_file:

        st.write(
            f"**Selected:** {uploaded_file.name}"
        )

        process_button = st.button(
            "Process PDF",
            type="primary",
            use_container_width=True
        )

        if process_button:

            if not backend_status:

                st.error(
                    "Backend is not available. "
                    "Please start the backend first."
                )

            else:

                with st.spinner(
                    "Processing PDF..."
                ):

                    result = process_pdf(
                        uploaded_file
                    )

                if result["success"]:

                    st.success(
                        result["message"]
                    )

                    st.session_state.document_processed = True

                    st.session_state.uploaded_filename = (
                        result["filename"]
                    )

                else:

                    st.error(
                        result["message"]
                    )

    else:

        st.info(
            "Upload a PDF and click "
            "**Process PDF** to start."
        )


# =========================================================
# DOCUMENT STATUS
# =========================================================

if st.session_state.document_processed:

    st.success(
        f"Document ready: "
        f"{st.session_state.uploaded_filename}"
    )


# =========================================================
# QUESTION SECTION
# =========================================================

if st.session_state.document_processed:

    st.subheader("Ask a Question")

    question = st.text_area(
        "Enter your question",
        placeholder=(
            "Example: What technologies are used "
            "in the web application?"
        ),
        height=120
    )

    ask_button = st.button(
        "Ask Question",
        type="primary",
        use_container_width=True
    )

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        elif not check_backend():

            st.error(
                "Backend is not available."
            )

        else:

            with st.spinner(
                "Searching the document and generating answer..."
            ):

                result = ask_backend(
                    question.strip()
                )

            if result["success"]:

                st.subheader("Answer")

                st.markdown(
                    '<div class="answer-box">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    result["answer"]
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

            else:

                st.error(
                    result["message"]
                )


# =========================================================
# GET STARTED
# =========================================================

else:

    st.subheader("Get Started")

    st.write(
        "Upload a PDF from the sidebar and click "
        "**Process PDF**."
    )

    st.write(
        "The document will be processed into:"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            "**Vector Database**\n\n"
            "FAISS stores semantic representations "
            "of document chunks."
        )

    with col2:

        st.info(
            "**Knowledge Graph**\n\n"
            "Neo4j stores entities and relationships "
            "extracted from the document."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Hybrid RAG • Streamlit + FastAPI + FAISS + Neo4j + GPT-4.1-mini"
)