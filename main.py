import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatCohere
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from dotenv import load_dotenv
import os

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

import io
import base64

# Streamlit UI
st.set_page_config(page_title="Road Safety Chatbot", layout="centered")
st.title("💬 Road Safety Intervention Chatbot (RAG-Powered)")
st.markdown("Ask multiple road safety questions. Context is retrieved fresh for each.")

# Background
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("Road Safety.png")

# Load env env vars
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")

# Load Chroma DB
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = Chroma(
        persist_directory="index",
        embedding_function=embeddings,
        client_settings=Settings(anonymized_telemetry=False)
    )
    return db

# Load Cohere LLM
@st.cache_resource
def load_llm():
    return ChatCohere(model="command-r-plus", temperature=0, cohere_api_key=cohere_api_key)

# RAG function
def get_intervention(query, db, llm, k=3):
    results = db.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in results])[:2500]

    prompt = f"""
    Based on the following road safety guidelines:\n{context}\n
    Respond to this issue: {query}
    Provide a clear recommendation with reasoning and reference any clause or section if applicable.
    If unsure, respond with 'I don't know'.
    """
    return llm.invoke(prompt).content

# PDF function
def generate_pdf(chat_history):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)

    bg_image = "Pedestrian2.png"
    width, height = letter

    def draw_background():
        pdf.drawImage(bg_image, 0, 0, width=width, height=height, mask='auto')

    draw_background()

    pdf.setFont("Times-Bold", 16)
    pdf.drawString(40, 750, "Road Safety Intervention Chatbot - Conversation History")
    pdf.setFont("Times-Roman", 12)

    x, y = 40, 710
    max_width = 520

    for sender, msg in chat_history:
        pdf.setFont("Times-Bold", 12)
        pdf.drawString(x, y, f"{sender.upper()}:")
        y -= 20

        wrapped = simpleSplit(msg, "Times-Roman", 12, max_width)
        pdf.setFont("Times-Roman", 12)

        for line in wrapped:
            if y < 60:
                pdf.showPage()
                draw_background()
                y = 750
            pdf.drawString(x + 20, y, line)
            y -= 18

        y -= 10

    pdf.save()
    buffer.seek(0)
    return buffer

# Load resources
db = load_db()
llm = load_llm()

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input UI
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask your question:", key="input")
    submit = st.form_submit_button("Send")

if submit and user_input:
    with st.spinner("Thinking..."):
        answer = get_intervention(user_input, db, llm)
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", answer))

# Display chat
for sender, message in st.session_state.chat_history:
    if sender == "user":
        st.markdown(f"🧑 **You:** {message}")
    else:
        st.markdown(f"🤖 **Bot:** {message}")

# PDF download
if st.session_state.chat_history:
    pdf_buffer = generate_pdf(st.session_state.chat_history)
    st.download_button(
        label="📄 Download Chat as PDF",
        data=pdf_buffer,
        file_name="road_safety_chat_history.pdf",
        mime="application/pdf"
    )
