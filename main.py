import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatCohere
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from dotenv import load_dotenv

import os, io, base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

# -------------------------------------------------
# UI CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Road Safety Chatbot", layout="centered")
st.title("💬 Road Safety Intervention Chatbot")
st.markdown("Context-driven IRC/MoRTH recommendations with sensor-assisted reasoning.")

# -------------------------------------------------
# BACKGROUND
# -------------------------------------------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("Road Safety.png")

# -------------------------------------------------
# ENV
# -------------------------------------------------
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# -------------------------------------------------
# DB
# -------------------------------------------------
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    return Chroma(
        persist_directory="index",
        embedding_function=embeddings,
        client_settings=Settings(anonymized_telemetry=False)
    )

# -------------------------------------------------
# LLM
# -------------------------------------------------
@st.cache_resource
def load_llm():
    return ChatCohere(
        model="command-r-plus",
        temperature=0,
        cohere_api_key=COHERE_API_KEY
    )

db = load_db()
llm = load_llm()

# -------------------------------------------------
# RAG FUNCTION (STRICT SEPARATION)
# -------------------------------------------------
def get_intervention(issue, sensor_block, db, llm, k=3):

    search_text = issue if issue else sensor_block
    docs = db.similarity_search(search_text, k=k)
    context = "\n".join(doc.page_content for doc in docs)[:2500]

    prompt = f"""
You are a Road Safety Audit Assistant.

STRICT RULES (MANDATORY):
1. IRC / MoRTH Codes and Clauses MUST be taken ONLY from:
   - The provided CONTEXT, and/or
   - Official MoRTH / IRC websites such as:
     • https://morth.nic.in/hi/node/120
     • https://morth.nic.in/search/node/irc
2. Do NOT use unofficial websites or prior knowledge.
3. Flowchart logic is used ONLY to identify which road condition is active.
   - Do NOT treat the flowchart as a guideline or standard.
4. Do NOT invent or assume clauses.
5. If no applicable clause or official guidance exists, respond exactly:
   I don't know

================ CONDITION IDENTIFICATION RULES (INTERNAL) ================

1. FOG CONDITION RULE
- Fog condition becomes TRUE if:
  (Temperature < 45°C AND Humidity > 35%)
  OR
  (Air Quality Sensor MQ135 > 200 ppm)

- When Fog condition is TRUE:
  - Fog condition flag is activated
  - Sensor data is transmitted to server via WiFi
  - RAG model is queried with "Fog condition data"
  - System prepares to enforce visibility-based speed and alert rules

--------------------------------------------------------------------------

2. MOTION / HEAVY VEHICLE DETECTION RULE
- Motion condition becomes TRUE if:
  Motion Sensor detects continuous movement around the vehicle

- When Motion condition is TRUE:
  - System assumes presence of heavy vehicle or surrounding traffic
  - Driver alert is triggered for situational awareness
  - Condition is passed to object detection stage

--------------------------------------------------------------------------

3. OBJECT / POTHOLE DETECTION RULE
- Object detection condition becomes TRUE if:
  IR Sensor detects surface discontinuity or obstacle (e.g., pothole)

- When Object is detected:
  - Buzzer alert is activated to warn the driver
  - RAG model is queried for:
    "Speed limit for potholes / under-construction road"
  - Safe-speed recommendation is generated

--------------------------------------------------------------------------

4. STEERING & LANE TRANSITION RULE
- Lane transition condition becomes TRUE if ALL of the following hold:
  - Potentiometer value > 50% (indicating lane reduction scenario)
  - Gyroscope does NOT detect steering movement

- Interpretation:
  - Highway transition from 3 lanes to 2 lanes
  - Driver may not be reacting adequately

- When Lane transition condition is TRUE:
  - Condition is passed through logical AND gate
  - RAG model is queried with:
    "3 lane to 2 lane condition rules"
  - Advisory or warning is generated

--------------------------------------------------------------------------

5. TURNING SAFETY RULE
- Turning risk condition becomes TRUE if:
  Gyroscope turn angle > predefined limit
  AND
  Acceleration value > predefined safety threshold

- When Turning risk condition is TRUE:
  - RAG model is queried with:
    "All information regarding the turning condition"
  - Safety guidance or corrective alert is issued

--------------------------------------------------------------------------

6. ALERT & FEEDBACK RULE
- Audible alert (Buzzer) is activated if:
  - Object proximity condition is TRUE
  OR
  - Turning risk condition is TRUE

- Alerts are designed to:
  - Prompt immediate driver response
  - Reduce collision or loss-of-control risk
  
--------------------------------------------------------------------------
  
7. SHARP TURN / OVERSPEED CONDITION RULE
- Sharp turn overspeed condition becomes TRUE if:
  Absolute Gyroscope Angle Difference (Gyro Angle diff) > 25 degrees

- Interpretation:
  - Vehicle is turning sharply at a speed higher than safe limits
  - High risk of skidding or loss of control

- When Sharp turn overspeed condition is TRUE:
  - System classifies vehicle speed as unsafe for the current turn
  - Immediate driver alert is generated
  - Buzzer is activated to warn the driver
  - RAG model is queried with:
    "Safe speed reduction guidance for sharp turns"
  - Recommendation is issued to slow down the vehicle


=========================================================================



CONTEXT (PRIMARY SOURCE FOR CODES & CLAUSES):
{context}

SENSOR DATA (may be absent):
{sensor_block if sensor_block else "Not provided"}

USER QUERY (may be absent):
{issue if issue else "Not provided"}

TASK:
- Identify which condition(s) became TRUE based on sensor data and/or query.
- Describe the identified condition(s) in words (e.g., reduced visibility, nearby movement).
- Provide CLEAR recommendations for:
  (A) Road Safety Auditors – what must be checked, enforced, or improved.
  (B) Drivers – what actions must be taken immediately.
- Wherever a recommendation is given, ATTACH the relevant IRC / MoRTH clause
  inline in brackets, if available.
- Use CONTEXT and/or official MoRTH / IRC websites for justification.
- Do NOT explain the flowchart diagram.

RESPONSE FORMAT (STRICT):

**Condition Identified:**  
<Describe which condition(s) became true and why>

**Recommendations for Drivers:**  
- <Action 1> (Ref: IRC/MoRTH <code>, Clause <number>)  
- <Action 2> (Ref: MoRTH Guideline <section>, if applicable)

**Recommendations for Road Safety Auditors:**  
- <Audit / enforcement action> (Ref: IRC/MoRTH <code>, Clause <number>)  
- <Infrastructure / signage / control measure> (Ref: official guideline)

**Reasoning:**  
<Explain how the identified condition necessitates the above actions, supported
by the cited clause or official guidance>

**References:**  
- <IRC / MoRTH code and clause from CONTEXT>  
- <Flowchart condition identified (textual description)>  
- <Official website URL used (if any)>

"""

    return llm.invoke(prompt).content

# -------------------------------------------------
# SESSION
# -------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------------------------
# SENSOR INPUT (OPTIONAL)
# -------------------------------------------------
st.subheader("📡 Sensor Inputs (Optional)")

st.markdown("""
- Distance (meters) - IR Sensor
- Temperature (°C)  - DHT 22
- Humidity (%)      - DHT 22
- Motion Sensor (0 or 1) - PIR Motion sensor
- PPM  - mq135
- Gyro Angle diff - 6 axis accel& Gyro sensor
""")

sensor_input = st.text_area(
    "Sensor Data:",
    height=160,
    placeholder=
"""Distance:
Temperature:
Humidity:
Motion Sensor (0 or 1):
PPM:
Gyro Angle diff:"""
)

# -------------------------------------------------
# QUERY INPUT (OPTIONAL)
# -------------------------------------------------
issue = st.text_input("Road Safety Query (Optional):")

# -------------------------------------------------
# ANALYZE
# -------------------------------------------------
if st.button("Analyze"):
    if not sensor_input.strip() and not issue.strip():
        st.warning("Please provide sensor data, a query, or both.")
    else:
        with st.spinner("Analyzing conditions and matching safety clauses..."):
            response = get_intervention(issue, sensor_input, db, llm)

            if issue.strip():
                st.session_state.chat_history.append(("🧑‍💼 User Query", issue))
            if sensor_input.strip():
                st.session_state.chat_history.append(("📋 Sensor Data", sensor_input))

            st.session_state.chat_history.append(("🦺 Bot", response))

# -------------------------------------------------
# DISPLAY
# -------------------------------------------------
for role, msg in st.session_state.chat_history:
    st.markdown(f"**{role}:** {msg}")

# -------------------------------------------------
# PDF EXPORT
# -------------------------------------------------
def generate_pdf(history):
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=letter)
    pdf.setFont("Times-Roman", 12)

    y = 750
    for role, message in history:
        pdf.drawString(40, y, f"{role}:")
        y -= 15
        for line in simpleSplit(message, "Times-Roman", 12, 520):
            pdf.drawString(60, y, line)
            y -= 14
        y -= 10

    pdf.save()
    buf.seek(0)
    return buf

if st.session_state.chat_history:
    st.download_button(
        "📄 Download Chat as PDF",
        generate_pdf(st.session_state.chat_history),
        "road_safety_chat_history.pdf",
        "application/pdf"
    )
