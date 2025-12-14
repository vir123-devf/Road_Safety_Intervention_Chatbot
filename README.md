# 🚦 Road Safety Intervention Chatbot (RAG-Powered)

This chatbot is a GPT-based AI tool designed to identify the most appropriate road safety interventions for specific traffic safety issues observed within a road network. It leverages Retrieval-Augmented Generation (RAG) to search and retrieve relevant guidance from a curated database of interventions compiled from global best practices, IRC and MoRTH guidelines, and road safety audit reports.

The tool accepts descriptive input about road safety concerns—such as the type of road, nature of the hazard, and surrounding environmental conditions—and returns intervention recommendations that are:

* Tailored to the specific problem described
* Supported by relevant clauses or sections from the intervention database
* Clearly explained for transparency and traceability

Built using Streamlit, FAISS, HuggingFace embeddings, and Cohere’s `command-r-plus` model, the chatbot provides a user-friendly interface for engineers, planners, and traffic safety professionals to generate actionable, evidence-based decisions for improving road safety.

---

## 🌐 Live Demo

👉 **[Try the Demo](https://road-safety-intervention.streamlit.app/)**
*(Hosted via Streamlit Community Cloud or your own deployment)*

---

## 📸 Screenshots
<img width="1919" height="906" alt="image" src="https://github.com/user-attachments/assets/4c83b9ed-52ec-46c8-85b8-5eac31e3f498" />
<img width="1919" height="906" alt="image" src="https://github.com/user-attachments/assets/945cbcfc-b0a7-4fef-a203-1e2cc28389de" />
<img width="1919" height="833" alt="image" src="https://github.com/user-attachments/assets/1e496bd6-4593-4dd9-a80f-4d3dac4f80f6" />


---

## 📄 Sample PDF Report

Download a sample generated report from a real chat session:

📥 **[Sample Chat Report (PDF)](road_safety_chat_history.pdf)** 

The report includes:

* Question and answer pairs
* Clause-level references from IRC guidelines
* Structured, readable recommendations
* Styled background with official branding

---

## 🧠 Features

* 🗣 Accepts natural language queries about road safety problems
* 🔍 Retrieves relevant interventions using semantic vector search (FAISS + HuggingFace embeddings)
* 🧾 Generates responses grounded in official guidelines using Cohere’s `command-r-plus` model
* 📌 Cites specific clauses from IRC, MoRTH, or audit reports
* 🖨️ Exports chat history into a branded PDF report
* 🖼️ Includes visual backgrounds in both app and report

---
## Work FlowChart:-
<img width="1041" height="495" alt="image" src="https://github.com/user-attachments/assets/44e994e9-8d35-4004-877b-caa1a1802876" />



## 📘 Project Use Case

**Road safety intervention planning** often requires context-aware decisions based on road type, the specific safety issue, and local conditions. This tool enables quick identification of suitable interventions using AI by:

* Automatically interpreting the described problem
* Searching a curated interventions database
* Providing recommended solutions supported by official sources

It is especially useful for:

* Road safety auditors
* Transportation engineers
* Traffic planning authorities
* Urban development and smart city initiatives

---

## 🏗️ Project Structure

```
📦 road-safety-chatbot/
├── index/                      # FAISS vector index
├── GPT_Input_DB.xlsx          # Road safety interventions database
├── main.py                    # Streamlit app logic
├── cxc.py                     # (Optional) vector builder script
├── requirements.txt           # Python dependencies
├── .env_example               # Sample .env file
├── Pedestrian2.png            # Background for PDF
├── Road Safety.png            # Background for UI
```

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/road-safety-chatbot.git
cd road-safety-chatbot
```

### 2. Set Up Environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.env` file with your Cohere API key:

```
COHERE_API_KEY=your-cohere-api-key
```

### 4. Launch the App

```bash
streamlit run main.py
```

---

## 💬 Example Prompts

Try asking:

* “What should be done if a STOP sign on a rural road is faded?”
* “Suggest interventions for pedestrian crashes near schools.”
* “How do I handle poor visibility at an unsignalized junction?”
* “What signage is required for a narrow bridge in hilly areas?”

---

## 🛠️ Tech Stack

| Component      | Tech Stack                        |
| -------------- | --------------------------------- |
| UI             | Streamlit                         |
| Embeddings     | HuggingFace (MPNET Sentence BERT) |
| Vector Store   | FAISS                             |
| Language Model | Cohere `command-r-plus`           |
| PDF Generator  | ReportLab                         |
| File Handling  | dotenv, base64                    |

---

## 🔖 Future Enhancements

* 📍 Location-based filtering for region-specific interventions
* 🧾 QR Code on PDF reports to open full chat session
* 🧠 Fine-tuned LLM for domain-specific scoring
* 🌐 Integration with OpenStreetMap or GIS tools

---

## 🙏 Acknowledgements

* Indian Roads Congress (IRC)
* Ministry of Road Transport and Highways (MoRTH)
* Cohere AI
* HuggingFace
* LangChain & FAISS open-source contributors
* Streamlit & ReportLab teams

---

## ✨ Made By

*Team Name:* *Team Rakshak*

*Team Members:*

* Vidhan Gauswami
* Virendra Badgotya

