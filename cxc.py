# cxc file for faiss index
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load Excel into structured text docs
df = pd.read_excel("GPT_Input_DB.xlsx")
docs = [Document(page_content=" | ".join(map(str, row))) for _, row in df.iterrows()]

# Embed and store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = FAISS.from_documents(docs, embeddings)
db.save_local("index")
print("✅ FAISS index saved as 'index/'")

