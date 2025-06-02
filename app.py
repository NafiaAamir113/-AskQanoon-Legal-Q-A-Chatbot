import streamlit as st
import requests
import pinecone
from sentence_transformers import SentenceTransformer

# Streamlit page setup
st.set_page_config(page_title="AskQanoon - Legal Assistant", layout="wide")
st.title("⚖️ AskQanoon - Legal Assistant Chatbot")
st.markdown("Ask legal questions and get AI-powered assistance.")

# Load secrets from .streamlit/secrets.toml
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
INDEX_NAME = st.secrets["INDEX_NAME"]
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]

# Initialize Pinecone v3 client
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Check if the index exists
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    st.error(f"Index '{INDEX_NAME}' not found.")
    st.stop()

# Connect to the index
index = pc.Index(INDEX_NAME)

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')  # or your own fine-tuned model

# Input from user
query = st.text_input("Enter your legal question:")

if st.button("Ask"):
    if query.strip() == "":
        st.warning("Please enter a legal question.")
    else:
        # Step 1: Generate embedding
        query_embedding = embed_model.encode(query).tolist()

        # Step 2: Query Pinecone index
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

        context = ""
        for match in results["matches"]:
            context += match["metadata"].get("text", "") + "\n"

        # Step 3: Prompt LLaMA model
        prompt = f"""You are a legal assistant chatbot. Use the context below to answer the question.

Context:
{context}

Question: {query}
Answer:"""

        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "togethercomputer/llama-2-13b-chat",
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.7,
            "stop": ["User:", "Assistant:"]
        }

        response = requests.post("https://api.together.xyz/v1/completions", headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            answer = result.get("choices", [{}])[0].get("text", "No answer found.")
            st.success("✅ Response:")
            st.write(answer.strip())
        else:
            st.error("❌ Failed to fetch response from Together AI API.")
