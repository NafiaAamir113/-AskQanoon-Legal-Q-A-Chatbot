import streamlit as st
import requests
import pinecone
from sentence_transformers import SentenceTransformer, CrossEncoder

# Streamlit page setup
st.set_page_config(page_title="ASKQANOON", layout="wide")

# Load secrets from Streamlit secrets management
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
TOGETHER_AI_API_KEY = st.secrets["TOGETHER_AI_API_KEY"]

# Pinecone setup
INDEX_NAME = "lawdata-index"
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    st.error(f"Index '{INDEX_NAME}' not found.")
    st.stop()

# Initialize Pinecone index
index = pc.Index(INDEX_NAME)

# Load embedding models
embedding_model = SentenceTransformer("BAAI/bge-large-en")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Page Title and Description
st.title("⚖️ LEGAL ASSISTANT")
st.markdown("This AI-powered legal assistant retrieves relevant legal documents and provides accurate responses to your legal queries.")

# Input field
query = st.text_input("Enter your legal question:")

# Generate Answer Button
if st.button("Generate Answer"):
    if not query:
        st.warning("Please enter a legal question before generating an answer.")
        st.stop()

    if len(query.split()) < 4:  # simple heuristic for incomplete queries
        st.warning("Your query seems incomplete. Please provide more details.")
        st.stop()

    with st.spinner("Searching for relevant documents..."):
        try:
            # Create query embedding
            query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()

            # Query Pinecone index
            search_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        except Exception as e:
            st.error(f"Pinecone query failed: {e}")
            st.stop()

        if not search_results or "matches" not in search_results or not search_results["matches"]:
            st.warning("No relevant results found. Try rephrasing your query.")
            st.stop()

        # Extract text from results
        context_chunks = [match["metadata"]["text"] for match in search_results["matches"]]

        # Rerank results with CrossEncoder
        rerank_scores = reranker.predict([(query, chunk) for chunk in context_chunks])
        ranked_results = sorted(zip(context_chunks, rerank_scores), key=lambda x: x[1], reverse=True)

        # Take top 5 chunks (or fewer if less results)
        num_chunks = min(len(ranked_results), 5)
        context_text = "\n\n".join([r[0] for r in ranked_results[:num_chunks]])

        # Construct prompt for the LLM
        prompt = f"""You are a legal assistant. Based on the following legal documents, provide a detailed and clear answer.

Context:
{context_text}

Question: {query}

Answer:"""

        # Call Together AI Completions API
        response = requests.post(
            "https://api.together.xyz/v1/completions",
            headers={
                "Authorization": f"Bearer {TOGETHER_AI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "prompt": prompt,
                "temperature": 0.2,
                "max_tokens": 512,
                "stop": ["\n\n"]
            }
        )

        if response.status_code == 200:
            result = response.json()
            answer = result.get("choices", [{}])[0].get("text", "").strip()
            if not answer:
                st.error("Received empty response from AI model.")
            else:
                st.success("AI Response:")
                st.write(answer)
        else:
            st.error(f"Failed to fetch response from Together AI API. Status code: {response.status_code}")
            st.text(response.text)

# Footer
st.markdown("<p style='text-align: center;'>🚀 Built with Streamlit</p>", unsafe_allow_html=True)
