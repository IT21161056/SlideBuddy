import os
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# ─────────────────────────────────────────────────────────────────────────────
# 1. API Key Setup
# ─────────────────────────────────────────────────────────────────────────────
api_key = "*****************************************"
genai.configure(api_key=api_key)
st.set_page_config(page_title="Gemini Chatbot", page_icon="💬")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Embeddings & Model
# ─────────────────────────────────────────────────────────────────────────────
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)
model = genai.GenerativeModel("gemini-1.5-flash")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Load and Chunk Documents with Improved Chunking
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_vector_store():
    # Add safety parameter to handle pickle deserialization
    if os.path.exists("my_vectorstore") and not st.sidebar.button("Reload Documents"):
        try:
            return FAISS.load_local("my_vectorstore", embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.warning(f"Error loading existing vector store: {e}")
            # If loading fails, delete the vectorstore and recreate it
            import shutil
            if os.path.exists("my_vectorstore"):
                shutil.rmtree("my_vectorstore")
            st.warning("Recreating vector store...")
    
    with st.spinner("Processing documents... This may take a minute."):
        documents = []
        if not os.path.exists("./documents"):
            os.makedirs("./documents")
            st.warning("Please add PDF documents to the 'documents' folder.")
            return FAISS.from_texts(["No documents available"], embeddings)
            
        for file in os.listdir("./documents"):
            if file.endswith(".pdf"):
                try:
                    loader = PyPDFLoader(os.path.join("./documents", file))
                    documents.extend(loader.load())
                    st.success(f"Loaded: {file}")
                except Exception as e:
                    st.error(f"Error loading {file}: {e}")

        if not documents:
            st.warning("No PDF documents found in the 'documents' folder.")
            return FAISS.from_texts(["No documents available"], embeddings)

        # Improved chunking strategy for better context preservation
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,       # Smaller chunks for more precise retrieval
            chunk_overlap=100,    # Decent overlap to maintain context
            separators=["\n\n", "\n", ". ", " ", ""],  # More natural text splits
            length_function=len
        )
        chunks = splitter.split_documents(documents)
        
        # Add information about document origin to each chunk
        for i, chunk in enumerate(chunks):
            if not chunk.metadata.get("source"):
                chunk.metadata["chunk_id"] = i
                
        st.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        store = FAISS.from_documents(chunks, embeddings)
        store.save_local("my_vectorstore")
        return store

# Add sidebar for configurations
st.sidebar.title("Configuration")
st.sidebar.markdown("### Retrieval Settings")
k_docs = st.sidebar.slider("Number of chunks to retrieve", 1, 10, 5)
score_threshold = st.sidebar.slider("Similarity threshold", 0.0, 1.0, 0.3)

# Load the vector store
vectorstore = load_vector_store()
retriever = vectorstore.as_retriever(
    search_kwargs={"k": k_docs}
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Ask Gemini with Improved RAG Prompting
# ─────────────────────────────────────────────────────────────────────────────
def ask_with_context(query):
    docs = retriever.get_relevant_documents(query)
    
    # If no relevant documents found
    if not docs:
        return "I couldn't find any relevant information in the documents to answer your question. Could you rephrase or ask something related to the documents' content?"
    
    # Format context with document references
    formatted_context = ""
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        source_info = f"{os.path.basename(source)} (Page {page})" if page else os.path.basename(source)
        formatted_context += f"\n\nDocument {i+1} [{source_info}]:\n{doc.page_content.strip()}"
    
    # Improved prompt with more specific instructions
    prompt = f"""You are an assistant specifically trained to provide accurate information based on the provided documents.

CONTEXT INFORMATION:
{formatted_context}

TASK:
Based only on the context information provided above, answer the following question: "{query}"

IMPORTANT INSTRUCTIONS:
1. Only use information present in the context documents.
2. If the context doesn't contain enough information to answer the question completely, say so clearly.
3. Do not make up information or use your general knowledge unless it directly confirms what's in the documents.
4. Cite your sources by mentioning which document(s) you got the information from.
5. Keep your answer concise and directly focused on the question."""

    generation_config = {
        "temperature": 0.2,  # Lower temperature for more factual responses
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 1024,
    }
    
    response = model.generate_content(
        prompt, 
        generation_config=generation_config
    )
    
    return response.text

# ─────────────────────────────────────────────────────────────────────────────
# 5. Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("💬 Gemini Chatbot with PDF Knowledge")
st.markdown("Ask anything based on your uploaded PDF documents!")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:", "")

if st.button("Send") and user_input:
    with st.spinner("Thinking..."):
        try:
            answer = ask_with_context(user_input)
            # Add both question and answer to history
            st.session_state.history.append(("You", user_input))
            st.session_state.history.append(("Bot", answer))
        except Exception as e:
            st.error(f"Error: {e}")

# Display chat history
for role, msg in st.session_state.history:
    if role == "You":
        st.markdown(f"**👤 {role}:** {msg}")
    else:
        st.markdown(f"**🤖 {role}:** {msg}")

# Option to clear chat history
if st.button("Clear Chat History"):
    st.session_state.history = []
    st.experimental_rerun()
