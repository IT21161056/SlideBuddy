# 💬 SlideBuddy — PDF-Powered Chatbot with Gemini AI

SlideBuddy is an intelligent conversational chatbot that lets you **ask questions directly from your PDF documents**. It combines the power of Google Gemini 1.5 Flash with a Retrieval-Augmented Generation (RAG) pipeline to deliver accurate, source-cited answers grounded in your own uploaded content.

---

## ✨ Features

- 📄 **PDF Knowledge Base** — Load any number of PDF files into a local vector store and query them in natural language
- 🧠 **RAG Pipeline** — Uses FAISS vector search + LangChain to retrieve the most relevant document chunks before answering
- 🤖 **Gemini 1.5 Flash** — Google's fast generative model provides concise, factual responses
- 🔍 **Source Citations** — Every answer references the document(s) and page(s) it drew from
- 🎛️ **Configurable Retrieval** — Adjust the number of retrieved chunks and similarity threshold via the sidebar
- 💾 **Persistent Vector Store** — FAISS index is saved locally so documents only need to be processed once
- 🖥️ **Streamlit UI** — Clean, interactive chat interface with session history

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | [Streamlit](https://streamlit.io/) |
| LLM | Google Gemini 1.5 Flash (`google-generativeai`) |
| Embeddings | Google Generative AI Embeddings (`embedding-001`) |
| Vector Store | FAISS (`faiss-cpu`) |
| RAG Framework | LangChain + LangChain Community |
| PDF Loader | PyPDF (`pypdf`) |
| Language | Python 3.9+ |

---

## 📁 Project Structure

```
SlideBuddy/
├── chatbot.py          # Main Streamlit application
├── notebook.ipynb      # Jupyter notebook for experimentation
├── requirements.txt    # Python dependencies
├── documents/          # 📂 Place your PDF files here (auto-created)
└── my_vectorstore/     # FAISS index (auto-generated on first run)
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/IT21161056/SlideBuddy.git
cd SlideBuddy
```

### 2. Install Dependencies

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set Up Your Google API Key

Open `chatbot.py` and replace the placeholder API key with your own Google Gemini API key:

```python
api_key = "YOUR_GOOGLE_API_KEY_HERE"
```

> **Tip:** For better security, use an environment variable instead of hardcoding the key:
> ```python
> import os
> api_key = os.environ.get("GOOGLE_API_KEY")
> ```
> Then run: `export GOOGLE_API_KEY=your_key_here`

You can obtain a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

### 4. Add Your PDF Documents

Create a `documents/` folder in the project root and place your PDF files inside it:

```bash
mkdir documents
cp your_document.pdf documents/
```

### 5. Run the App

```bash
streamlit run chatbot.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🧪 How It Works

<img width="1440" height="1160" alt="image" src="https://github.com/user-attachments/assets/975f2a5d-2b5a-4630-be34-818b921b1b60" />


1. **Ingestion** — PDFs in the `documents/` folder are loaded, split into overlapping 500-character chunks, embedded using Google's `embedding-001` model, and stored in a local FAISS index.
2. **Retrieval** — When you ask a question, the top-k most semantically similar chunks are retrieved from the index.
3. **Generation** — The retrieved chunks are injected into a structured prompt, and Gemini 1.5 Flash generates a cited, fact-grounded response.

---

## ⚙️ Configuration

Use the **sidebar** in the app to tune retrieval behaviour at runtime:

| Setting | Default | Description |
|---|---|---|
| Number of chunks to retrieve | 5 | How many document chunks are passed to the model |
| Similarity threshold | 0.3 | Minimum relevance score for a chunk to be included |

To force re-processing of documents (e.g. after adding new PDFs), click **"Reload Documents"** in the sidebar.

---

## 📦 Dependencies

```
streamlit==1.30.0
google-generativeai==0.3.2
langchain==0.1.11
langchain-google-genai==0.0.7
langchain-community==0.0.19
langchain-core==0.1.28
langchain-text-splitters==0.0.1
faiss-cpu==1.7.4
pypdf==3.17.1
```

Install them all with:

```bash
pip install -r requirements.txt
```

---

## ⚠️ Important Notes

- **API Key Security** — Never commit your Google API key to a public repository. Use environment variables or a secrets manager in production.
- **Document Freshness** — The FAISS index is cached. After adding or removing PDFs, use the "Reload Documents" button or delete the `my_vectorstore/` folder and restart the app.
- **PDF Quality** — Results are only as good as the content in your PDFs. Scanned image-only PDFs will not be readable without OCR pre-processing.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for bug fixes, new features, or improvements.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source. Please check the repository for license details.

---

## 👤 Author

**IT21161056** — [GitHub Profile](https://github.com/IT21161056)
