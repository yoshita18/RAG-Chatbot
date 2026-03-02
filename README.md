# 💬 RAG Chatbot (PDF + Vector Search)

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDF documents and ask natural language questions about them. Answers are grounded strictly in your uploaded documents — no hallucinations, but cross-document reasoning is a known limitation currently being worked on.

---

## 🚀 Demo

| ✅ What it does well | ❌ What it can't do yet |
|---|---|
| Zero hallucinations — says "I don't know" when unsure | Cross-document reasoning (e.g. comparing two resumes) |
| Semantic search across large PDFs | Abstract or analytical questions |
| Cites page numbers in answers | Understanding document-level context |

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| LLM | Google Gemini (`gemini-2.0-flash`) |
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| Vector Store | FAISS |
| Orchestration | LangChain |
| PDF Parsing | PyPDFLoader |
| Text Splitting | RecursiveCharacterTextSplitter |

---

## ⚙️ How It Works

1. Upload one or more PDFs
2. Documents are split into overlapping chunks and embedded into a FAISS vector store
3. When you ask a question, the query is embedded and the most semantically similar chunks are retrieved
4. Retrieved chunks are injected into the prompt along with conversation history
5. Gemini answers using only the provided context

---

## 📦 Installation

**1. Clone the repo**
```bash
git clone https://github.com/your-username/llm-with-rag.git
cd llm-with-rag
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set your Google API key**
```bash
export GOOGLE_API_KEY='your_key_here'
```
Or create a `.env` file:
```
GOOGLE_API_KEY=your_key_here
```

**5. Run the app**
```bash
streamlit run app.py
```

---

## 🎛️ Configurable Settings

All settings are adjustable via the sidebar in the UI:

| Setting | Default | Description |
|---|---|---|
| Chat model | `gemini-2.0-flash` | Gemini model for answering |
| Embedding model | `gemini-embedding-001` | Model for vectorizing chunks |
| Temperature | `0.2` | Controls response creativity |
| Chunk size | `1000` | Characters per text chunk |
| Chunk overlap | `150` | Overlap between chunks |
| Top K | `4` | Number of chunks retrieved per query |

---

## 📁 Project Structure
```
llm-with-rag/
├── app.py              # Main application
├── requirements.txt    # Dependencies
├── .gitignore
└── README.md
```

---

## 🔭 Roadmap

- [ ] Hybrid search (semantic + keyword)
- [ ] Agentic RAG for multi-step reasoning
- [ ] Cross-document comparison and analysis
- [ ] Better chunking strategies (semantic chunking)
- [ ] GraphRAG for relationship-aware retrieval
- [ ] Persistent vector store (no rebuild on refresh)

---

## ⚠️ Known Limitations

- **No reasoning:** The model retrieves and repeats — it cannot compare, evaluate, or synthesize across documents. Asking "which resume is better?" will return no answer even though the content exists in the files.
- **Context sensitivity:** Vague queries can retrieve irrelevant chunks (e.g. asking "what is the PDF about?" in a statistics-heavy book returned the probability density function definition).
- **Context cap:** Retrieved content is capped at 12,000 characters per query to avoid oversized prompts.

---

## 📄 License

MIT
