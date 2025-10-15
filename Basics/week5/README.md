                ┌──────────────────────────────────────────┐
                │           ExpertKnowledgeWorker          │
                └──────────────────────────────────────────┘
                                   │
                                   ▼
             ┌────────────────────────────────────┐
             │       1️⃣ Load Knowledge Base       │
             │ - Scans `knowledge-base/` folders   │
             │ - Loads `.md` files using DirectoryLoader │
             └────────────────────────────────────┘
                                   │
                                   ▼
             ┌────────────────────────────────────┐
             │   2️⃣ Split Documents into Chunks    │
             │ - Uses CharacterTextSplitter        │
             │ - chunk_size=1000, overlap=200      │
             └────────────────────────────────────┘
                                   │
                                   ▼
             ┌────────────────────────────────────┐
             │   3️⃣ Embed Text Chunks              │
             │ - Model: BAAI/bge-small-en          │
             │ - Normalized embeddings             │
             └────────────────────────────────────┘
                                   │
                                   ▼
             ┌────────────────────────────────────┐
             │   4️⃣ Store in Chroma Vector DB      │
             │ - Persistent local DB (vector_db)   │
             │ - Indexed for semantic retrieval    │
             └────────────────────────────────────┘
                                   │
                                   ▼
             ┌────────────────────────────────────┐
             │   5️⃣ Initialize LLM (Groq Llama3)   │
             │ - Model: llama-3.1-8b-instant       │
             │ - Low-latency via Groq API          │
             └────────────────────────────────────┘
                                   │
                                   ▼
             ┌────────────────────────────────────┐
             │   6️⃣ Build RAG Chain (LangChain)    │
             │ - ConversationalRetrievalChain      │
             │ - Includes retriever + memory       │
             └────────────────────────────────────┘
                                   │
                                   ▼
             ┌────────────────────────────────────┐
             │   7️⃣ Answer User Questions          │
             │ - query() → Single response          │
             │ - stream_answer() → Live streaming   │
             └────────────────────────────────────┘
                                   │
                                   ▼
             ┌────────────────────────────────────┐
             │   8️⃣ UI Options                    │
             │ - Gradio Chat Interface             │
             │ - Command-line menu                 │
             └────────────────────────────────────┘
