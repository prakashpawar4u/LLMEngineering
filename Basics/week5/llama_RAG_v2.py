# expert_knowledge_worker_v2.py
"""
ExpertKnowledgeWorker v2
- Semantic sectioning (markdown headings, paragraph based)
- Deduplication & cleaning (boilerplate removal)
- Incremental ingestion (manifest.json)
- BM25 lexical index + Chroma embedding store (hybrid search)
- Hybrid search: BM25 (fast keyword) + semantic embeddings (deep understanding)
- Local LLM (llama_cpp) or remote Groq LLM option
- Async streaming responses for integration with Gradio/HTTP
- Feedback loop (store labeled QA pairs for re-ranking / future fine-tune)
"""

import os
import json
import glob
import time
import hashlib
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional, Tuple

# External libs (install as required)
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Optional LLM clients:
try:
    from langchain_groq import ChatGroq
    import groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

try:
    # local Llama via llama-cpp-python (optional, fast local inference)
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except Exception:
    LLAMA_CPP_AVAILABLE = False

# BM25 lexical search
try:
    from rank_bm25 import BM25Okapi
    RANK_BM25_AVAILABLE = True
except Exception:
    RANK_BM25_AVAILABLE = False

# Text loaders / parsing
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter

# For async streaming we will use asyncio; Gradio will be used for UI (async-friendly)
import gradio as gr

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("ExpertKWv2")


# -------------------------
# Utilities
# -------------------------
def md5_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def read_manifest(manifest_path: str) -> Dict[str, Any]:
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def write_manifest(manifest_path: str, manifest: Dict[str, Any]) -> None:
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# -------------------------
# Text cleaning & semantic sectioning
# -------------------------
BOILERPLATE_PATTERNS = [
    r"^\s*#\s*Table of Contents[\s\S]*",  # simple header-based removal
    r"^\s*Copyright.*$",
    r"^\s*All rights reserved.*$",
]

import re
def clean_text(text: str) -> str:
    """Remove repeated boilerplate and normalize whitespace."""
    # remove common repeated header/footer lines (very simple heuristics)
    lines = text.splitlines()
    filtered = []
    for line in lines:
        if len(line.strip()) == 0:
            filtered.append("")  # keep paragraph breaks
            continue
        # remove lines that look like page numbers or "Page X of Y"
        if re.match(r"^page\s*\d+(\s*of\s*\d+)?$", line.strip().lower()):
            continue
        filtered.append(line)
    cleaned = "\n".join(filtered)
    # Remove other boilerplate via patterns
    for p in BOILERPLATE_PATTERNS:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    # collapse multiple blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def semantic_section_split_markdown(text: str, min_section_words: int = 40) -> List[Dict[str, Any]]:
    """
    Split markdown documents into semantic sections using headers.
    Returns list of dicts: {'title': str, 'content': str}
    """
    # Strategy:
    # - Use markdown headers (#, ##, ###) as section boundaries
    # - If a top-level header is large, allow subheaders
    # - Fallback: split paragraphs into sections of min_section_words
    sections = []
    lines = text.splitlines()
    current_title = "root"
    current_buf = []
    header_pattern = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)")
    for ln in lines:
        m = header_pattern.match(ln)
        if m:
            # flush previous
            if current_buf:
                content = "\n".join(current_buf).strip()
                if content:
                    sections.append({"title": current_title, "content": content})
            current_title = m.group(2).strip() or "untitled"
            current_buf = []
        else:
            current_buf.append(ln)
    if current_buf:
        content = "\n".join(current_buf).strip()
        if content:
            sections.append({"title": current_title, "content": content})

    # merge tiny sections with previous
    merged = []
    for sec in sections:
        word_count = len(sec["content"].split())
        if merged and word_count < min_section_words:
            merged[-1]["content"] += "\n\n" + sec["content"]
        else:
            merged.append(sec)
    return merged


# -------------------------
# Core class
# -------------------------
class ExpertKnowledgeWorkerV2:
    def __init__(
        self,
        knowledge_base_path: str = "knowledge-base",
        db_dir: str = "vector_db_v2",
        manifest_path: str = "vector_db_v2/manifest.json",
        feedback_path: str = "vector_db_v2/feedback.json",
        embedding_model: str = "BAAI/bge-small-en",
        use_gpu: bool = False,
        local_llm: bool = False,
        local_llm_path: Optional[str] = None,
        groq_model: str = "llama-3.1-8b-instant",
        hybrid_weight: float = 0.5,  # weight for semantic (0..1). BM25 gets 1-weight
        bm25_topk: int = 10,
        semantic_topk: int = 10,
    ):
        load_dotenv(override=True)
        self.knowledge_base_path = knowledge_base_path
        self.db_dir = db_dir
        self.manifest_path = manifest_path
        self.feedback_path = feedback_path
        self.embedding_model = embedding_model
        self.use_gpu = use_gpu
        self.local_llm = local_llm and LLAMA_CPP_AVAILABLE
        self.local_llm_path = local_llm_path
        self.groq_model = groq_model
        self.hybrid_weight = float(hybrid_weight)
        self.bm25_topk = bm25_topk
        self.semantic_topk = semantic_topk

        # Load keys
        self.groq_api_key = os.getenv("GROQ_API_KEY", None)

        # Initialize components
        self._init_embeddings()
        self._init_vectorstore()
        self._load_manifest()
        self.bm25_index = None
        self.bm25_corpus_docs = []  # textual corpus for BM25
        self._init_llm()
        self._build_retriever_chain()
        # Async lock for ingestion to avoid races
        self._ingest_lock = asyncio.Lock()

        # Load feedback store
        self._load_feedback_store()

    # -------------------------
    # Embeddings & vectorstore
    # -------------------------
    def _init_embeddings(self):
        device = "cuda" if (self.use_gpu and os.environ.get("CUDA_VISIBLE_DEVICES")) else "cpu"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info(f"Embeddings initialized on device {device}")

    def _init_vectorstore(self):
        # create or open Chroma directory
        self.vectorstore = Chroma(persist_directory=self.db_dir, embedding_function=self.embeddings)
        # vectorstore could be empty if first run

    # -------------------------
    # Manifest (for incremental ingestion)
    # -------------------------
    def _load_manifest(self):
        self.manifest = read_manifest(self.manifest_path)
        # manifest maps file_path -> { "md5": "<md5>", "mtime": float, "chunks": [ids], "updated_at": ts }
        if not self.manifest:
            self.manifest = {}
        logger.info(f"Loaded manifest with {len(self.manifest)} entries")

    # -------------------------
    # Feedback store
    # -------------------------
    def _load_feedback_store(self):
        if os.path.exists(self.feedback_path):
            with open(self.feedback_path, "r", encoding="utf-8") as f:
                self.feedback_store = json.load(f)
        else:
            self.feedback_store = {"items": []}
        logger.info(f"Feedback store loaded: {len(self.feedback_store.get('items', []))} items")

    def _save_feedback_store(self):
        Path(self.feedback_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.feedback_path, "w", encoding="utf-8") as f:
            json.dump(self.feedback_store, f, indent=2)

    def add_feedback(self, question: str, answer: str, label: str, context: Optional[str] = None):
        """
        label: 'positive'|'negative'|'neutral' or custom tags
        """
        item = {"ts": time.time(), "question": question, "answer": answer, "label": label, "context": context}
        self.feedback_store.setdefault("items", []).append(item)
        self._save_feedback_store()
        logger.info("Feedback recorded")

    # -------------------------
    # LLM init (Groq remote or local Llama)
    # -------------------------
    def _init_llm(self):
        self.llm_client = None
        if self.local_llm:
            if not LLAMA_CPP_AVAILABLE:
                raise RuntimeError("llama-cpp-python not available, cannot use local_llm flag")
            if not self.local_llm_path:
                raise ValueError("local_llm_path must point to a local LLAMA .bin model")
            # Initialize local Llama client (llama-cpp-python)
            self.llm_client = Llama(model_path=self.local_llm_path, n_ctx=2048)
            logger.info("Local Llama (llama-cpp) initialized")
        else:
            if GROQ_AVAILABLE and self.groq_api_key:
                # LangChain wrapper for Groq (ChatGroq)
                self.llm_client = ChatGroq(temperature=0.0, model_name=self.groq_model, groq_api_key=self.groq_api_key)
                # Also keep groq client for streaming
                try:
                    import groq as _groq
                    self._groq_client = _groq.Groq(api_key=self.groq_api_key)
                except Exception:
                    self._groq_client = None
                logger.info("Groq ChatGroq client initialized")
            else:
                logger.warning("No remote Groq configured and local Llama disabled. LLM calls will error.")

    # -------------------------
    # Build RAG chain placeholder (we'll create a LangChain conversational retriever)
    # -------------------------
    def _build_retriever_chain(self):
        # conversational chain will be (optionally) used for query() function
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.semantic_topk})
        # Use LangChain's ConversationalRetrievalChain if llm_client supports LangChain interface (ChatGroq does)
        try:
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm_client, retriever=retriever, memory=memory
            )
            logger.info("ConversationalRetrievalChain created")
        except Exception as e:
            logger.warning(f"ConversationalRetrievalChain not created: {e}")
            self.conversation_chain = None

    # -------------------------
    # Ingestion (incremental)
    # -------------------------
    def _collect_source_files(self) -> List[str]:
        """Collect doc files from knowledge_base_path - supports .md and .txt for now"""
        patterns = ["**/*.md", "**/*.txt"]
        files = []
        for p in patterns:
            files.extend([str(Path(pth)) for pth in Path(self.knowledge_base_path).glob(p)])
        return sorted(files)

    async def incremental_ingest(self, force_rebuild_bm25: bool = False):
        """
        Ingest new or changed files only.
        - Read files
        - Clean text
        - Semantic section split
        - Deduplicate (by md5 of section)
        - Add new chunks to chroma vectorstore
        - Update manifest
        - Rebuild BM25 corpus if new docs added or force_rebuild_bm25=True
        """
        async with self._ingest_lock:
            files = self._collect_source_files()
            new_chunks = []
            changed = False

            for fpath in files:
                try:
                    stat = os.stat(fpath)
                    mtime = stat.st_mtime
                    with open(fpath, "r", encoding="utf-8") as fh:
                        raw = fh.read()
                except Exception as e:
                    logger.warning(f"Failed to read {fpath}: {e}")
                    continue

                md5 = md5_text(raw)
                prev = self.manifest.get(fpath)
                if prev and prev.get("md5") == md5 and prev.get("mtime") == mtime:
                    # unchanged: skip
                    continue

                # changed/new
                cleaned = clean_text(raw)
                sections = semantic_section_split_markdown(cleaned)
                # build Document-like objects for Chroma (langchain Document)
                docs_for_store = []
                for sec in sections:
                    sec_text = f"# {sec['title']}\n\n{sec['content']}"
                    sec_md5 = md5_text(sec_text)
                    # avoid re-adding same section if previously in manifest (dedupe by md5 across all files)
                    if any(prev_info.get("sections", {}).get(sec_md5) for prev_info in self.manifest.values()):
                        logger.debug("Skipping duplicate section by md5")
                        continue
                    doc = Document(page_content=sec_text, metadata={"source": fpath, "section_md5": sec_md5, "title": sec['title']})
                    docs_for_store.append(doc)

                if docs_for_store:
                    # add to vectorstore in a batch (Chroma.from_documents supports append via add_documents or using collection API)
                    # Note: langchain-chroma may not expose add_documents directly; here we use from_documents with persist_directory to append
                    try:
                        # Append semantics: create a temporary Chroma and merge or directly use vectorstore.add_documents
                        # We'll try vectorstore.add_documents first (Chroma exposes "add_documents" on some adapters)
                        if hasattr(self.vectorstore, "add_documents"):
                            self.vectorstore.add_documents(docs_for_store, embedding=self.embeddings)
                        else:
                            # fallback: use from_documents and rely on persist_directory; this may re-create collection
                            Chroma.from_documents(documents=docs_for_store, embedding=self.embeddings, persist_directory=self.db_dir)
                        logger.info(f"Added {len(docs_for_store)} new sections from {fpath}")
                    except Exception as e:
                        logger.error(f"Failed to add documents for {fpath}: {e}")
                        continue

                    # record in manifest
                    self.manifest[fpath] = {
                        "md5": md5,
                        "mtime": mtime,
                        "ts": time.time(),
                        "sections": {doc.metadata["section_md5"]: {"title": doc.metadata["title"]} for doc in docs_for_store},
                    }
                    changed = True

            if changed:
                write_manifest(self.manifest_path, self.manifest)
                logger.info("Manifest updated")
                # rebuild retriever chain to ensure indexes are current
                self._init_vectorstore()
                self._build_retriever_chain()

            # Rebuild BM25 corpus if new docs added or forced
            if changed or force_rebuild_bm25 or (self.bm25_index is None):
                await self._rebuild_bm25()
                logger.info("BM25 index rebuilt")

    async def _rebuild_bm25(self):
        if not RANK_BM25_AVAILABLE:
            logger.warning("rank_bm25 not available - skipping BM25 build")
            self.bm25_index = None
            return

        docs = []
        # Extract textual corpus from Chroma: use vectorstore._collection.get to retrieve documents (implementation-specific)
        try:
            # Attempt to use the collection API
            collection = self.vectorstore._collection
            # Some Chroma adapters permit get()["documents"] - cannot rely on exact API across versions.
            # We'll rely on similarity_search to pull all docs by doing a naive approach: this is best-effort.
            # Instead of pulling all, we can reconstruct BM25 from manifest sections content:
            for fpath, info in self.manifest.items():
                for sec_md5 in info.get("sections", {}):
                    # retrieve by metadata? We'll store simple text corpus mapping in manifest in future runs.
                    # For now, read back the file and re-split to extract text for BM25 build.
                    try:
                        with open(fpath, "r", encoding="utf-8") as fh:
                            raw = fh.read()
                        cleaned = clean_text(raw)
                        sections = semantic_section_split_markdown(cleaned)
                        for sec in sections:
                            docs.append(sec["content"])
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"BM25 rebuild: couldn't access vectorstore collection directly: {e}")
            # fallback: rebuild corpus by reading all source files
            for fpath in self._collect_source_files():
                try:
                    with open(fpath, "r", encoding="utf-8") as fh:
                        raw = fh.read()
                    cleaned = clean_text(raw)
                    sections = semantic_section_split_markdown(cleaned)
                    for sec in sections:
                        docs.append(sec["content"])
                except Exception:
                    continue

        # Preprocess docs for BM25 (tokenization simple)
        tokenized = [doc.split() for doc in docs]
        if tokenized:
            self.bm25_index = BM25Okapi(tokenized)
            self.bm25_corpus_docs = docs
        else:
            self.bm25_index = None
            self.bm25_corpus_docs = []

    # -------------------------
    # Hybrid search: BM25 lexical + semantic embeddings
    # -------------------------
    async def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Return combined candidates with scores and metadata.
        Steps:
        1) BM25 top N (fast lexical candidates)
        2) Embedding similarity via Chroma retriever top M
        3) Merge candidates, normalize scores, weighted sum by self.hybrid_weight
        """
        bm25_hits = []
        semantic_hits = []

        # BM25 candidates
        if self.bm25_index:
            token_q = query.split()
            bm25_scores = self.bm25_index.get_scores(token_q)
            top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[: self.bm25_topk]
            for idx in top_indices:
                bm25_hits.append({"source": "bm25", "doc_idx": idx, "score": float(bm25_scores[idx]), "text": self.bm25_corpus_docs[idx]})

        # semantic candidates
        try:
            semantic_docs = self.vectorstore.similarity_search(query, k=self.semantic_topk)
            # depending on API, semantic_docs contain page_content and metadata
            for d in semantic_docs:
                # assume retriever returns a distance or score attribute? not always present
                semantic_hits.append({"source": "semantic", "text": d.page_content, "score": None, "metadata": d.metadata})
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")

        # Normalize BM25 scores
        bm25_scores_list = [h["score"] for h in bm25_hits] if bm25_hits else []
        if bm25_scores_list:
            max_b = max(bm25_scores_list)
            min_b = min(bm25_scores_list)
        else:
            max_b = min_b = 0.0

        # For semantic hits, compute embedding similarity score using embedding model to get numeric similarity
        semantic_scores = []
        if semantic_hits:
            # embed query and compute cos similarity against hits (naive approach using embeddings.encode)
            q_emb = self.embeddings.embed_query(query)
            # vectorstore might provide L2/distance; as fallback compute similarity using chroma collection's similarity API (not exposed uniformly)
            for h in semantic_hits:
                try:
                    doc_emb = self.embeddings.embed_query(h["text"])  # suboptimal: re-embed text, but works
                    # cosine similarity
                    import numpy as np
                    qv = np.array(q_emb, dtype=float)
                    dv = np.array(doc_emb, dtype=float)
                    sim = float((qv @ dv) / (np.linalg.norm(qv) * np.linalg.norm(dv) + 1e-9))
                    h["score"] = sim
                except Exception:
                    h["score"] = 0.0

        # Now normalize scores to 0..1
        def normalize_scores(items):
            scores = [i["score"] for i in items if i["score"] is not None]
            if not scores:
                return items
            maxs = max(scores)
            mins = min(scores)
            rng = maxs - mins if maxs - mins > 0 else 1.0
            for i in items:
                if i["score"] is None:
                    i["norm_score"] = 0.0
                else:
                    i["norm_score"] = (i["score"] - mins) / rng
            return items

        bm25_norm = []
        if bm25_hits:
            # normalize bm25 by min/max
            if max_b != min_b:
                for h in bm25_hits:
                    h["norm_score"] = (h["score"] - min_b) / (max_b - min_b)
            else:
                for h in bm25_hits:
                    h["norm_score"] = 1.0
            bm25_norm = bm25_hits

        sem_norm = normalize_scores(semantic_hits)

        # Merge candidates by text content (de-duplicate similar texts) and compute hybrid score
        merged = {}
        for h in (bm25_norm or []):
            key = h["text"][:400]
            merged.setdefault(key, {"text": h["text"], "bm25": h.get("norm_score", 0.0), "semantic": 0.0})
        for h in (sem_norm or []):
            key = h["text"][:400]
            merged.setdefault(key, {"text": h["text"], "bm25": 0.0, "semantic": h.get("norm_score", 0.0), "metadata": h.get("metadata")})
            # if already present, update semantic
            merged[key]["semantic"] = max(merged[key].get("semantic", 0.0), h.get("norm_score", 0.0))

        # Calculate final score
        results = []
        for k, v in merged.items():
            final_score = self.hybrid_weight * v.get("semantic", 0.0) + (1.0 - self.hybrid_weight) * v.get("bm25", 0.0)
            results.append({"text": v["text"], "score": float(final_score), "bm25": float(v.get("bm25", 0.0)), "semantic": float(v.get("semantic", 0.0)), "metadata": v.get("metadata")})

        # sort by score
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        return results

    # -------------------------
    # Query methods & streaming
    # -------------------------
    async def query(self, question: str) -> str:
        """
        Simple synchronous-style query using conversation_chain if available or hybrid_search->LLM.
        """
        # Prefer LangChain conversation chain if available
        try:
            if self.conversation_chain:
                res = self.conversation_chain({"question": question})
                return res.get("answer", "")
        except Exception as e:
            logger.warning(f"Conversational chain failed: {e}")

        # Fallback: hybrid search to build context, then call LLM for single-shot answer
        candidates = await self.hybrid_search(question, top_k=5)
        context = "\n\n---\n\n".join([c["text"] for c in candidates])
        prompt = f"You are a helpful assistant. Use the context below to answer carefully. If context doesn't contain answer, say so.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        # call LLM (sync local or remote)
        if self.local_llm:
            out = self.llm_client.create(prompt=prompt, max_tokens=512)
            return out.get("choices", [{}])[0].get("text", "") if isinstance(out, dict) else ""
        else:
            if self.llm_client:
                resp = self.llm_client(prompt)  # ChatGroq wrapper might differ - assume it returns text
                # adapt for ChatGroq interface
                if isinstance(resp, dict) and "text" in resp:
                    return resp["text"]
                elif hasattr(resp, "content"):
                    return resp.content
                else:
                    return str(resp)
            else:
                raise RuntimeError("No LLM client available")

    async def stream_answer(self, question: str):
        """
        Async generator yielding strings (chunks) for streaming.
        - Builds context via hybrid_search
        - Streams from local Llama (if available) or Groq streaming API
        """
        candidates = await self.hybrid_search(question, top_k=5)
        context = "\n\n---\n\n".join([c["text"] for c in candidates])
        system = "You are a helpful assistant for answering company knowledge base questions. Use the provided context and cite sources by filename when applicable."
        user_message = f"Context:\n{context}\n\nQuestion: {question}"

        # If local llama-cpp, stream via its streaming callback (llama-cpp supports streaming)
        if self.local_llm:
            # llama-cpp streaming example - using callback
            def token_callback(token: str):
                # this callback runs synchronously; in async env, we push into an asyncio.Queue (not shown)
                pass
            # note: llama-cpp's exact API usage may vary; here is a generic call
            out = self.llm_client.create(prompt=system + "\n" + user_message, stream=False, max_tokens=512)
            # not streaming - yield full
            yield out.get("choices", [{}])[0].get("text", "")
            return

        # else use Groq streaming (if available)
        if GROQ_AVAILABLE and getattr(self, "_groq_client", None):
            stream = self._groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user_message}],
                stream=True,
            )
            # stream yields partial chunks
            full = ""
            for chunk in stream:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None) or (delta.get("content") if isinstance(delta, dict) else None)
                if content:
                    full += content
                    yield content
            return
        # fallback: call synchronous endpoint and stream characters
        out = await self.query(question)
        for i in range(0, len(out), 256):
            await asyncio.sleep(0)  # allow event loop to schedule
            yield out[i : i + 256]

    # -------------------------
    # Gradio interface helper
    # -------------------------
    def gradio_chat_fn(self, user_message: str, chat_history: List[Tuple[str, str]]):
        """
        This wrapper can be plugged into Gradio. Not async (Gradio supports async functions too).
        """
        # run query synchronously in event loop
        resp = asyncio.run(self.query(user_message))
        # append to chat history and return
        chat_history = chat_history or []
        chat_history.append((user_message, resp))
        return "", chat_history

    async def gradio_streaming_chat(self, user_message: str):
        """
        Async Gradio streaming endpoint. Yields messages to the UI.
        """
        async for chunk in self.stream_answer(user_message):
            yield chunk

    # -------------------------
    # Utility: reset collections, drop manifest
    # -------------------------
    def reset_store(self, drop_manifest: bool = False):
        # careful destructive operation
        try:
            self.vectorstore._client.reset()  # may or may not exist depending on chroma version
        except Exception:
            # fallback: remove directory
            import shutil

            try:
                shutil.rmtree(self.db_dir)
            except Exception:
                logger.exception("Failed to delete db dir")
        if drop_manifest and os.path.exists(self.manifest_path):
            os.remove(self.manifest_path)
        logger.info("Vector store reset (best-effort).")

# -------------------------
# Example usage / main
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", default="knowledge-base")
    parser.add_argument("--db", default="vector_db_v2")
    parser.add_argument("--local-llm", action="store_true", help="Use local llama-cpp model")
    parser.add_argument("--local-llm-path", type=str, default=None)
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    worker = ExpertKnowledgeWorkerV2(
        knowledge_base_path=args.kb,
        db_dir=args.db,
        use_gpu=args.use_gpu,
        local_llm=args.local_llm,
        local_llm_path=args.local_llm_path,
    )

    # trigger incremental ingest at startup
    asyncio.run(worker.incremental_ingest())

    # quick CLI
    while True:
        q = input("\nAsk (or 'exit'): ")
        if q.lower() in ("exit", "quit", "q"):
            break
        # stream responses
        print("Answer (streaming):")
        async def show():
            async for chunk in worker.stream_answer(q):
                print(chunk, end="", flush=True)
            print()
        asyncio.run(show())







# Explanations / Design choices
# Semantic sectioning
# Uses markdown headers to break documents into readable sections, then merges tiny sections. This generates more natural "chunks" (conceptual units) than slicing arbitrary characters.
# Deduplication & cleaning
# Each section is hashed (section_md5) and stored in a manifest. Duplicate sections are skipped. clean_text() removes page numbers, basic boilerplate, and collapses whitespace.
# Incremental ingestion (manifest)
# manifest.json records source file md5 + mtime + section md5s. At startup, only files with changed md5/mtime are reprocessed and only new sections embedded.
# BM25 lexical search
# Builds a BM25Okapi index on the section corpus for fast keyword search — useful for exact-match queries and filters. Acts as a complementary signal to embeddings.
# Hybrid search (BM25 + semantic)
# hybrid_search() merges BM25 and semantic candidates, normalizes scores, and computes a weighted final score (hybrid_weight). This avoids pure semantic hallucination and provides better precision on queries with keywords.
# Local LLM vs Groq
# Supports both: local via llama-cpp-python (fast, offline) and remote Groq ChatGroq streaming. You choose at init with local_llm and local_llm_path. If neither available, LLM calls will raise.
# Streaming & async
# stream_answer() is an async generator that yields text chunks. It supports Groq streaming if available, or local non-streaming fallback. This can be directly wired to Gradio streaming endpoints or WebSocket endpoints.

# Feedback loop

# add_feedback() stores labeled QA pairs and context to feedback.json. Over time you can:

# Use positive examples to re-rank candidate passages

# Use negative examples to reduce false positives

# Use the dataset to fine-tune a local LLM or build a supervised reranker

# For production, feed feedback into a retraining pipeline (or a re-ranker model like MonoT5) and periodically reindex.

# Deduplication/performance considerations

# The code avoids re-embedding unchanged content. For high throughput, maintain a separate store mapping section_md5 -> chroma ID to speed updates.

# Scalability to 1000 concurrent users (practical guidance)

# Async streaming generator in code is the software piece. For production:

# Serve via FastAPI + Uvicorn/Gunicorn (async), not plain Gradio for 1k concurrent users.

# Use a connection manager (Redis) to manage sessions, conversational memory, and rate-limiting.

# Use worker pool for LLM calls (multiple processes or GPU-backed nodes). For example, run multiple instances of LLM service behind a load balancer.

# Use token streaming via WebSockets for low-latency streaming responses (Uvicorn + WebSockets).

# Offload embedding/nearest-neighbor searches to a scalable vector DB (FAISS, Milvus, Pinecone, or Chroma with multi-process support).

# Use circuit breakers & queueing to gracefully degrade when capacity is hit.

# Use monitoring (Prometheus, Grafana) and autoscale inference nodes.

# If using Groq or a cloud LLM, ensure you have sufficient concurrency/throughput quotas.

# Next steps I can do for you (pick any; I’ll implement directly)

# Wire this into a FastAPI server with WebSocket streaming and a simple React/Gradio frontend.

# Add a re-ranker stage (train a small Mono/BERT cross-encoder using feedback) to take hybrid candidates and reorder them before LLM prompt injection.

# Implement more advanced boilerplate removal using regex lists or small heuristics (headers, footers, license blocks).

# Build an auto-watcher (filesystem watchdog) that triggers incremental ingestion when files change (careful with race conditions).

# Add explainability: include source filenames and section titles with each LLM answer and a confidence score.

# Add a train-loop script that exports feedback as a dataset and fine-tunes a local Llama-compatible model.