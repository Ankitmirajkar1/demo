import os
from pathlib import Path
import shutil
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

# Robust imports: try community/langchain variations where helpful so the app is
# more tolerant of different installed langchain package layouts.
def _import(name: str, alt: str = None):
    try:
        return __import__(name, fromlist=[name.split('.')[-1]])
    except Exception:
        if alt:
            return __import__(alt, fromlist=[alt.split('.')[-1]])
        raise


# ---------- Config ----------
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")  # optional: loads GROQ_API_KEY, OPENAI keys, etc.

# Defaults (can be overridden via .env)
PDF_PATH = os.getenv("RAG_PDF_PATH", str(BASE_DIR / "data"))
CHROMA_DIR = os.getenv("RAG_CHROMA_DIR", str(BASE_DIR / "chroma_db"))

# Ensure directories exist
Path(PDF_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)


# ---------- Helper: robust loader/embeddings/vectorstore creators ----------
def get_pdf_loader_class():
    # Try community loader first, then langchain
    try:
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader
    except Exception:
        try:
            from langchain.document_loaders import PyPDFLoader

            return PyPDFLoader
        except Exception:
            raise ImportError(
                "PyPDFLoader not found. Install 'langchain' or 'langchain-community' with document loaders."
            )


def get_text_splitter_class():
    # Support several possible package names
    for mod in ("langchain_text_splitters", "langchain.text_splitter", "langchain.text_splitters"):
        try:
            m = __import__(mod, fromlist=["RecursiveCharacterTextSplitter"])
            return m.RecursiveCharacterTextSplitter
        except Exception:
            continue
    raise ImportError("RecursiveCharacterTextSplitter not found. Install langchain_text_splitters or compatible langchain.")


def get_hf_embeddings_class():
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings
    except Exception:
        try:
            from langchain.embeddings import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings
        except Exception:
            raise ImportError("HuggingFaceEmbeddings not found. Install langchain or langchain-community embeddings.")


def get_chroma_class():
    try:
        from langchain_community.vectorstores import Chroma

        return Chroma
    except Exception:
        try:
            from langchain.vectorstores import Chroma

            return Chroma
        except Exception:
            raise ImportError("Chroma vectorstore not found. Install chromadb and langchain or langchain-community.")


def get_groq_class():
    try:
        from langchain_groq import ChatGroq

        return ChatGroq
    except Exception:
        return None


# ---------- Helper: build / load vector store ----------
@st.cache_resource(show_spinner=True)
def get_vector_store(pdf_path: str = PDF_PATH, chroma_dir: str = CHROMA_DIR, hf_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Create or load a Chroma vector store. Caches result with Streamlit's cache_resource."""
    PyPDFLoader = get_pdf_loader_class()
    RSplitter = get_text_splitter_class()
    HuggingFaceEmbeddings = get_hf_embeddings_class()
    Chroma = get_chroma_class()

    embedding_model = HuggingFaceEmbeddings(model_name=hf_model)

    # If Chroma dir already exists and non-empty, attempt to load
    if os.path.exists(chroma_dir) and any(os.scandir(chroma_dir)):
        try:
            vect = Chroma(embedding_function=embedding_model, persist_directory=chroma_dir)
            return vect
        except Exception:
            # fall through and rebuild
            st.info("Existing Chroma persistence detected but failed to load; rebuilding from PDFs.")

    # If pdf_path is a directory, load all PDFs inside; else treat as single file
    pdfs = []
    p = Path(pdf_path)
    if p.is_dir():
        pdfs = [str(x) for x in sorted(p.glob("*.pdf"))]
    elif p.is_file():
        pdfs = [str(p)]
    else:
        raise FileNotFoundError(f"PDF path not found: {pdf_path}")

    # Load PDFs
    loader = PyPDFLoader
    all_docs = []
    for pdf in pdfs:
        try:
            ld = loader(pdf)
            docs = ld.load()
            all_docs.extend(docs)
        except Exception as e:
            st.warning(f"Failed to load {pdf}: {e}")

    if not all_docs:
        raise ValueError("No documents loaded from PDFs.")

    # Split into chunks
    splitter = RSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    # Build vectorstore and persist
    vect = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=chroma_dir)
    try:
        vect.persist()
    except Exception:
        pass
    return vect


def format_docs(docs: List) -> str:
    return "\n\n".join([d.page_content for d in docs])


def safe_retrieve(retriever, query: str):
    """Safely retrieve documents from a retriever object across LangChain versions.

    Tries common methods in order and returns a list of documents or an empty list on failure.
    """
    # 1) modern API
    try:
        if hasattr(retriever, "get_relevant_documents"):
            return retriever.get_relevant_documents(query)
    except Exception:
        pass

    # 2) older API
    try:
        if hasattr(retriever, "retrieve"):
            return retriever.retrieve(query)
    except Exception:
        pass

    # 3) some retrievers are callable
    try:
        if callable(retriever):
            return retriever(query)
    except Exception:
        pass

    # 4) give up gracefully
    return []


def get_indexed_sources(vector_store, limit_sources: int = 10):
    """Return a summary of indexed sources and sample chunks from the vector store.

    Tries several access patterns depending on the Chroma/langchain wrapper version.
    Returns a dict: {source_path: [sample_texts...]}
    """
    results = {}
    # 1) try native collection get
    try:
        coll = getattr(vector_store, "_collection", None)
        if coll is not None and hasattr(coll, "get"):
            native = coll.get(include=["metadatas", "documents"], limit=1000)
            metadatas = native.get("metadatas", [])
            docs = native.get("documents", [])
            for m, d in zip(metadatas, docs):
                src = m.get("source") if isinstance(m, dict) else None
                if not src:
                    src = m.get("uri") if isinstance(m, dict) else None
                if not src:
                    src = "unknown"
                results.setdefault(src, []).append(d[:1000])
            return {k: v[:3] for k, v in list(results.items())[:limit_sources]}
    except Exception:
        pass

    # 2) try vector_store.get
    try:
        native = vector_store.get(include=["metadatas", "documents"], limit=1000)
        metadatas = native.get("metadatas", [])
        docs = native.get("documents", [])
        for m, d in zip(metadatas, docs):
            src = None
            if isinstance(m, dict):
                src = m.get("source") or m.get("uri")
            if not src:
                src = "unknown"
            results.setdefault(src, []).append(d[:1000])
        return {k: v[:3] for k, v in list(results.items())[:limit_sources]}
    except Exception:
        pass

    # 3) fallback: try to scan retriever with common queries (not ideal)
    try:
        retr = vector_store.as_retriever(search_kwargs={"k": 5})
        # use filenames from data dir as seeds
        data_dir = Path(PDF_PATH)
        seeds = [p.name for p in data_dir.glob("*.pdf")][:limit_sources]
        for s in seeds:
            docs = safe_retrieve(retr, s)
            if docs:
                results.setdefault(s, []).extend([d.page_content[:1000] for d in docs[:3]])
        return results
    except Exception:
        pass

    return {}


@st.cache_resource(show_spinner=True)
def get_qa_chain(k: int = 3):
    """Return a runnable-style QA chain if available, else a simple function.

    Uses ChatGroq if available (GROQ_API_KEY expected in .env) as default.
    """
    ChGroq = get_groq_class()
    # Load vector store
    vect = get_vector_store()
    retriever = vect.as_retriever(search_kwargs={"k": k})

    # Attempt to create a runnable chain using langchain_core if available
    try:
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.documents import Document

        # Prefer Groq if available
        llm = ChGroq(model="openai/gpt-oss-120b", temperature=0) if ChGroq else None

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant that answers ONLY from the provided PDF context."),
                ("human", "Context: {context}\n\nQuestion: {question}\n\nAnswer concisely; if not in context say you don't know."),
            ]
        )

        def format_docs_local(docs: List[Document]) -> str:
            return "\n\n".join([d.page_content for d in docs])

        qa_chain = (
            {
                "context": retriever | format_docs_local,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        return qa_chain, retriever
    except Exception:
        # Fallback: simple function-style QA
        def simple_qa(question: str) -> Tuple[str, List]:
            try:
                docs = retriever.get_relevant_documents(question)
            except Exception:
                docs = retriever.retrieve(question)
            context = format_docs(docs)
            # If Groq available, try it; else return context when no LLM
            if ChGroq:
                try:
                    groq = ChGroq(model="openai/gpt-oss-120b", temperature=0)
                    if hasattr(groq, "predict"):
                        out = groq.predict(f"Context: {context}\n\nQuestion: {question}\n\nAnswer:")
                    else:
                        out = groq(f"Context: {context}\n\nQuestion: {question}\n\nAnswer:")
                    return out, docs
                except Exception as e:
                    return f"LLM call failed: {e}\n\nRetrieved context:\n\n{context}", docs
            return context, docs

        return simple_qa, retriever


# ---------- Streamlit UI ----------
st.set_page_config(page_title="PDF RAG QA", page_icon="📄", layout="wide")
st.title("📄 PDF RAG Question Answering")
st.markdown("Ask questions about PDFs in the data folder or set RAG_PDF_PATH in .env.")

with st.sidebar:
    st.header("Configuration")
    st.write("Chroma + HuggingFace embeddings + ChatGroq (if available)")
    st.text(f"PDF path: {PDF_PATH}")
    st.text(f"Chroma dir: {CHROMA_DIR}")

    st.markdown("---")
    st.write("Advanced: choose chunking and retriever options")
    chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=2000, value=200, step=50)
    k = st.slider("Retriever k (top-k chunks)", 1, 10, 3)

    if st.button("Rebuild vector store from PDFs"):
        # Clear cached resources so vector store rebuilds
        get_vector_store.clear()  # type: ignore[attr-defined]
        get_qa_chain.clear()      # type: ignore[attr-defined]
        st.success("Caches cleared — vector store will rebuild on next action.")

    uploaded = st.file_uploader("Upload a PDF to add to data folder", type=["pdf"])
    if uploaded:
        dest = Path(PDF_PATH)
        if dest.is_dir():
            out_path = dest / uploaded.name
        else:
            out_path = Path(PDF_PATH).parent / uploaded.name
        with open(out_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved {uploaded.name} to {out_path}")

        # Remove existing Chroma persistence so we don't reuse embeddings that
        # were created before the new PDF was added. This forces a rebuild.
        try:
            if os.path.exists(CHROMA_DIR):
                shutil.rmtree(CHROMA_DIR)
                st.info("Removed existing Chroma persistence to include the new PDF.")
        except Exception as e:
            st.warning(f"Failed to remove Chroma persistence: {e}")

        # Clear cached resources and trigger an immediate rebuild so the new
        # document is indexed and available for queries right away.
        get_vector_store.clear()  # type: ignore[attr-defined]
        get_qa_chain.clear()      # type: ignore[attr-defined]
        try:
            with st.spinner("Rebuilding embeddings to include the uploaded PDF (this may take a while)..."):
                _ = get_vector_store(pdf_path=str(out_path.parent), chroma_dir=CHROMA_DIR)
            st.success("Rebuild finished — the uploaded PDF is now indexed.")
        except Exception as e:
            st.error(f"Rebuild failed: {e}")


# Main workspace: list PDFs
pdf_list = []
pp = Path(PDF_PATH)
if pp.is_dir():
    pdf_list = sorted(pp.glob("*.pdf"))
elif pp.is_file():
    pdf_list = [pp]

if not pdf_list:
    st.warning("No PDFs found. Upload one via the sidebar or set RAG_PDF_PATH in .env to a PDF or folder.")
    st.stop()

st.write(f"Found {len(pdf_list)} PDF(s)")
for p in pdf_list:
    st.write("-", p.name)

# Build/load vector store and prepare QA chain
try:
    with st.spinner("Preparing vector store (may take time first run)..."):
        vect = get_vector_store(pdf_path=str(pp), chroma_dir=CHROMA_DIR)
    st.success("Vector store ready")
except Exception as e:
    st.error(f"Failed to prepare vector store: {e}")
    st.stop()

# Diagnostics: show indexed sources and sample chunks so user can confirm uploaded PDFs are present
with st.expander("Indexed sources and sample chunks (diagnostics)"):
    try:
        srcs = get_indexed_sources(vect, limit_sources=10)
        if not srcs:
            st.info("No indexed source metadata available from Chroma. Try rebuilding or check persistence format.")
        for src, samples in srcs.items():
            st.markdown(f"**{src}**")
            for s in samples:
                st.write(s if len(s) < 1000 else s[:1000] + " ... (truncated)")
    except Exception as e:
        st.warning(f"Diagnostics unavailable: {e}")

# show simple count when available
try:
    cnt = None
    if hasattr(vect, "_collection") and hasattr(vect._collection, "count"):
        cnt = vect._collection.count()
    elif hasattr(vect, "count"):
        cnt = vect.count()
    if cnt is not None:
        st.write(f"Total vectors in store: {cnt}")
except Exception:
    pass

query = st.text_input("Enter your question", value="What are the cancellation charges?")
if st.button("Ask"):
    if not query.strip():
        st.warning("Please type a question first.")
    else:
        try:
            qa_chain, retriever = get_qa_chain(k=k)
        except Exception as e:
            st.error(f"Failed to prepare QA chain: {e}")
            st.stop()

        try:
            # runnable-style chain returns via invoke, fallback returns (answer, docs)
            if hasattr(qa_chain, "invoke"):
                answer = qa_chain.invoke(query)
                docs = safe_retrieve(retriever, query)
            else:
                out = qa_chain(query)
                if isinstance(out, tuple) and len(out) == 2:
                    answer, docs = out
                else:
                    answer = out
                    docs = safe_retrieve(retriever, query)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Top retrieved chunks")
            for i, d in enumerate(docs[:k]):
                st.markdown(f"**Result {i+1}**")
                content = d.page_content
                st.write(content if len(content) < 2000 else content[:2000] + " ... (truncated)")
                if getattr(d, "metadata", None):
                    st.write("metadata:", d.metadata)
        except Exception as e:
            st.error(f"Query failed: {e}")

st.markdown("---")
st.markdown("Hints:")
st.markdown("- Use 'Rebuild vector store' after adding/removing PDFs to refresh embeddings.")
st.markdown("- Set GROQ_API_KEY or OPENAI keys in .env to enable LLM calls.")