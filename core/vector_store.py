import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

def build_vector_store(transcript: str) -> Chroma:
    if not transcript or not transcript.strip():
        raise ValueError("The transcript is empty; there is nothing to index.")

    print("Building vector store...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]
    embeddings = get_embeddings()
    # Keep each analysis isolated and in memory. A fixed persistent collection
    # would mix passages from previous meetings into the current chat.
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=f"meeting_{os.urandom(8).hex()}",
    )
    return vector_store

def load_vector_store() -> Chroma:
    """Load the legacy on-disk store, if one was created by an older version."""
    chroma_dir = os.getenv("CHROMA_DIR", "vector_db")
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=chroma_dir,
        embedding_function=embeddings
    )

    return vector_store

def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
