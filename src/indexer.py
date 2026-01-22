from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from src.config import DB_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from src.notion import get_client, fetch_root_pages
from src.vectorstore import VectorStore


def get_vectorstore():
    return VectorStore(DB_DIR)


def get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def index_all_pages(verbose: bool = True) -> int:
    """
    Fetch root pages and their children from Notion and index them.
    Returns the number of chunks indexed.
    """
    notion = get_client()
    store = get_vectorstore()
    embeddings = get_embeddings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # Clear existing data for fresh index
    existing_count = store.count()
    if existing_count > 0:
        store.clear()
        if verbose:
            print(f"Cleared {existing_count} existing chunks")

    total_chunks = 0

    for page in fetch_root_pages(notion):
        if verbose:
            print(f"Indexing: {page['title']}")

        chunks = splitter.split_text(page["content"])

        if not chunks:
            continue

        # Generate embeddings
        chunk_embeddings = embeddings.embed_documents(chunks)

        # Create metadata for each chunk
        metadatas = [
            {
                "page_id": page["id"],
                "title": page["title"],
                "url": page["url"],
                "last_edited": page["last_edited"],
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]

        # Add to store
        store.add(
            embeddings=chunk_embeddings,
            documents=chunks,
            metadatas=metadatas
        )

        total_chunks += len(chunks)

    if verbose:
        print(f"\nIndexed {total_chunks} chunks total")

    return total_chunks


def get_indexed_sources() -> list[dict]:
    """Get list of all indexed pages with metadata."""
    store = get_vectorstore()
    all_metadata = store.get_all_metadata()

    if not all_metadata:
        return []

    # Deduplicate by page_id
    pages = {}
    for meta in all_metadata:
        page_id = meta["page_id"]
        if page_id not in pages:
            pages[page_id] = {
                "title": meta["title"],
                "url": meta["url"],
                "last_edited": meta["last_edited"]
            }

    return list(pages.values())
