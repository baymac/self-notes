from langchain_ollama import OllamaEmbeddings, OllamaLLM

from src.config import DB_DIR, EMBEDDING_MODEL, LLM_MODEL, TOP_K_RESULTS
from src.vectorstore import VectorStore


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided notes.

Rules:
1. Answer ONLY using information from the notes below
2. If the answer is not in the notes, say "I don't have information about that in my notes"
3. Be concise and direct
4. When relevant, mention which note the information came from

Notes:
{context}

Question: {question}

Answer:"""


def query(question: str, verbose: bool = False) -> dict:
    """
    Query the indexed notes and return an answer with sources.

    Returns:
        dict with 'answer' and 'sources' keys
    """
    store = VectorStore(DB_DIR)

    if store.count() == 0:
        return {
            "answer": "No notes indexed yet. Run 'python cli.py index' first.",
            "sources": []
        }

    # Generate embedding for the question
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    question_embedding = embeddings.embed_query(question)

    # Search for relevant chunks
    results = store.query(
        query_embedding=question_embedding,
        n_results=TOP_K_RESULTS
    )

    if not results["documents"][0]:
        return {
            "answer": "No relevant notes found.",
            "sources": []
        }

    # Build context from retrieved chunks
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []
    sources = []
    seen_pages = set()

    for chunk, meta in zip(chunks, metadatas):
        context_parts.append(f"[From: {meta['title']}]\n{chunk}")

        if meta["page_id"] not in seen_pages:
            seen_pages.add(meta["page_id"])
            sources.append({
                "title": meta["title"],
                "url": meta["url"]
            })

    context = "\n\n---\n\n".join(context_parts)

    if verbose:
        print(f"Found {len(chunks)} relevant chunks from {len(sources)} pages\n")

    # Generate answer with LLM
    llm = OllamaLLM(model=LLM_MODEL)
    prompt = SYSTEM_PROMPT.format(context=context, question=question)
    answer = llm.invoke(prompt)

    return {
        "answer": answer,
        "sources": sources
    }
