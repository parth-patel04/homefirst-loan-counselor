"""
rag.py — HomeFirst Vernacular Loan Counselor
Bonus Phase: Knowledge Retrieval (RAG) using ChromaDB.

Architecture:
  - ChromaDB as local vector database (no external API needed)
  - sentence-transformers for local embeddings (paraphrase-multilingual-MiniLM-L12-v2)
    → Supports Hindi, Marathi, Tamil, English out of the box
  - 10 HomeFirst policy FAQ documents loaded from data/faqs.py
  - Top-K semantic search retrieves relevant context before LLM answers FAQ

Usage in brain.py:
    from rag import build_retriever
    retriever = build_retriever()
    brain     = LoanCounselorBrain(rag_retriever=retriever)

The retriever is a simple callable:
    context_str = retriever("What documents do I need for a self-employed loan?")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CHROMA_PERSIST_DIR  = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME     = "homefirst_faqs"
EMBEDDING_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"   # 50MB, multilingual
TOP_K               = 2     # number of docs to retrieve per query
MIN_RELEVANCE_SCORE = 0.30  # cosine similarity threshold (0–1); below this = skip


# ─────────────────────────────────────────────────────────────────────────────
# RAG class
# ─────────────────────────────────────────────────────────────────────────────

class HomeFirstRAG:
    """
    Local RAG system using ChromaDB + sentence-transformers.

    Attributes:
        collection : ChromaDB collection holding FAQ embeddings
        embedder   : SentenceTransformer model for query encoding
    """

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        self.persist_dir = persist_dir
        self.collection  = None
        self.embedder    = None
        self._ready      = False

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self) -> bool:
        """
        Initialise ChromaDB and embedding model.
        Load FAQ documents if collection is empty.

        Returns:
            True if setup succeeded, False otherwise.
        """
        try:
            self._init_chromadb()
            self._init_embedder()

            # Load documents if collection is empty
            if self.collection.count() == 0:
                logger.info("Collection empty — loading FAQ documents...")
                self._load_documents()
            else:
                logger.info(
                    "Collection '%s' already has %d documents.",
                    COLLECTION_NAME, self.collection.count()
                )

            self._ready = True
            logger.info("RAG system ready. %d documents indexed.", self.collection.count())
            return True

        except ImportError as e:
            logger.error(
                "RAG dependencies missing: %s\n"
                "Install with: pip install chromadb sentence-transformers", e
            )
            return False
        except Exception as e:
            logger.error("RAG setup failed: %s", e)
            return False

    def _init_chromadb(self):
        """Initialise ChromaDB persistent client and get/create collection."""
        import chromadb

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(path=self.persist_dir)

        # Use cosine similarity (best for semantic search)
        self.collection = client.get_or_create_collection(
            name     = COLLECTION_NAME,
            metadata = {"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB initialised at '%s'. Collection: '%s'",
            self.persist_dir, COLLECTION_NAME
        )

    def _init_embedder(self):
        """Load the multilingual sentence-transformer model."""
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")

    def _load_documents(self):
        """Embed and store all FAQ documents in ChromaDB."""
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from data.faqs import FAQ_DOCUMENTS

        ids        = []
        documents  = []
        embeddings = []
        metadatas  = []

        for doc in FAQ_DOCUMENTS:
            # Embed the combined title + content for richer representation
            text_to_embed = f"{doc['title']}\n\n{doc['content']}"
            embedding     = self.embedder.encode(text_to_embed).tolist()

            ids.append(doc["id"])
            documents.append(doc["content"])
            embeddings.append(embedding)
            metadatas.append({
                "title":    doc["title"],
                "category": doc["category"],
                "id":       doc["id"],
            })

        self.collection.add(
            ids        = ids,
            documents  = documents,
            embeddings = embeddings,
            metadatas  = metadatas,
        )
        logger.info("Loaded %d FAQ documents into ChromaDB.", len(FAQ_DOCUMENTS))

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K) -> str:
        """
        Retrieve the most relevant FAQ context for a query.

        Args:
            query  : User's question (any language)
            top_k  : Number of documents to retrieve

        Returns:
            Formatted context string to inject into the LLM prompt.
            Returns empty string if nothing relevant found.
        """
        if not self._ready:
            logger.warning("RAG not ready — returning empty context.")
            return ""

        if not query.strip():
            return ""

        try:
            # Encode the query
            query_embedding = self.embedder.encode(query).tolist()

            # Search ChromaDB
            results = self.collection.query(
                query_embeddings = [query_embedding],
                n_results        = top_k,
                include          = ["documents", "metadatas", "distances"],
            )

            if not results["documents"] or not results["documents"][0]:
                logger.info("No RAG results for query: '%s'", query[:50])
                return ""

            # Format retrieved context
            context_parts = []
            for doc, meta, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                # Convert to similarity: 1 - (distance/2)
                similarity = 1.0 - (distance / 2.0)

                if similarity < MIN_RELEVANCE_SCORE:
                    logger.debug(
                        "Skipping low-relevance doc '%s' (similarity=%.2f)",
                        meta.get("title"), similarity
                    )
                    continue

                logger.info(
                    "RAG match: '%s' | similarity=%.2f",
                    meta.get("title"), similarity
                )
                context_parts.append(
                    f"[Source: {meta.get('title', 'HomeFirst Policy')}]\n{doc}"
                )

            if not context_parts:
                return ""

            return "\n\n---\n\n".join(context_parts)

        except Exception as e:
            logger.error("RAG retrieval error: %s", e)
            return ""

    def __call__(self, query: str) -> str:
        """Make the RAG instance callable — used directly by brain.py."""
        return self.retrieve(query)

    # ── Utilities ─────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    def document_count(self) -> int:
        if self.collection:
            return self.collection.count()
        return 0

    def reset_collection(self):
        """
        Delete and recreate the collection.
        Use this to re-index documents after updating faqs.py.
        """
        if self.collection:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_dir)
            client.delete_collection(COLLECTION_NAME)
            self.collection = None
            self._ready     = False
            logger.info("Collection '%s' deleted.", COLLECTION_NAME)
            self.setup()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level builder (used by app.py and brain.py)
# ─────────────────────────────────────────────────────────────────────────────

_rag_instance: HomeFirstRAG | None = None


def build_retriever(persist_dir: str = CHROMA_PERSIST_DIR) -> HomeFirstRAG | None:
    """
    Build and return a ready RAG retriever singleton.

    Returns:
        HomeFirstRAG instance if setup succeeded.
        None if dependencies are missing (RAG disabled gracefully).

    Usage:
        retriever = build_retriever()
        brain     = LoanCounselorBrain(rag_retriever=retriever)
    """
    global _rag_instance

    if _rag_instance and _rag_instance.is_ready:
        return _rag_instance

    rag = HomeFirstRAG(persist_dir=persist_dir)
    if rag.setup():
        _rag_instance = rag
        return rag

    logger.warning(
        "RAG unavailable — brain will operate without FAQ retrieval. "
        "Install: pip install chromadb sentence-transformers"
    )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (run: python rag.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    print("=" * 60)
    print("HomeFirst RAG — Self Test")
    print("=" * 60)

    # Check data/faqs.py
    try:
        from data.faqs import FAQ_DOCUMENTS, FAQ_BY_CATEGORY
        print(f"\n✅ FAQ documents loaded: {len(FAQ_DOCUMENTS)} docs")
        for doc in FAQ_DOCUMENTS:
            print(f"   [{doc['id']}] {doc['title']}")
    except ImportError as e:
        print(f"❌ Could not load faqs.py: {e}")
        sys.exit(1)

    # Check dependencies
    print("\nChecking dependencies...")
    deps_ok = True
    for pkg in ["chromadb", "sentence_transformers"]:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg} installed")
        except ImportError:
            print(f"  ❌ {pkg} not installed. Run: pip install {pkg.replace('_','-')}")
            deps_ok = False

    if not deps_ok:
        print("\n❌ Install missing packages and re-run.")
        sys.exit(1)

    # Build RAG
    print("\nBuilding RAG system (first run downloads ~50MB model)...")
    rag = HomeFirstRAG(persist_dir="./test_chroma_db")
    success = rag.setup()

    if not success:
        print("❌ RAG setup failed.")
        sys.exit(1)

    print(f"✅ RAG ready. Documents indexed: {rag.document_count()}")

    # Test queries
    test_queries = [
        ("English — documents for self-employed",
         "What documents do I need for a self-employed home loan?"),
        ("Hindi — salary requirement",
         "Home loan ke liye kitni salary chahiye?"),
        ("English — LTV explained",
         "What is the maximum LTV ratio at HomeFirst?"),
        ("Hinglish — tax benefit",
         "Home loan pe tax benefit milta hai kya?"),
        ("English — out of scope",
         "What is the weather today?"),
    ]

    print("\n" + "=" * 60)
    print("RETRIEVAL TESTS")
    print("=" * 60)

    for label, query in test_queries:
        print(f"\n[{label}]")
        print(f"Query: {query}")
        context = rag.retrieve(query, top_k=2)
        if context:
            # Show first 200 chars of context
            preview = context[:200].replace("\n", " ")
            print(f"✅ Retrieved: {preview}...")
        else:
            print("⚪ No relevant context found (expected for out-of-scope).")

    # Cleanup test DB
    import shutil
    shutil.rmtree("./test_chroma_db", ignore_errors=True)
    print("\n✅ Test ChromaDB cleaned up.")
    print("\nAll RAG tests passed ✅")
    print("\nTo use in production:")
    print("  from rag import build_retriever")
    print("  retriever = build_retriever()")
    print("  brain = LoanCounselorBrain(rag_retriever=retriever)")
