from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embed_retriever import EmbeddingRetriever

# Initialize (loads from disk)
bm25 = BM25Retriever()
embed = EmbeddingRetriever()

# User claim
claim = "Vietnam's coffee production dropped by 40%"

# Get documents
docs_lexical = bm25.retrieve(claim, top_k=10)
docs_semantic = embed.retrieve(claim, top_k=10)

# Combine and deduplicate
all_docs = docs_lexical + docs_semantic
# ... proceed to Sentence Ranking