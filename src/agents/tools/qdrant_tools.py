import os
import logging
from typing import List, Dict, Any, Optional

import boto3
from langchain_core.tools import tool
from langchain_aws import BedrockEmbeddings
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "documents")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
AWS_REGION = os.getenv("AWS_REGION_NAME", "us-east-1")


def _build_vector_store(collection_name: str) -> QdrantVectorStore:
    bedrock_rt = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    embeddings = BedrockEmbeddings(client=bedrock_rt, model_id=EMBEDDING_MODEL_ID)
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )


class QdrantSearchInput(BaseModel):
    query: str = Field(description="Natural-language query to find relevant document chunks")
    collection_name: Optional[str] = Field(
        default=None,
        description=f"Qdrant collection to search. Defaults to '{QDRANT_COLLECTION_NAME}'.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top-scoring chunks to retrieve (1–20).",
    )


@tool(
    "qdrant_search",
    description="Search the Qdrant vector database for document chunks relevant to the user query. Returns ranked chunks with content, metadata, and similarity score. Use for any factual or knowledge-based question requiring retrieval from the knowledge base.",
    args_schema=QdrantSearchInput,
)
def qdrant_search_tool(
    query: str,
    collection_name: Optional[str] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    collection = collection_name or QDRANT_COLLECTION_NAME
    logger.info("qdrant_search | collection=%s top_k=%d query='%.80s'", collection, top_k, query)
    try:
        store = _build_vector_store(collection)
        results = store.similarity_search_with_score(query, k=top_k)
        chunks = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            }
            for doc, score in results
        ]
        logger.info("qdrant_search returned %d chunks", len(chunks))
        return chunks
    except Exception as e:
        logger.error("qdrant_search failed: %s", e, exc_info=True)
        return [{"error": str(e), "content": "", "metadata": {}, "score": 0.0}]
