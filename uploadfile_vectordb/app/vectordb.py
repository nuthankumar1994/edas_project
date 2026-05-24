import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

_client: Optional[QdrantClient] = None


def get_client(url: str) -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=url)
    return _client


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    document_id: str,
    doc_meta: Dict[str, Any],
    file_name: str,
) -> None:
    points = [
        PointStruct(
            id=str(uuid.uuid1()),
            vector=embedding,
            payload={**chunk, **doc_meta, "document_id": document_id, "file_name": file_name},
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    client.upsert(collection_name=collection_name, points=points)


def delete_document(client: QdrantClient, collection_name: str, document_id: str) -> None:
    client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        ),
    )


def list_documents(client: QdrantClient, collection_name: str) -> List[Dict[str, Any]]:
    """Return one summary record per unique document_id."""
    seen, docs, offset = set(), [], None
    while True:
        results, offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            doc_id = point.payload.get("document_id")
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                docs.append(
                    {
                        "document_id": doc_id,
                        "file_name": point.payload.get("file_name"),
                        "file_type": point.payload.get("file_type"),
                        "total_chunks": point.payload.get("total_chunks"),
                        "uploaded_at": point.payload.get("uploaded_at"),
                    }
                )
        if offset is None:
            break
    return docs
