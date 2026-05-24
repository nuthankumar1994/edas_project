from typing import Optional
from pydantic import BaseModel


class UploadResponse(BaseModel):
    document_id: str
    file_name: str
    file_type: str
    total_chunks: int
    total_pages: int
    author: Optional[str]
    title: Optional[str]
    message: str


class DocumentInfo(BaseModel):
    document_id: str
    file_name: str
    file_type: str
    total_chunks: int
    uploaded_at: str


class DeleteResponse(BaseModel):
    document_id: str
    message: str
