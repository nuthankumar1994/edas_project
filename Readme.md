## 1. High-Level System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DocAssist+ Platform                            │
│                                                                         │
│   ┌──────────┐     ┌─────────────┐     ┌──────────────────────────┐   │
│   │  Client  │────▶│ API Gateway │────▶│   Application Services   │   │
│   │ (Web/App)│◀────│ (Auth + Rate│◀────│ (Document + Query Layer) │   │
│   └──────────┘     │  Limiting)  │     └──────────────────────────┘   │
│                    └─────────────┘               │                     │
│                                                  │                     │
│              ┌───────────────────────────────────┤                     │
│              │                                   │                     │
│              ▼                                   ▼                     │
│   ┌─────────────────────┐          ┌─────────────────────────┐        │
│   │  Document Pipeline  │          │    Query Pipeline        │        │
│   │  (Ingestion Path)   │          │    (Query Path)          │        │
│   └─────────────────────┘          └─────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2. Full Component Map

```
┌───────────────────────────────────────────────────────────────────────┐
│                        DocAssist+ Components                          │
│                                                                       │
│  ┌─────────────┐   ┌──────────────┐   ┌───────────────────────────┐  │
│  │  API Layer  │   │  Processing  │   │       Storage Layer       │  │
│  │             │   │  Services    │   │                           │  │
│  │ • /upload   │   │              │   │ ┌─────────────────────┐   │  │
│  │ • /query    │   │ • Text       │   │ │  Vector DB          │   │  │
│  │             │   │   Extractor  │   │ │  (Qdrant)           │   │  │
│  │ Auth via    │   │ • Chunker    │   │ │                     │   │  │
│  │ JWT token   │   │ • Embedder   │   │ │                     │   │  │
│  │             │   │ • Intent     │   │ │  Indexed by:        │   │  │
│  │ Rate limit  │   │   Classifier │   │ │  userId +           │   │  │
│  │ per userId  │   │ • Emotion    │   │ │  documentId         │   │  │
│  └─────────────┘   │   Classifier │   │ └─────────────────────┘   │  │
│                    │ • Retriever  │   │                           │  │
│                    │ • Prompt     │   │ ┌─────────────────────┐   │  │
│                    │   Assembler  │   │ │  Metadata DB        │   │  │
│                    │ • LLM Client │   │ │  (PostgreSQL)       │   │  │
│                    └──────────────┘   │ │                     │   │  │
│                                       │ │  users, documents,  │   │  │
│  ┌─────────────┐                      │ │  query logs         │   │  │
│  │  Cache      │                      │ └─────────────────────┘   │  │
│  │  (Redis)    │                      │                           │  │
│  │             │                      │ ┌─────────────────────┐   │  │
│  │ • Query     │                      │ │  Object Store       │   │  │
│  │   response  │                      │ │  (S3 / GCS)         │   │  │
│  │   cache     │                      │ │                     │   │  │
│  │ • Embedding │                      │ │  Raw uploaded files │   │  │
│  │   cache     │                      │ └─────────────────────┘   │  │
│  └─────────────┘                      └───────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────┘
```
