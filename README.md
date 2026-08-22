# Medical RAG Chatbot — AI Forward Deployed Engineering (AI FDE) Architecture

> **An evidence-grounded conversational AI knowledge assistant for clinical guidelines, drug monographs, and medical research documents.**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-RAG%20Pipeline-1C3C3C.svg)](https://python.langchain.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3%20Inference-F05A28.svg)](https://groq.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF4F00.svg)](https://www.trychroma.com)
[![BGE Embeddings](https://img.shields.io/badge/HuggingFace-BGE%20Small%20v1.5-FFD21E.svg)](https://huggingface.co/BAAI/bge-small-en-v1.5)
[![System Design](https://img.shields.io/badge/Architecture-System%20Design%20Doc-purple.svg)](SYSTEM_DESIGN.md)

> 📘 **Full Architecture & Engineering Specifications**: See the complete [System Design Document](SYSTEM_DESIGN.md) for detailed Mermaid diagrams, multi-tenant RBAC matrices, vector pre-filtering specifications, and production deployment topology.

---

## 📸 Application Screenshots

### 1. Chat Interface — Grounded Clinical Q&A with Citations
> The main conversational interface showing a clinical query about Pediatric Acute Otitis Media. The AI returns a **grounded, evidence-based answer** with drug dosing tables, verified page citations, and a medical disclaimer — all retrieved from indexed clinical guidelines.

![Chat Interface — Clinical Q&A with Verified Citations](Photo/chat-interface.png)

---

### 2. Knowledge Base Library — 25 Indexed Medical Documents
> The **Reference Library** modal displaying all 25 pre-indexed clinical documents organized by category (Guidelines, Manuals, Research, Protocols). Each card shows a summary and allows one-click querying.

![Knowledge Base Library — 25 Indexed Documents](Photo/knowledge-base-library.png)

---

### 3. RBAC Authentication — Role-Based Persona Switcher
> The **Authentication & Role-Based Access** modal with 1-click Quick-Login personas for 4 enterprise roles: `ADMIN` (full platform), `FDE_ENGINEER` (internal SOPs + assigned tenant), `CUSTOMER` (isolated tenant clearance). Includes custom credentials login with JWT token authentication.

![RBAC Authentication — Quick-Login Persona Switcher](Photo/rbac-auth-modal.png)

---

### 4. Admin Console — User Management & Security Governance
> The **Enterprise Administration & Governance** panel showing user CRUD operations, role assignment (`ADMIN`, `FDE_ENGINEER`, `CUSTOMER`), tenant provisioning (`system`, `customer_001`, `customer_002`), and the Security Audit Logs tab for real-time event monitoring.

![Admin Console — User Management & Governance](Photo/admin-console.png)

---

## 1. Executive Summary & Customer Problem

Healthcare institutions, medical centers, and clinical research groups manage tens of thousands of pages of clinical practice guidelines, pharmacopeia drug monographs, and consensus statements. Medical practitioners and researchers spend extensive hours manually searching, cross-referencing, and synthesizing information across disparate documents.

### The Core Problem
* **Time-Consuming Manual Search**: Clinicians spend valuable clinical time hunting for specific contraindications, dosing adjustments, and glycemic targets across multi-page PDFs.
* **Risk of Hallucination in Standard LLMs**: General-purpose LLMs hallucinate dosages, invent non-existent clinical trials, or conflate conflicting institutional guidelines.
* **Lack of Verifiable Source Attribution**: Clinicians cannot trust answers without verbatim document names and exact page citations.

### The AI FDE Solution
A **chat-only Medical RAG Assistant** that pre-indexes verified medical guidelines into a high-performance vector store, performs meaning-based two-stage retrieval (dense semantic search + Cross-Encoder reranking), and generates grounded answers with exact document and page citations using Groq's high-speed inference.

```
Ask Question  ───►  Semantic Search + Reranking  ───►  Grounded Answer + Verifiable Citations
```

---

## 2. End-to-End Architecture

```
                         USER (Web Client)
                               │
                               │ Medical Query
                               ▼
                    ┌─────────────────────┐
                    │ HTML5 / CSS3 / JS   │
                    │   Chat Interface    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       FastAPI       │
                    │    Backend Server   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      LangChain      │
                    │    RAG Pipeline     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   BGE Embedder      │
                    │ BAAI/bge-small-en   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      ChromaDB       │
                    │ Dense Vector Search │
                    └──────────┬──────────┘
                               │
                          Top-10 Chunks
                               │
                               ▼
                    ┌─────────────────────┐
                    │  BGE Cross-Encoder  │
                    │   Reranker Base     │
                    └──────────┬──────────┘
                               │
                           Top-4 Best Context
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Groq LLM       │
                    │  LLaMA 3.3 (70B)    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Grounded Response  │
                    │  + Page Citations   │
                    └──────────┬──────────┘
                               │
                               ▼
                         USER INTERFACE
```

---

## 3. The Two Core Pipelines

### Pipeline A: Knowledge Ingestion (Offline Pre-Indexing)
Runs during system initialization or scheduled knowledge updates. End users do not upload documents.

```
Medical Documents: PDFs & TXT files (medical_documents/)
     │
     ▼
Medical Document Loader ──► Page-by-Page Extraction + Layout Cleaning
 (PyMuPDF for PDF & UTF-8 Page Parser for TXT)
     │
     ▼
Medical Chunker        ──► Semantic Character Splitting (600 chars, 120 overlap)
                          Inherits metadata: doc_id, doc_name, page_num, section
     │
     ▼
BGE Embeddings         ──► 384-dimensional dense vectors
     │
     ▼
ChromaDB               ──► Persistent Vector Index (HNSW Cosine Space)
```

### Pipeline B: User Consultation & Retrieval (Online Inference)
Runs each time a user asks a clinical question.

```
User Query
     │
     ▼
BGE Query Embedding (with asymmetric search instruction)
     │
     ▼
ChromaDB Semantic Search (Top-10 candidate chunks, threshold filtering)
     │
     ▼
BGE Cross-Encoder Reranker (Calculates cross-attention pairs: (Query, Chunk))
     │
     ▼
Context & Source Assembly (Strict metadata formatting with doc & page attribution)
     │
     ▼
Medical RAG Prompt Guardrails (Factual grounding, conflict identification)
     │
     ▼
Groq LLM (LLaMA 3.3-70B-versatile / 3.1-8B-instant)
     │
     ▼
Structured JSON Response (Answer, Citations array, Latency in ms)
     │
     ▼
SQLite Persistence (Multi-turn session history)
```

---

## 4. AI FDE Engineering Decisions & Trade-Offs

### A. RAG vs Fine-Tuning Decision Matrix
| Dimension | RAG (Selected) | Fine-Tuning |
| :--- | :--- | :--- |
| **Hallucination Risk** | Extremely low (constrained to retrieved text) | High (parametric memory drift) |
| **Verifiable Citations** | Native (exact document, section, page number) | Impossible without synthetic citation tuning |
| **Knowledge Updates** | Instantaneous (drop new PDF, re-index chunks) | Expensive retraining / catastrophic forgetting |
| **Regulatory & Audit** | 100% auditable evidence trail | Black-box parametric weight retrieval |
| **Cost & Latency** | Low inference cost via Groq | High training compute costs |

### B. Vector Database: ChromaDB
* **Rationale**: ChromaDB provides lightweight, embedded, persistent HNSW vector indexing without external cloud service overhead, making it ideal for local air-gapped clinical deployments or microservice environments.

### C. Embeddings: BAAI BGE (`bge-small-en-v1.5`)
* **Rationale**: BGE models consistently rank at the top of the Massive Text Embedding Benchmark (MTEB). It outputs 384-dimensional normalized vectors with asymmetric query-passage matching, outperforming legacy models like OpenAI `text-embedding-ada-002` while running locally with sub-10ms latency.

### D. Second-Stage Reranking: BGE Cross-Encoder (`bge-reranker-base`)
* **Rationale**: Vector similarity uses bi-encoders (computing query and chunk embeddings independently). Cross-encoders compute full cross-attention across every word pair `(query, document_chunk)`. While too slow for the full database, applying it to the top-10 candidate chunks drastically improves precision and filters out false positive vector matches before LLM context construction.

### E. LLM Inference: Groq LPU + LLaMA 3.3
* **Rationale**: Groq's Tensor Streaming Processor (LPU) delivers 300–500 tokens/second inference speed. In clinical settings where milliseconds matter, Groq eliminates the 5–10s generation lag of traditional GPUs.

### F. Chunking Strategy
* `chunk_size = 600` characters: Captures complete clinical guidelines paragraphs and drug dosage rules without overflowing token limits.
* `chunk_overlap = 120` characters: Prevents loss of critical sentence context across chunk boundaries.
* **Full Metadata Inheritance**: Every chunk retains `document_id`, `document_name`, `page_number`, `total_pages`, `section`, and unique `chunk_id`.

---

## 5. Medical Grounding & Hallucination Prevention Guardrails

The system enforces strict operational constraints:
1. **Context-Only Grounding**: The LLM is prohibited from answering based on pre-trained parametric knowledge.
2. **Deterministic Negative Handling**: If retrieval returns no high-confidence chunks, the system automatically returns:
   > *"I could not find sufficient information in the provided medical documents to answer this question."*
3. **Multi-Document Conflict Detection**: When guidelines differ (e.g., target BP `<130/80` in modern guidelines vs `<140/90` in legacy protocols), the prompt forces the LLM to explicitly contrast both viewpoints rather than hallucinating an arbitrary consensus.
4. **Verbatim Page Citations**: Citations must map to physical pages present in the knowledge base.

---

## 6. Preloaded Medical Knowledge Base

The repository includes pre-indexed sample clinical documents across both `.pdf` and `.txt` formats:

### PDF Clinical Guidelines & Monographs
1. `diabetes_guidelines.pdf`: Type 2 Diabetes diagnostic criteria, Metformin first-line dosing, SGLT2i/GLP-1 RA cardiorenal indications, HbA1c targets, and Hypoglycemia "Rule of 15".
2. `clinical_guidelines.pdf`: Hypertension stages, first-line antihypertensive classes (ACEi, ARB, CCB), target BP recommendations, and guideline variations.
3. `cardiology_guidelines.pdf`: Heart Failure with reduced ejection fraction (HFrEF), GDMT Quadruple Therapy (ARNI, Beta-blockers, MRA, SGLT2i), and Acute Coronary Syndrome DAPT protocols.
4. `drug_information.pdf`: Detailed monographs for Metformin, Lisinopril, Empagliflozin, and Atorvastatin (including eGFR contraindications and adverse reactions).
5. `medical_research.pdf`: Clinical trial evidence reviews (DAPA-HF, EMPA-REG) and comparative efficacy analyses.

### Text Format Clinical Protocols (`.txt`)
6. `emergency_protocols.txt`: ACLS resuscitation algorithms, Epinephrine 1:1000 IM dosing for anaphylaxis, and Acute Ischemic Stroke thrombolysis windows (4.5 hours).
7. `pediatric_guidelines.txt`: Pediatric acute otitis media first-line high-dose Amoxicillin dosing (80-90 mg/kg/day) and dehydration assessment & ORS/IV fluid protocols.
8. `oncology_clinical_pathways.txt`: Immune checkpoint inhibitor adverse reaction grading (irAEs), corticosteroid dosing algorithms, and febrile neutropenia empiric antipseudomonal antibiotic regimens.

---

## 7. Project Directory Structure

```
medical-rag-chatbot/
│
├── frontend/
│   ├── index.html            # Clinical assistant UI with citation drawer
│   ├── style.css             # High-end clinical design system (Plus Jakarta Sans)
│   └── app.js                # Session management, API client, markdown parser
│
├── backend/
│   ├── main.py               # FastAPI entrypoint, CORS, static mounting
│   ├── config.py             # Pydantic Settings (.env configuration)
│   │
│   ├── api/
│   │   ├── chat.py           # POST /api/chat (Two-stage RAG execution)
│   │   └── conversations.py  # Session CRUD endpoints
│   │
│   ├── rag/
│   │   ├── chain.py          # End-to-end RAG pipeline orchestrator
│   │   ├── retriever.py      # ChromaDB vector search
│   │   ├── reranker.py       # BGE Cross-Encoder reranker
│   │   ├── embeddings.py     # BGE Embeddings adapter
│   │   ├── prompts.py        # Strict Medical RAG system prompt
│   │   └── context.py        # Context builder & citation extractor
│   │
│   ├── llm/
│   │   └── groq_client.py    # Groq Chat client with fallback
│   │
│   └── database/
│       ├── database.py       # SQLite engine & repository
│       └── models.py         # SQLAlchemy Conversation & Message models
│
├── ingestion/
│   ├── loader.py             # PyMuPDF extractor with page boundary preservation
│   ├── cleaner.py            # Medical text normalizer
│   ├── chunker.py            # Semantic chunker with metadata inheritance
│   ├── embedder.py           # BGE embeddings batch embedder
│   ├── chroma_store.py       # ChromaDB vector store manager & indexing CLI
│   └── generate_sample_pdfs.py # Clinical guidelines PDF generator
│
├── medical_documents/        # Directory containing preloaded medical PDFs
│   ├── clinical_guidelines.pdf
│   ├── diabetes_guidelines.pdf
│   ├── cardiology_guidelines.pdf
│   ├── drug_information.pdf
│   └── medical_research.pdf
│
├── evaluation/
│   ├── dataset.json          # Benchmark dataset (Single-doc, Multi-doc, Conflicting, Negative)
│   └── evaluate.py           # Quantitative RAG evaluation suite
│
├── tests/
│   ├── test_ingestion.py     # Ingestion & chunking unit tests
│   ├── test_rag.py           # Context & prompt tests
│   └── test_api.py           # FastAPI integration tests
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 8. Installation & Quickstart

### Prerequisites
* Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.13)
* A free [Groq API Key](https://console.groq.com)

### Step 1: Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate virtual environment
source venv/bin/activate
```

---

### Step 2: Install Dependencies

```powershell
pip install -r requirements.txt
```
Edit `.env` and set your `GROQ_API_KEY`:
```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

### 3. Generate Knowledge Base & Run Ingestion
```bash
python -m ingestion.chroma_store --reset
```
This will automatically generate the clinical guideline PDFs in `medical_documents/` and index all chunks into `chroma_db/`.

### 4. Start the Application Server
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser at:
👉 **`http://127.0.0.1:8000`**

---

## 9. API Reference

### 1. Send Medical Query
`POST /api/chat`

**Request:**
```json
{
  "conversation_id": "conv_001",
  "message": "What is the recommended first-line treatment for Type 2 Diabetes and the target HbA1c?"
}
```

**Response:**
```json
{
  "conversation_id": "conv_001",
  "answer": "According to the provided clinical guidelines, Metformin is the preferred initial pharmacological agent...",
  "sources": [
    {
      "document": "diabetes_guidelines.pdf",
      "page": 2,
      "section": "Pharmacological Management",
      "chunk_id": "diabetes_guidelines_p2_c1",
      "relevance_score": 0.945,
      "snippet": "First-Line Pharmacotherapy: Metformin remains the preferred initial pharmacological agent..."
    },
    {
      "document": "diabetes_guidelines.pdf",
      "page": 3,
      "section": "Glycemic Targets",
      "chunk_id": "diabetes_guidelines_p3_c1",
      "relevance_score": 0.892,
      "snippet": "Glycemic Goals: An HbA1c goal of < 7.0% is recommended for most non-pregnant adult patients..."
    }
  ],
  "latency_ms": 412.5
}
```

### 2. Health & Knowledge Base Status
`GET /api/health`

**Response:**
```json
{
  "status": "healthy",
  "service": "Medical RAG Assistant API",
  "llm_model": "llama-3.3-70b-versatile",
  "groq_configured": true,
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "reranker_model": "BAAI/bge-reranker-base",
  "use_reranker": true,
  "vector_store": {
    "status": "healthy",
    "indexed_chunks": 18,
    "persist_directory": "chroma_db"
  }
}
```

### 3. Conversations Management
* `GET /api/conversations` — List all consultations
* `GET /api/conversations/{id}` — Fetch conversation history with messages & citations
* `DELETE /api/conversations/{id}` — Delete a consultation session

---

## 10. Quantitative RAG Evaluation

Run the evaluation benchmark:
```bash
python evaluation/evaluate.py
```

### Key Metrics Tracked:
* **Recall@10**: Proportion of test queries where the ground-truth document and page were retrieved in the initial dense vector search.
* **Top-4 Cross-Encoder Precision**: Proportion of queries where the relevant evidence was retained in the top-4 reranked context passed to the LLM.
* **Mean Reciprocal Rank (MRR)**: Evaluates how high the correct clinical evidence ranked in the candidate pool.
* **Zero-Hallucination Rejection Accuracy**: Verification that out-of-scope or ungrounded queries are rejected with standard disclaimers rather than fabricated answers.

---

## 11. Observability (LangSmith Tracing Integration)

The pipeline integrates natively with LangSmith for production LLM distributed tracing and telemetry.

To enable LangSmith:
```ini
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_your_langsmith_api_key_here
LANGCHAIN_PROJECT=medical-rag-chatbot
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### What LangSmith Traces:
* **End-to-End Latency Traces**: Vector retrieval, Cross-Encoder reranking, and Groq token generation times.
* **Retrieved & Reranked Context Blocks**: Complete inspection of chunk text, similarity scores, and reranker probabilities.
* **System & Grounded Prompts**: Verbatim prompts submitted to Groq.
* **Token Usage & Costs**: Real-time prompt tokens, completion tokens, and token throughput per consultation.
* **Metadata & Tags**: Grouped by `conversation_id`, query category, and deployment environment.

---

## 12. Automated Test Suite

Run pytest:
```bash
pytest tests/ -v
```
Tests cover text cleaning, PyMuPDF page loading, metadata preservation across chunking, context builder formatting, prompt construction, and FastAPI endpoint lifecycle.
