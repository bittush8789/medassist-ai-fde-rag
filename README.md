# MedAssist AI FDE RAG — Enterprise Clinical Intelligence Platform

> **A production-grade, evidence-grounded Clinical Knowledge Assistant built with Two-Stage Retrieval (ChromaDB + BGE Cross-Encoder), Role-Based Access Control (RBAC), Multi-Tenant Document Isolation, and Ultra-Low Latency Groq LLM Inference.**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg?logo=python&logoColor=white)](https://www.python.org)
[![LangChain](https://img.shields.io/badge/LangChain-RAG%20Orchestrator-1C3C3C.svg?logo=langchain&logoColor=white)](https://python.langchain.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3%20Inference-F05A28.svg?logo=groq&logoColor=white)](https://groq.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF4F00.svg)](https://www.trychroma.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-BGE%20Small%20v1.5-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co/BAAI/bge-small-en-v1.5)
[![CrossEncoder](https://img.shields.io/badge/CrossEncoder-BGE%20Reranker%20Base-10B981.svg)](https://huggingface.co/BAAI/bge-reranker-base)
[![Tests](https://img.shields.io/badge/Pytest-13%2F13%20Passed-brightgreen.svg?logo=pytest&logoColor=white)](tests/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](Dockerfile)
[![System Design](https://img.shields.io/badge/Architecture-System%20Design%20Doc-8B5CF6.svg)](SYSTEM_DESIGN.md)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📑 Table of Contents

- [Overview & Clinical Problem](#-overview--clinical-problem)
- [Application Screenshots](#-application-screenshots)
- [Key Features & Capabilities](#-key-features--capabilities)
- [System Architecture](#-system-architecture)
- [Role-Based Access Control (RBAC) & Security](#-role-based-access-control-rbac--security)
- [Two-Stage RAG Pipeline Specification](#-two-stage-rag-pipeline-specification)
- [Pre-Indexed Medical Knowledge Base (25 Documents)](#-pre-indexed-medical-knowledge-base-25-documents)
- [Project Directory Structure](#-project-directory-structure)
- [Getting Started & Quickstart](#-getting-started--quickstart)
  - [Local Setup](#1-local-development-setup)
  - [Docker Setup](#2-docker--container-deployment)
- [REST API Reference](#-rest-api-reference)
- [Evaluation & Benchmark Results](#-evaluation--benchmark-results)
- [Observability & Diagnostics](#-observability--diagnostics)
- [Automated Testing Suite](#-automated-testing-suite)
- [Production Deployment Topology](#-production-deployment-topology)
- [License & Acknowledgements](#-license--acknowledgements)

---

## 🏥 Overview & Clinical Problem

Healthcare networks, hospital systems, and clinical research groups manage tens of thousands of pages of clinical practice guidelines, pharmacopeia drug monographs, emergency protocols, and customer-specific formularies.

### The Operational Challenge
1. **High Retrieval Latency**: Clinicians spend up to 20% of their time searching multi-page PDFs to locate critical drug interactions, pediatric weight-based dosing, and emergency algorithms.
2. **Generative Hallucination Risks**: General-purpose LLMs hallucinate dosages, fabricate phantom clinical trials, and conflate conflicting institutional protocols.
3. **Data Boundary & Multi-Tenant Compliance**: Hospital systems require strict role and tenant boundaries where proprietary formularies or internal engineering operations are protected against unauthorized queries and cross-tenant leakage.

### The AI FDE Solution
**MedAssist** is an enterprise-grade, conversational clinical assistant that:
* Pre-indexes **25 verified medical guidelines** into **219 dense semantic chunks**.
* Executes **Retrieval-Level Vector Pre-Filtering** so unauthorized documents are never retrieved.
* Employs **Two-Stage Retrieval**: Dense vector similarity (`bge-small-en-v1.5`) + Cross-Encoder deep reranking (`bge-reranker-base`).
* Generates strictly grounded, evidence-backed clinical summaries with exact **document and page citations** using Groq's high-speed inference engine.

---

## 📸 Application Screenshots

### 1. Chat Interface — Grounded Clinical Q&A with Citations
> The primary clinical consultation interface showing a query regarding Pediatric Acute Otitis Media. The assistant returns a **grounded response with drug dosing tables, verified citations with page numbers, and medical disclaimers**.

![Chat Interface — Clinical Q&A with Verified Citations](Photo/chat-interface.png)

---

### 2. Knowledge Base Library — 25 Pre-Indexed Clinical Documents
> The **Reference Library Explorer** displaying all 25 pre-indexed medical guidelines, pharmacopeia monographs, emergency protocols, and internal runbooks with 1-click query suggestions.

![Knowledge Base Library — 25 Indexed Documents](Photo/knowledge-base-library.png)

---

### 3. RBAC Authentication & 1-Click Persona Switcher
> The **Authentication & Role-Based Access Control** modal featuring 1-click Quick-Login personas (`ADMIN`, `FDE_ENGINEER`, `CUSTOMER 1`, `CUSTOMER 2`) plus custom credentials login with stateless JWT verification.

![RBAC Authentication — Quick-Login Persona Switcher](Photo/rbac-auth-modal.png)

---

### 4. Enterprise Admin Console — User Management & Audit Logs
> The **Admin Governance Panel** for user account provisioning, role/tenant assignment, and a live stream of **Security Audit Logs** (`LOGIN`, `RAG_QUERY`, `UNAUTHORIZED_ACCESS`).

![Admin Console — User Management & Governance](Photo/admin-console.png)

---

## ⚡ Key Features & Capabilities

* **🔒 Retrieval-Level RBAC Enforcement**: Authorization filters (`where={"$and": [...]}`) run directly at the vector store query layer, preventing data leakage before LLM context construction.
* **🎯 Two-Stage Retrieval Subsystem**:
  * *Stage 1*: ChromaDB dense vector search ($k=10$ candidate chunks).
  * *Stage 2*: BAAI BGE Cross-Encoder reranker ($k=4$ highest-scored chunks).
* **⚡ Ultra-Low Latency Inference**: Groq LPU acceleration achieves an end-to-end SLA of `< 1800ms` (Retrieval: ~110ms, Rerank: ~280ms, LLM Synthesis: ~1000ms).
* **🛡️ Zero-Hallucination Gate**: Strict prompt guardrails enforce context-only answers with automatic zero-exposure negative fallback on ungrounded queries.
* **🔑 Enterprise Security**: Bcrypt password hashing (12 salt rounds), stateless JWT tokens (HS256), and security audit logging.
* **📊 AI FDE Live Diagnostics**: Interactive telemetry drawer displaying latency waterfalls, candidate chunk count, and active RBAC clearance tags.
* **🎨 Modern Responsive UI**: Vanilla CSS design system with Dark/Light mode, consultation history management, notes export, and voice search support.
* **🧪 100% Automated Test Coverage**: Comprehensive pytest test suite covering ingestion, RAG pipeline, RBAC authorization, and API contracts.

---

## 📐 System Architecture

> 📘 *For comprehensive diagrams, sequence flows, and disaster recovery runbooks, refer to the [System Design Document](SYSTEM_DESIGN.md).*

```
                              [ 1. User Web Client ]
                 (HTML5 / Modern Vanilla CSS / JS / FDE Diagnostics)
                                        │
                                        ▼
                       [ 2. FastAPI Gateway & Auth ]
                 (Stateless JWT Validation / Bcrypt Password Hash)
                                        │
               ┌────────────────────────┼────────────────────────┐
               ▼                        ▼                        ▼
        ┌─────────────┐        ┌──────────────────┐       ┌──────────────┐
        │   `ADMIN`   │        │  `FDE_ENGINEER`  │       │  `CUSTOMER`  │
        └──────┬──────┘        └────────┬─────────┘       └──────┬───────┘
               │                        │                        │
               └────────────────────────┼────────────────────────┘
                                        ▼
                          [ 3. LangChain RAG Pipeline ]
                                        │
               ┌────────────────────────┴────────────────────────┐
               ▼                                                 ▼
      [ 4. ChromaDB Vector Store ]                    [ 5. SQLite Multi-Tenant ]
     (219 Chunks, 384-d Embeddings)                     (Users, Audit Logs,
    `where={"$and": [...], "$or": [...]}`                  Conversations)
               │
               ▼
      [ 6. BGE Cross-Encoder ]
      (BAAI/bge-reranker-base)
      (Top-10 Candidates ➔ Top-4)
               │
               ▼
      [ 7. Groq Inference Engine ]
      (LLaMA 3.3 70B Versatile)
      (Zero-Hallucination Gate)
               │
               ▼
      [ 8. Structured JSON Response + Page Citations + Telemetry ]
```

---

## 🛡️ Role-Based Access Control (RBAC) & Security

### Role Clearance & Access Matrix

| Feature / Resource | `CUSTOMER` | `FDE_ENGINEER` | `ADMIN` |
| :--- | :---: | :---: | :---: |
| **Public Clinical Guidelines (22 Docs)** | ✅ Read | ✅ Read | ✅ Read |
| **Assigned Customer Tenant Docs (`tenant_id`)** | ✅ Read | ✅ Read | ✅ Read |
| **Other Customer Tenant Docs** | ❌ Blocked (Zero Leak) | ❌ Blocked (Zero Leak) | ✅ Read |
| **Internal FDE Runbooks & SOPs** | ❌ Blocked (Zero Leak) | ✅ Read | ✅ Read |
| **User Management (Create/Delete Users)** | ❌ 403 Forbidden | ❌ 403 Forbidden | ✅ Full CRUD |
| **Security Audit Logs** | ❌ 403 Forbidden | ❌ 403 Forbidden | ✅ Read Only |
| **Own Conversation History** | ✅ Read / Write | ✅ Read / Write | ✅ Read / Write |
| **Other Users' Conversations** | ❌ Blocked | ❌ Blocked | ❌ Blocked |

### Pre-Configured Demo Credentials

| Persona | Username | Password | Role | Tenant ID | Clearance Level |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Platform Administrator** | `admin` | `Admin@12345` | `ADMIN` | `system` | Full platform access, user CRUD, audit logs, all 25 documents |
| **Lead FDE Engineer** | `engineer` | `Engineer@12345` | `FDE_ENGINEER` | `customer_001` | Query RAG, internal SOPs + MetroHealth hospital tenant |
| **MetroHealth Hospital** | `customer1` | `Customer@12345` | `CUSTOMER` | `customer_001` | Isolated to `customer_001` formulary and public guidelines |
| **Apex Clinical Network** | `customer2` | `Customer@12345` | `CUSTOMER` | `customer_002` | Isolated to `customer_002` guidelines and public guidelines |

### Vector Pre-Filtering Implementation
```python
# Customer clearance filter (strictly excludes internal docs)
where = {
    "$and": [
        {"$or": [{"tenant_id": "customer_001"}, {"tenant_id": "all"}]},
        {"classification": {"$ne": "internal"}}
    ]
}

# FDE Engineer clearance filter (includes internal runbooks + assigned tenant)
where = {
    "$or": [
        {"tenant_id": "customer_001"},
        {"tenant_id": "all"},
        {"classification": "internal"}
    ]
}
```

---

## 🔍 Two-Stage RAG Pipeline Specification

```mermaid
sequenceDiagram
    autonumber
    actor User as Clinician / Engineer
    participant API as FastAPI Gateway
    participant Auth as JWT & RBAC Guard
    participant Chroma as ChromaDB Vector Store
    participant Rerank as BGE Cross-Encoder
    participant LLM as Groq LLaMA 3.3
    participant DB as SQLite Database

    User->>API: POST /api/chat (Query + Bearer Token)
    API->>Auth: Validate JWT & Extract (Role, Tenant)
    Auth-->>API: Authorized User Context
    API->>Chroma: Vector Similarity Search (Top-10 + RBAC Where Filter)
    Chroma-->>API: 10 Candidate Chunks
    API->>Rerank: Cross-Attention Scoring (Query, Chunk Pairs)
    Rerank-->>API: Top-4 High-Relevance Chunks
    API->>LLM: Context-Grounded Prompt with Page Citations
    LLM-->>API: Grounded Answer Synthesis
    API->>DB: Persist Messages & Telemetry
    API-->>User: Structured Answer + Page Citations + Telemetry
```

### Retrieval Hyperparameters
* **Chunking Strategy**: Recursive Character Splitting (`chunk_size=600`, `chunk_overlap=100`, page boundary preserved).
* **Embedding Model**: `BAAI/bge-small-en-v1.5` (384-dimensional dense vectors, normalized).
* **Vector Distance Metric**: Cosine Similarity.
* **Reranker Model**: `BAAI/bge-reranker-base` (Cross-Encoder relevance threshold: `0.60`).
* **Generation Model**: `llama-3.3-70b-versatile` on Groq LPUs (`temperature=0.1`, `max_tokens=1024`).

---

## 📚 Pre-Indexed Medical Knowledge Base (25 Documents)

| Category | Document Name | File | Primary Clinical Topics |
| :--- | :--- | :--- | :--- |
| **Cardiology** | Hypertension Guidelines | `clinical_guidelines.pdf` | Stage 1/2 HTN, first-line ACEi/ARB/CCB, target BP `<130/80`. |
| **Cardiology** | Heart Failure Guide | `cardiology_guidelines.pdf` | HFrEF GDMT Quadruple therapy (ARNI, Beta-blocker, MRA, SGLT2i). |
| **Endocrinology** | Diabetes Clinical Guidelines | `diabetes_guidelines.pdf` | T2DM diagnostic criteria, Metformin titration, Rule of 15. |
| **Pharmacology** | Medication Safety Manual | `drug_information.pdf` | Metformin eGFR cutoffs, Lisinopril contraindications, Atorvastatin. |
| **Evidence Base** | Clinical Research Evidence | `medical_research.pdf` | DAPA-HF, EMPA-REG OUTCOME clinical trial syntheses. |
| **Emergency** | Resuscitation & Shock Protocols | `emergency_protocols.txt` | ACLS cardiac arrest algorithms, Epinephrine 1:1000 IM anaphylaxis. |
| **Pediatrics** | Child Health Guidelines | `pediatric_guidelines.txt` | Pediatric AOM Amoxicillin 80–90 mg/kg/day, ORS fluid plans. |
| **Oncology** | Cancer Care Pathways | `oncology_clinical_pathways.txt` | Immune checkpoint irAE grading, Febrile Neutropenia bundle. |
| **Infectious Dis.** | Sepsis & Infectious Disease | `infectious_disease_guidelines.txt` | 1-Hour Sepsis Bundle, CURB-65 pneumonia, C. diff Vancomycin. |
| **Neurology** | Neurology & Mental Health | `neurology_and_psychiatry_guidelines.txt` | Acute ischemic stroke tPA (4.5h), Status epilepticus IV Lorazepam. |
| **Nephrology** | Kidney Health & CKD Guidelines | `nephrology_and_ckd_guidelines.txt` | CKD KDIGO staging, Hyperkalemia Calcium Gluconate emergency. |
| **Pulmonology** | Asthma & COPD Guidelines | `pulmonology_asthma_copd_guidelines.txt` | GINA Stepwise asthma, COPD GOLD staging, SABA/LAMA/LABA. |
| **Gastroenterology**| Gastroenterology & Liver Guide | `gastroenterology_and_hepatology_guidelines.txt` | Acute Upper GI Bleed IV PPI, Octreotide, Hepatic Encephalopathy. |
| **Endocrine Em.** | Endocrine Emergencies Guide | `endocrinology_thyroid_and_adrenal_guidelines.txt` | DKA protocol (0.1 U/kg/h IV insulin, potassium rule), Thyroid storm. |
| **Rheumatology** | Autoimmune & Joint Care | `rheumatology_and_autoimmune_guidelines.txt` | Rheumatoid Arthritis Methotrexate + Folic acid, Acute Gout. |
| **OB/GYN** | Obstetrics & Women's Health | `obstetrics_and_gynecology_guidelines.txt` | Severe Preeclampsia IV Magnesium Sulfate, Postpartum Hemorrhage. |
| **Dermatology** | Dermatology & Wound Care | `dermatology_and_wound_care_guidelines.txt` | Stevens-Johnson Syndrome SCORTEN, Diabetic Foot Ulcer Wagner. |
| **Hematology** | Anticoagulation & Blood Guide | `hematology_and_anticoagulation_guidelines.txt` | DVT/PE DOAC dosing, Heparin-Induced Thrombocytopenia (HIT). |
| **Geriatrics** | Geriatric Care & Palliative | `geriatrics_and_palliative_care_guidelines.txt` | Beers Criteria contraindications, Palliative opioid rotation. |
| **Toxicology** | Poisoning & Overdose Protocols | `toxicology_and_poisoning_guidelines.txt` | Acetaminophen NAC protocol (Rumack-Matthew), Opioid Naloxone. |
| **Trauma** | Trauma Life Support & Ortho | `orthopedics_and_trauma_guidelines.txt` | ATLS Primary Survey (ABCDE), Massive Transfusion (1:1:1), Compartment. |
| **ENT & Eye** | Eye, Ear, Nose & Throat Em. | `ophthalmology_and_ent_emergencies.txt` | Acute Angle-Closure Glaucoma Timolol/Acetazolamide, Epistaxis. |
| **Customer 001** | MetroHealth Formulary | `customer_001_formulary_guidelines.txt` | MetroHealth exclusive SGLT2i formulary (Empagliflozin 10mg preferred). |
| **Customer 002** | Apex Clinic Guidelines | `customer_002_formulary_guidelines.txt` | Apex Clinic HEC Quadruple antiemetic bundle (Olanzapine/Aprepitant). |
| **Internal FDE** | AI FDE Operations Runbook | `internal_fde_troubleshooting_runbook.txt` | ChromaDB cluster failover command, BGE drift threshold (>0.08). |

---

## 📁 Project Directory Structure

```
medical-chatbot/
├── .env.example                     # Environment template with config docs
├── .gitignore                       # Git ignore rules (protects .env & SQLite)
├── Dockerfile                       # Production multi-stage Docker build
├── docker-compose.yml               # Container orchestration
├── requirements.txt                 # Pinned dependencies
├── README.md                        # Primary project documentation
├── SYSTEM_DESIGN.md                 # Complete System Design Document
│
├── backend/                         # FastAPI Application Backend
│   ├── main.py                      # Application entrypoint & static file mount
│   ├── config.py                    # Pydantic Settings (.env configuration)
│   ├── api/                         # REST API Route Controllers
│   │   ├── auth.py                  # POST /api/auth/login, /me, /logout
│   │   ├── chat.py                  # POST /api/chat (Two-stage RAG execution)
│   │   ├── conversations.py         # Session CRUD & history management
│   │   └── users.py                 # Admin User Management & Audit Logs
│   ├── auth/                        # Security & Access Control Subsystem
│   │   ├── security.py              # Bcrypt hashing & PyJWT token generator
│   │   └── dependencies.py          # FastAPI dependencies (get_current_user, require_roles)
│   ├── database/                    # Persistence Layer
│   │   ├── database.py              # SQLite repositories & default seeding
│   │   └── models.py                # SQLAlchemy Models (Users, AuditLogs, Chats)
│   ├── llm/                         # LLM Client Integrations
│   │   └── groq_client.py           # Groq Chat client with resilient model fallback
│   └── rag/                         # Core RAG Orchestration Engine
│       ├── chain.py                 # End-to-end RAG orchestrator & telemetry
│       ├── retriever.py             # ChromaDB retriever with RBAC where filtering
│       ├── reranker.py              # BGE Cross-Encoder reranker
│       ├── embeddings.py            # BGE Embeddings adapter
│       ├── context.py               # Citation formatter & source assembler
│       └── prompts.py               # Strict clinical system prompt & guardrails
│
├── frontend/                        # Web Client Presentation Layer
│   ├── index.html                   # Semantic HTML5 clinical interface
│   ├── style.css                    # Modern CSS design system (Dark/Light themes)
│   └── app.js                       # Client controller, JWT manager & markdown parser
│
├── ingestion/                       # Knowledge Pre-Indexing Subsystem
│   ├── loader.py                    # PyMuPDF & TXT document loader
│   ├── cleaner.py                   # Medical text normalizer
│   ├── chunker.py                   # Semantic chunker with RBAC metadata tagging
│   ├── embedder.py                  # BGE batch embedding generator
│   ├── chroma_store.py              # ChromaDB vector store manager & CLI
│   └── generate_sample_pdfs.py      # Automated PDF guideline generator
│
├── medical_documents/               # 25 Clinical Guideline & Protocol Source Files
│
├── evaluation/                      # RAG Benchmark & Testing Suite
│   ├── dataset.json                 # Quantitative golden test dataset
│   ├── evaluate.py                  # Automated RAG evaluation runner
│   └── evaluation_results.json      # Benchmark execution metrics
│
├── tests/                           # Automated Test Suite (PyTest)
│   ├── conftest.py                  # Test fixtures & database isolation
│   ├── test_api.py                  # REST API endpoint integration tests
│   ├── test_ingestion.py            # Ingestion & chunking unit tests
│   ├── test_rag.py                  # Retrieval & prompt guardrail tests
│   └── test_rbac.py                 # Security, JWT, & RBAC authorization tests
│
└── Photo/                           # Application Screenshots & Media Assets
    ├── chat-interface.png
    ├── knowledge-base-library.png
    ├── rbac-auth-modal.png
    └── admin-console.png
```

---

## 🚀 Getting Started & Quickstart

### 1. Local Development Setup

#### Prerequisites
* Python 3.10+ installed
* A free [Groq API Key](https://console.groq.com)

#### Step 1: Clone Repository
```bash
git clone https://github.com/bittush8789/medassist-ai-fde-rag.git
cd medassist-ai-fde-rag
```

#### Step 2: Set Up Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Configure Environment Variables
Create a `.env` file in the project root (or copy from `.env.example`):
```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-base
USE_RERANKER=true
JWT_SECRET=your-secure-jwt-secret-key-here
```

#### Step 5: Ingest Documents into Vector Store
```bash
python -m ingestion.chroma_store --reset
```
*Indexes all 25 documents (219 chunks) into `chroma_db/` with RBAC metadata.*

#### Step 6: Start Application Server
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser!

---

### 2. Docker & Container Deployment

Run the entire platform using Docker Compose:
```bash
docker-compose up --build -d
```
Access the application at `http://localhost:8000`.

---

## 🔌 REST API Reference

### Authentication Endpoints

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :---: |
| `POST` | `/api/auth/login` | Authenticate user & issue JWT Bearer token | No |
| `GET` | `/api/auth/me` | Retrieve authenticated user profile & claims | Yes |
| `POST` | `/api/auth/logout` | Invalidate session & log audit event | Yes |

#### Sample Request (`POST /api/auth/login`):
```json
{
  "username": "customer1",
  "password": "Customer@12345"
}
```

---

### Chat & Clinical Consultation Endpoints

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :---: |
| `POST` | `/api/chat` | Submit medical query and execute Two-Stage RAG | Yes |

#### Sample Request (`POST /api/chat`):
```json
{
  "message": "What is the recommended first-line therapy for HFrEF?",
  "conversation_id": "optional-uuid"
}
```

#### Sample Response (`POST /api/chat`):
```json
{
  "answer": "According to the Cardiology Guidelines, first-line GDMT for HFrEF consists of four foundational drug classes: 1. ARNI (Sacubitril/Valsartan), 2. Evidence-based Beta-Blockers (Carvedilol, Metoprolol Succinate), 3. Mineralocorticoid Receptor Antagonists (Spironolactone), and 4. SGLT2 Inhibitors (Empagliflozin, Dapagliflozin).",
  "sources": [
    {
      "document": "cardiology_guidelines.pdf",
      "page": 2,
      "chunk_id": "cardio_p2_c1",
      "excerpt": "Guideline-directed medical therapy (GDMT) quadruple therapy..."
    }
  ],
  "latency_ms": 1412.5,
  "telemetry": {
    "retrieval_ms": 110.2,
    "rerank_ms": 285.4,
    "llm_ms": 1016.9,
    "model": "llama-3.3-70b-versatile",
    "candidates_count": 10,
    "retrieval_count": 4,
    "rbac_role": "CUSTOMER",
    "rbac_tenant": "customer_001"
  }
}
```

---

### Conversations Management

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :---: |
| `GET` | `/api/conversations` | List user's consultation sessions | Yes |
| `GET` | `/api/conversations/{id}` | Fetch session history & citations | Yes |
| `DELETE` | `/api/conversations/{id}` | Delete a consultation session | Yes |

---

### Administration & Security Governance (Admin Only)

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :---: |
| `GET` | `/api/users` | List all provisioned user accounts | `ADMIN` Role |
| `POST` | `/api/users` | Create new user account with role & tenant | `ADMIN` Role |
| `DELETE` | `/api/users/{id}` | Delete a user account | `ADMIN` Role |
| `GET` | `/api/audit-logs` | Fetch real-time security audit log feed | `ADMIN` Role |

---

### System Health & Diagnostics

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :---: |
| `GET` | `/api/health` | Health check, vector store state & chunk count | No |

---

## 📊 Evaluation & Benchmark Results

Run the quantitative evaluation benchmark:
```bash
python -m evaluation.evaluate
```

### Benchmark Metrics Summary:
* **Dense Retrieval Recall@10**: `96.4%` (Correct evidence retrieved in top-10 candidate pool)
* **Cross-Encoder Reranking Precision@4**: `94.1%` (Top-4 context chunks contain ground truth)
* **Mean Reciprocal Rank (MRR)**: `0.912`
* **Zero-Hallucination Rejection Accuracy**: `100.0%` (Zero document leakage on restricted queries)
* **Average Response Latency**: `1380ms`

---

## 📈 Observability & Diagnostics

### Live AI FDE Diagnostics Mode
Click **⚡ FDE Mode** in the navigation header to open real-time diagnostic waterfalls:
* **Vector Retrieval Latency**: Time to compute query embedding and query ChromaDB.
* **Cross-Encoder Latency**: Joint cross-attention reranking duration.
* **LLM Synthesis Latency**: Time-to-complete generation on Groq LPUs.
* **RBAC Clearance Tag**: Active clearance verification (`[ROLE] [@TENANT]`).

### Optional LangSmith Integration
Enable distributed tracing in `.env`:
```ini
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_your_key_here
LANGCHAIN_PROJECT=medassist-ai-fde-rag
```

---

## 🧪 Automated Testing Suite

The repository includes a comprehensive test suite executed with `pytest`:

```bash
pytest tests/ -v
```

### Test Coverage Breakdown (13 Tests Passing):
* **`tests/test_rbac.py`**:
  * Bcrypt salt hashing and verification.
  * JWT creation, custom claims, and expiration lifecycle.
  * Endpoint authorization enforcement (`401 Unauthorized`, `403 Forbidden`).
  * ChromaDB RBAC `where` filter generation.
* **`tests/test_api.py`**:
  * End-to-end authenticated chat workflows.
  * Session isolation and conversation CRUD operations.
  * Health check diagnostic response validation.
* **`tests/test_rag.py`**:
  * Context builder formatting and citation extraction.
  * Strict medical prompt guardrails.
* **`tests/test_ingestion.py`**:
  * PyMuPDF extraction, text cleaner, and semantic chunking with metadata inheritance.

---

## 🌐 Production Deployment Topology

```mermaid
flowchart TD
    subgraph Ingress["Edge & Load Balancing"]
        DNS["Cloudflare DNS / CDN"]
        Nginx["Nginx Reverse Proxy & SSL Termination"]
    end

    subgraph AppCluster["Application Tier"]
        Worker1["FastAPI Instance 1\n(Gunicorn / Uvicorn)"]
        Worker2["FastAPI Instance 2\n(Gunicorn / Uvicorn)"]
    end

    subgraph StateStorage["Persistence & Cache"]
        Postgres[("PostgreSQL / SQLite Database\n(Users, Logs, Chats)")]
        RedisCache[("Redis Cache\n(Rate Limiting & Sessions)")]
    end

    subgraph AIServices["AI & Vector Infrastructure"]
        ChromaCluster[("ChromaDB Vector Store\n(Persistent Index)")]
        GroqAPI["Groq Cloud LPU\n(High-Throughput LLM)"]
    end

    DNS --> Nginx
    Nginx --> Worker1
    Nginx --> Worker2

    Worker1 --> Postgres
    Worker1 --> RedisCache
    Worker1 --> ChromaCluster
    Worker1 --> GroqAPI

    Worker2 --> Postgres
    Worker2 --> RedisCache
    Worker2 --> ChromaCluster
    Worker2 --> GroqAPI
```

---

## 📜 License & Acknowledgements

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

### Acknowledgements
* **[LangChain](https://github.com/langchain-ai/langchain)** for the foundational RAG abstractions.
* **[Groq](https://groq.com)** for ultra-fast LPU inference acceleration.
* **[ChromaDB](https://www.trychroma.com)** for lightweight embedded vector search.
* **[BAAI](https://huggingface.co/BAAI)** for state-of-the-art BGE embeddings and Cross-Encoder rerankers.
