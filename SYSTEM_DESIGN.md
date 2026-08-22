# AI FDE Medical RAG System Design Document

**Document Version:** `2.0.0`  
**Classification:** Enterprise Engineering Architecture  
**System Name:** MedAssist Clinical Intelligence Platform (AI FDE Medical RAG)  
**Target SLA:** `< 2000ms` Total Latency, `99.9%` Service Availability, Zero-Hallucination Strict Grounding  

---

## 1. Executive Overview & Problem Statement

### 1.1 Business & Clinical Context
Modern healthcare networks, clinical research institutes, and hospital systems maintain tens of thousands of pages of dynamic clinical guidelines, pharmacopeia drug monographs, emergency resuscitation protocols, and customer-specific formulary guidelines. 

Clinicians, medical researchers, and hospital administrators face three core operational challenges:
1. **High Information Retrieval Latency**: Clinicians spend up to 20% of their time navigating dense, multi-page PDFs to locate critical contraindications, pediatric weight-based dosages, and oncology regimens.
2. **Generative Hallucination Risks**: General-purpose LLMs hallucinate dosages, invent phantom citations, or conflate conflicting clinical trial consensus statements.
3. **Tenant & Data Clearance Boundaries**: Healthcare organizations require strict multi-tenant isolation where one hospital tenant cannot access another hospital's proprietary formulary or internal engineering operations.

### 1.2 Solution Scope
The **MedAssist AI FDE Medical RAG Platform** is an enterprise-grade, retrieval-augmented conversational intelligence system featuring:
* **Two-Stage Dense Retrieval + Cross-Encoder Reranking** for high precision evidence retrieval.
* **Retrieval-Level Role-Based Access Control (RBAC)** to enforce tenant and role isolation before vector retrieval and LLM context synthesis.
* **Stateless JWT Authentication & Password Hashing** (`bcrypt`).
* **Ultra-Fast LLM Inference** via Groq LPUs (`llama-3.3-70b-versatile`).
* **Real-time Forward Deployed Engineering (FDE) Diagnostics** with latency waterfalls and zero-hallucination guardrails.

---

## 2. High-Level System Architecture

```mermaid
flowchart TB
    subgraph ClientLayer["1. Client Presentation Layer"]
        UI["Web Client (HTML5 / Vanilla CSS / JS)"]
        AuthUI["Auth & Quick-Login Modal"]
        AdminUI["Admin Console & Audit Log Stream"]
        DiagUI["FDE Live Diagnostics Panel"]
    end

    subgraph GatewayLayer["2. API Gateway & Security Filter"]
        FastAPI["FastAPI Application Server"]
        JWTMw["JWT Bearer Token Validator"]
        RBACMw["Role & Tenant Access Guard"]
        AuditMw["Security Audit Logger"]
    end

    subgraph StorageLayer["3. Relational Persistence Layer (SQLite)"]
        UserTable[("Users Table\n(Hashed Passwords, Role, Tenant)")]
        AuditTable[("Audit Logs Table\n(Event, IP, Role, Tenant)")]
        ConvTable[("Conversations Table\n(Tenant Isolated)")]
        MsgTable[("Messages Table\n(Citations & Telemetry)")]
    end

    subgraph RagOrchestrator["4. LangChain RAG Pipeline"]
        Retriever["MedicalRetriever\n(RBAC Metadata Filter)"]
        Reranker["Cross-Encoder Reranker\n(BAAI/bge-reranker-base)"]
        Synthesizer["Prompt Synthesizer\n(Zero-Hallucination Gate)"]
    end

    subgraph VectorEngine["5. Vector Search Engine (ChromaDB)"]
        ChromaStore[("ChromaDB Vector Store\n(219 Semantic Chunks)")]
        BGE["Embedder\n(BAAI/bge-small-en-v1.5)"]
    end

    subgraph InferenceEngine["6. High-Speed Inference (Groq Cloud)"]
        GroqLLM["LLaMA 3.3 70B Versatile\n(~1200ms Synthesis)"]
    end

    %% Flow Connections
    UI -->|HTTP Requests + Bearer Token| FastAPI
    AuthUI -->|Login & Token Request| FastAPI
    AdminUI -->|User CRUD & Audit Ingestion| FastAPI
    
    FastAPI --> JWTMw
    JWTMw --> RBACMw
    RBACMw --> AuditMw

    AuditMw -->|Persist Logs & Users| StorageLayer
    RBACMw -->|Pass Authenticated User Context| RagOrchestrator

    RagOrchestrator -->|Query Embedding| BGE
    BGE -->|Dense Vector + RBAC Where Clause| ChromaStore
    ChromaStore -->|Top-10 Candidate Chunks| Retriever
    Retriever -->|Candidates + Query| Reranker
    Reranker -->|Top-4 Highest Scored Chunks| Synthesizer
    Synthesizer -->|Strict Context Grounded Prompt| GroqLLM
    GroqLLM -->|Grounded Stream Response| Synthesizer
    Synthesizer -->|Structured JSON + Citations + Telemetry| FastAPI
    FastAPI -->|Response Payload| UI
```

---

## 3. Data Architecture & Storage Schema

### 3.1 Relational Storage (SQLite Schema)

```mermaid
erDiagram
    USERS ||--o{ AUDIT_LOGS : generates
    USERS ||--o{ CONVERSATIONS : owns
    CONVERSATIONS ||--o{ MESSAGES : contains

    USERS {
        string id PK "UUID"
        string username UK "Unique login name"
        string email UK "User email address"
        string password_hash "Bcrypt salted hash"
        string full_name "Display name"
        string role "ADMIN | FDE_ENGINEER | CUSTOMER"
        string tenant_id "system | customer_001 | customer_002"
        datetime created_at "Timestamp"
    }

    AUDIT_LOGS {
        int id PK "Auto-increment ID"
        string user_id FK "Reference to USERS.id"
        string username "Snapshot of username"
        string role "Role at event time"
        string tenant_id "Tenant at event time"
        string action "LOGIN | RAG_QUERY | UNAUTHORIZED_ACCESS"
        string resource "Endpoint or document accessed"
        string status "SUCCESS | FORBIDDEN | FAILED"
        string details "Context or error details"
        string ip_address "Client IP address"
        datetime timestamp "Event timestamp"
    }

    CONVERSATIONS {
        string id PK "UUID"
        string title "Conversation subject"
        string user_id FK "Owner User ID"
        string tenant_id "Tenant Isolation ID"
        datetime created_at "Creation timestamp"
        datetime updated_at "Last activity timestamp"
    }

    MESSAGES {
        string id PK "UUID"
        string conversation_id FK "Parent conversation"
        string role "user | assistant | system"
        string content "Message text content"
        string sources_json "JSON array of chunk citations"
        string telemetry_json "JSON object of latency waterfall"
        datetime created_at "Message timestamp"
    }
```

### 3.2 Vector Database Schema (ChromaDB)

* **Embedding Model**: `BAAI/bge-small-en-v1.5` (384-dimensional dense vectors, normalized).
* **Distance Metric**: Cosine Similarity ($1 - \text{cosine\_distance}$).
* **Indexed Units**: 219 Semantic Chunks across 25 verified clinical documents.
* **Vector Metadata Schema**:
  ```json
  {
    "doc_id": "cust001_formulary_p1_c0",
    "document": "customer_001_formulary_guidelines.txt",
    "page": 1,
    "total_pages": 3,
    "chunk_index": 0,
    "tenant_id": "customer_001",
    "classification": "customer",
    "access_roles": "[\"CUSTOMER\", \"FDE_ENGINEER\", \"ADMIN\"]",
    "document_type": "customer_guideline",
    "char_count": 582
  }
  ```

---

## 4. Ingestion & Pre-Indexing Pipeline

```mermaid
sequenceDiagram
    autonumber
    actor Admin as FDE Ingestion Engineer
    participant Parser as PyMuPDF & TXT Parser
    participant Chunker as Recursive Character Splitter
    participant Embedder as BGE Small v1.5 Embedder
    participant Chroma as ChromaDB Vector Store

    Admin->>Parser: Ingest 25 Medical Documents (PDFs & TXTs)
    Parser->>Chunker: Extract Clean Text + Page Trackers
    Note over Chunker: Chunk Size: 600 chars<br/>Chunk Overlap: 100 chars<br/>Separators: [\n\n, \n, ., " "]
    Chunker->>Chunker: Attach RBAC Metadata Tags (tenant_id, classification, roles)
    Chunker->>Embedder: Generate 384-d Dense Embeddings
    Embedder->>Chroma: Bulk Upsert (219 Chunks + Vector Embeddings)
    Chroma-->>Admin: Indexing Complete (Status: HEALTHY, 219 chunks)
```

### Ingestion Metadata Rules:
1. **Standard Clinical Guidelines**: `tenant_id = "all"`, `classification = "public"`, `access_roles = ["CUSTOMER", "FDE_ENGINEER", "ADMIN"]`.
2. **Customer Tenant Guidelines**: `tenant_id = "customer_001"`, `classification = "customer"`, `access_roles = ["CUSTOMER", "FDE_ENGINEER", "ADMIN"]`.
3. **Internal FDE Operations Runbook**: `tenant_id = "system"`, `classification = "internal"`, `access_roles = ["FDE_ENGINEER", "ADMIN"]`.

---

## 5. Two-Stage Retrieval & Reranking Architecture

```mermaid
flowchart LR
    subgraph QueryIngress["Query Ingress"]
        Q["User Query: 'What is the ChromaDB cluster failover command?'"]
        RoleCtx["User Clearance: [CUSTOMER, tenant: customer_001]"]
    end

    subgraph Stage1["Stage 1: Pre-Filtered Dense Vector Search"]
        FilterCalc["Compute ChromaDB Where Filter:\n`$and: [{$or: [tenant: customer_001, tenant: all]}, {classification: {$ne: 'internal'}}]`"]
        ChromaQuery["ChromaDB Vector Query\n(k = 10 candidates)"]
    end

    subgraph Stage2["Stage 2: Cross-Encoder Reranking"]
        CrossEncoder["BAAI/bge-reranker-base\nJoint Query-Doc Relevance Scoring"]
        TopKSelect["Top-4 High Relevance Chunks (Relevance > 0.0)"]
    end

    subgraph Stage3["Stage 3: Grounded Synthesis"]
        LLM["Groq LLaMA 3.3 70B Synthesis\n(Strict Grounded Gate)"]
        Output["Zero-Exposure Negative Response or Grounded Answer + Page Citations"]
    end

    Q --> FilterCalc
    RoleCtx --> FilterCalc
    FilterCalc --> ChromaQuery
    ChromaQuery --> CrossEncoder
    CrossEncoder --> TopKSelect
    TopKSelect --> LLM
    LLM --> Output
```

---

## 6. Security, Authentication & Role-Based Access Control (RBAC)

### 6.1 Role Clearance & Permissions Matrix

| Feature / Resource | `CUSTOMER` | `FDE_ENGINEER` | `ADMIN` |
| :--- | :---: | :---: | :---: |
| **Public Medical Guidelines** | ✅ Read | ✅ Read | ✅ Read |
| **Assigned Tenant Documents (`tenant_id`)** | ✅ Read | ✅ Read | ✅ Read |
| **Other Tenant Documents** | ❌ Blocked (Zero Leak) | ❌ Blocked (Zero Leak) | ✅ Read |
| **Internal FDE Operations & Runbooks** | ❌ Blocked (Zero Leak) | ✅ Read | ✅ Read |
| **Own Conversation History** | ✅ Read / Write | ✅ Read / Write | ✅ Read / Write |
| **Other Users' Conversations** | ❌ Blocked | ❌ Blocked | ❌ Blocked |
| **User Management (Create/Delete/List)** | ❌ 403 Forbidden | ❌ 403 Forbidden | ✅ Full CRUD |
| **Security Audit Logs Access** | ❌ 403 Forbidden | ❌ 403 Forbidden | ✅ Read Only |

### 6.2 Retrieval-Level Vector Pre-Filtering Specification
Authorization is strictly enforced **at the database query layer**, preventing unauthorized data from ever entering the LLM prompt context:

1. **Customer Role (`customer_001`)**:
   ```python
   where = {
       "$and": [
           {"$or": [{"tenant_id": "customer_001"}, {"tenant_id": "all"}]},
           {"classification": {"$ne": "internal"}}
       ]
   }
   ```
2. **FDE Engineer Role (`customer_001`)**:
   ```python
   where = {
       "$or": [
           {"tenant_id": "customer_001"},
           {"tenant_id": "all"},
           {"classification": "internal"}
       ]
   }
   ```
3. **Admin Role**:
   ```python
   where = None  # Full platform clearance across all indexed documents
   ```

### 6.3 Zero-Exposure Negative Response Policy
If a query yields 0 authorized chunks after RBAC filtering, the application generates a standardized negative response without disclosing the existence of restricted documents:
> *"I couldn't find relevant information in your authorized knowledge base. Please refer to official clinical guidelines or consult a healthcare professional."*

---

## 7. API Architecture & Endpoint Contracts

```mermaid
graph TD
    API["FastAPI Router Gateway"]
    
    API --> AuthRoutes["/api/auth/*"]
    AuthRoutes --> POST_Login["POST /api/auth/login"]
    AuthRoutes --> GET_Me["GET /api/auth/me"]
    AuthRoutes --> POST_Logout["POST /api/auth/logout"]

    API --> ChatRoutes["/api/chat/*"]
    ChatRoutes --> POST_Chat["POST /api/chat"]

    API --> ConvRoutes["/api/conversations/*"]
    ConvRoutes --> GET_Convs["GET /api/conversations"]
    ConvRoutes --> POST_Convs["POST /api/conversations"]
    ConvRoutes --> GET_ConvID["GET /api/conversations/{id}"]
    ConvRoutes --> DEL_ConvID["DELETE /api/conversations/{id}"]

    API --> AdminRoutes["/api/* (Admin Only)"]
    AdminRoutes --> GET_Users["GET /api/users"]
    AdminRoutes --> POST_Users["POST /api/users"]
    AdminRoutes --> DEL_Users["DELETE /api/users/{id}"]
    AdminRoutes --> GET_Logs["GET /api/audit-logs"]

    API --> SystemRoutes["/api/health"]
```

---

## 8. Observability & Performance Telemetry

```mermaid
gantt
    title End-to-End Query Execution Waterfall (~1420ms)
    dateFormat X
    axisFormat %s ms

    section Gateway & Auth
    JWT & RBAC Validation      :0, 15
    section Vector Retrieval
    BGE Small Embedding        :15, 65
    ChromaDB Pre-Filtered Search :65, 127
    section Cross-Reranking
    BGE Reranker Joint Scoring :127, 411
    section LLM Generation
    Groq LLaMA 3.3 Synthesis   :411, 1420
    section Persistence
    SQLite Message Logging     :1420, 1435
```

### Telemetry KPI Metrics:
* **Retrieval Latency ($p95$)**: `< 150ms`
* **Cross-Encoder Rerank Latency ($p95$)**: `< 350ms`
* **LLM Time-to-First-Token (TTFT)**: `< 300ms`
* **Total SLA Latency ($p95$)**: `< 1800ms`
* **Grounded Precision Score**: `> 92%`

---

## 9. Production Deployment Topology

```mermaid
flowchart TD
    subgraph Edge["Edge Infrastructure"]
        DNS["DNS / Cloudflare CDN"]
        SSL["TLS 1.3 Termination"]
    end

    subgraph Compute["Application Cluster (Gunicorn / Uvicorn)"]
        LB["Nginx Load Balancer"]
        Worker1["FastAPI Worker 1\n(Stateless API)"]
        Worker2["FastAPI Worker 2\n(Stateless API)"]
        WorkerN["FastAPI Worker N\n(Auto-Scaled)"]
    end

    subgraph StateAndCache["State & Cache Layer"]
        Redis[("Redis Cluster\nSession & Rate Limiting")]
        SQLiteDB[("Primary Database (PostgreSQL / SQLite)\nRead-Replica Topology")]
    end

    subgraph VectorCluster["Vector Storage & AI Infrastructure"]
        ChromaCluster[("ChromaDB / Qdrant\nPersistent Vector Store")]
        GroqCluster["Groq Cloud LPU\nHigh-Concurrency Inference"]
    end

    DNS --> SSL
    SSL --> LB
    LB --> Worker1
    LB --> Worker2
    LB --> WorkerN

    Worker1 --> Redis
    Worker1 --> SQLiteDB
    Worker1 --> ChromaCluster
    Worker1 --> GroqCluster

    Worker2 --> Redis
    Worker2 --> SQLiteDB
    Worker2 --> ChromaCluster
    Worker2 --> GroqCluster
```

---

## 10. Disaster Recovery & Security Runbook

1. **Failover Procedure for ChromaDB**:
   * Run orchestrator command: `make chroma-failover-secondary`
   * Validates chunk hash checksums across replica nodes.
2. **Embedding Drift Detection**:
   * Scheduled cron job computes cosine drift on reference queries.
   * If drift $> 0.08$ variance, triggers automatic pipeline re-indexing.
3. **Secret Rotation Policy**:
   * `JWT_SECRET_KEY` rotated quarterly with a 24-hour dual-token grace period.
   * Passwords protected via one-way bcrypt salting with cost factor 12.
