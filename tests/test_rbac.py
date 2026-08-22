import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.auth.security import hash_password, verify_password, create_access_token, decode_access_token
from backend.database.database import init_db, SessionLocal, UserRepository, AuditLogRepository
from backend.rag.retriever import MedicalRetriever

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_database():
    """Ensures database and seed users are initialized before tests."""
    init_db()


def test_password_hashing():
    pwd = "SecurePassword@2026"
    hashed = hash_password(pwd)
    assert hashed != pwd
    assert verify_password(pwd, hashed) is True
    assert verify_password("WrongPassword", hashed) is False


def test_jwt_token_creation_and_decoding():
    token = create_access_token(
        user_id="usr_test_123",
        username="testuser",
        role="CUSTOMER",
        tenant_id="customer_001",
        full_name="Test User",
    )
    assert isinstance(token, str) and len(token) > 20
    payload = decode_access_token(token)
    assert payload is not None
    assert payload["sub"] == "usr_test_123"
    assert payload["role"] == "CUSTOMER"
    assert payload["tenant_id"] == "customer_001"


def test_auth_login_success_and_failure():
    # 1. Success login
    res = client.post("/api/auth/login", json={"username": "admin", "password": "Admin@12345"})
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["user"]["role"] == "ADMIN"
    assert data["user"]["username"] == "admin"

    # 2. Invalid password
    res_bad = client.post("/api/auth/login", json={"username": "admin", "password": "WrongPassword"})
    assert res_bad.status_code == 401


def test_rbac_endpoint_permissions():
    # Login as Customer
    cust_res = client.post("/api/auth/login", json={"username": "customer1", "password": "Customer@12345"})
    cust_token = cust_res.json()["access_token"]
    cust_headers = {"Authorization": f"Bearer {cust_token}"}

    # Customer attempts to list users -> 403 Forbidden
    res_forbidden = client.get("/api/users", headers=cust_headers)
    assert res_forbidden.status_code == 403

    # Customer attempts to view audit logs -> 403 Forbidden
    res_audit_forbidden = client.get("/api/audit-logs", headers=cust_headers)
    assert res_audit_forbidden.status_code == 403

    # Login as Admin
    admin_res = client.post("/api/auth/login", json={"username": "admin", "password": "Admin@12345"})
    admin_token = admin_res.json()["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # Admin lists users -> 200 OK
    res_admin = client.get("/api/users", headers=admin_headers)
    assert res_admin.status_code == 200
    users = res_admin.json()
    assert len(users) >= 4

    # Admin views audit logs -> 200 OK
    res_logs = client.get("/api/audit-logs", headers=admin_headers)
    assert res_logs.status_code == 200
    logs = res_logs.json()
    assert len(logs) > 0


def test_retrieval_level_rbac_filtering():
    retriever = MedicalRetriever()

    # 1. Customer 1 query for internal runbook -> should NOT retrieve internal SOPs
    cust1_filter = retriever.build_rbac_where_clause(role="CUSTOMER", tenant_id="customer_001")
    assert cust1_filter == {
        "$and": [
            {"$or": [{"tenant_id": "customer_001"}, {"tenant_id": "all"}]},
            {"classification": {"$ne": "internal"}}
        ]
    }

    # 2. FDE Engineer filter -> includes internal SOPs and assigned tenant
    fde_filter = retriever.build_rbac_where_clause(role="FDE_ENGINEER", tenant_id="customer_001")
    assert fde_filter == {
        "$or": [
            {"tenant_id": "customer_001"},
            {"tenant_id": "all"},
            {"classification": "internal"}
        ]
    }

    # 3. Admin filter -> None (Access All)
    admin_filter = retriever.build_rbac_where_clause(role="ADMIN", tenant_id="system")
    assert admin_filter is None


def test_unauthenticated_chat_rejected():
    # Attempting to query chat without token -> 401 Unauthorized
    res = client.post("/api/chat", json={"message": "What is diabetes?"})
    assert res.status_code == 401
