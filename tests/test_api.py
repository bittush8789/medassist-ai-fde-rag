import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database.database import init_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


def get_auth_headers():
    res = client.post("/api/auth/login", json={"username": "admin", "password": "Admin@12345"})
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "vector_store" in data


def test_conversations_crud():
    headers = get_auth_headers()

    # 1. Create Conversation
    create_res = client.post("/api/conversations", json={"title": "Test Consultation"}, headers=headers)
    assert create_res.status_code == 201
    conv_data = create_res.json()
    conv_id = conv_data["id"]
    assert conv_data["title"] == "Test Consultation"

    # 2. List Conversations
    list_res = client.get("/api/conversations", headers=headers)
    assert list_res.status_code == 200
    conv_list = list_res.json()
    assert any(c["id"] == conv_id for c in conv_list)

    # 3. Get Conversation Detail
    get_res = client.get(f"/api/conversations/{conv_id}", headers=headers)
    assert get_res.status_code == 200
    detail = get_res.json()
    assert detail["id"] == conv_id

    # 4. Delete Conversation
    del_res = client.delete(f"/api/conversations/{conv_id}", headers=headers)
    assert del_res.status_code == 200

    # 5. Verify 404
    get_after_del = client.get(f"/api/conversations/{conv_id}", headers=headers)
    assert get_after_del.status_code == 404
