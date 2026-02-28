from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app


def test_register_login_and_me_flow():
    with TestClient(app) as client:
        email = f"test-{uuid4().hex[:10]}@example.com"
        password = "strongpass123"

        register_response = client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": password},
        )
        assert register_response.status_code == 201
        register_payload = register_response.json()
        assert register_payload["token_type"] == "bearer"
        assert register_payload["user"]["email"] == email

        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": password},
        )
        assert login_response.status_code == 200
        login_payload = login_response.json()
        assert login_payload["access_token"]

        me_response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {login_payload['access_token']}"},
        )
        assert me_response.status_code == 200
        assert me_response.json()["email"] == email
