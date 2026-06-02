from framed import create_app


def test_health_endpoint():
    client = create_app().test_client()
    r = client.get("/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data.get("status") == "healthy"
