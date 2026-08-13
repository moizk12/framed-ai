from framed import create_app


def test_health_endpoint():
    client = create_app({"TESTING": True}).test_client()
    r = client.get("/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data.get("status") == "healthy"


def test_readiness_and_version_are_safe():
    client = create_app(
        {
            "TESTING": True,
            "FRAMED_VERSION": "beta.1",
            "FRAMED_BUILD_SHA": "abcdef1234567",
        }
    ).test_client()
    assert client.get("/ready").get_json()["status"] == "ready"
    response = client.get("/version")
    assert response.status_code == 200
    assert response.get_json() == {
        "service": "framed-public-beta",
        "version": "beta.1",
        "build_sha": "abcdef1234567",
        "api_contract": "v1",
    }
