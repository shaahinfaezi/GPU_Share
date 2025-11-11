import pytest
import requests

def test_smoke():
    url = "http://localhost:6000/infer"
    response = requests.post(url, json={"prompt": "Test prompt"})
    
    assert response.status_code == 200
    assert "output" in response.json()