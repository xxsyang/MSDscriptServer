import requests



response = requests.post("http://localhost:8964/generate", json={
        "prompt": "5 * (",
        "max_length": 50
})