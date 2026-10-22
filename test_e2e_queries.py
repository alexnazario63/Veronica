import tempfile
from pathlib import Path
from starlette.testclient import TestClient
from services.api.app import create_app

with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
    app = create_app(data_dir=Path(tmp_dir))
    client = TestClient(app)

    # Cria conversa
    conv = client.post("/api/conversations", json={"title": "Teste Veronica Gemini Pro"}, headers={"x-user-id": "usr-01"}).json()
    cid = conv["conversation_id"]

    # Pergunta 1: OSPF Huawei
    r1 = client.post(
        f"/api/conversations/{cid}/messages",
        json={"content": "veronica como posso configuarar um ospf no huawei ?"},
        headers={"x-user-id": "usr-01"}
    )
    data1 = r1.json()
    print("=== RESPOSTA 1 (OSPF Huawei) ===")
    print("Status:", r1.status_code)
    print("Content preview (first 400 chars):")
    print(data1["content"][:400])
    assert "Exemplo Prático de Topologia" in data1["content"] or "Exemplo Pratico de Topologia" in data1["content"]
    assert "display ospf peer" in data1["content"]
    assert "Nao encontrei material" not in data1["content"]

    # Pergunta 2: Follow-up "busque sobre"
    r2 = client.post(
        f"/api/conversations/{cid}/messages",
        json={"content": "busque sobre"},
        headers={"x-user-id": "usr-01"}
    )
    data2 = r2.json()
    print("\n=== RESPOSTA 2 (Follow-up 'busque sobre') ===")
    print("Status:", r2.status_code)
    print("Content preview (first 400 chars):")
    print(data2["content"][:400])
    assert "Nao encontrei material" not in data2["content"]

    # Pergunta 3: "BGP protocolo"
    r3 = client.post(
        f"/api/conversations/{cid}/messages",
        json={"content": "BGP protocolo"},
        headers={"x-user-id": "usr-01"}
    )
    data3 = r3.json()
    print("\n=== RESPOSTA 3 ('BGP protocolo') ===")
    print("Status:", r3.status_code)
    print("Content preview (first 400 chars):")
    print(data3["content"][:400])
    assert "display bgp peer" in data3["content"]
    assert "Nao encontrei material" not in data3["content"]
    assert "['Retomando" not in data3["content"]

    print("\n>>> TODOS OS TESTES END-TO-END PASSARAM COM SUCESSO! <<<")