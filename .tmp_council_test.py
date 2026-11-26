import json
import urllib.request

BASE = "http://127.0.0.1:8001"
PROMPT = "Lets say we are predicting the exam score of a student based on the number of hours he studied."

req = urllib.request.Request(
    f"{BASE}/api/conversations",
    data=b"{}",
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as r:
    conv = json.loads(r.read().decode())

cid = conv["id"]
payload = json.dumps({"content": PROMPT}).encode()
req2 = urllib.request.Request(
    f"{BASE}/api/conversations/{cid}/message",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req2, timeout=240) as r:
    out = json.loads(r.read().decode())

result = {
    "conversation_id": cid,
    "stage1": out.get("stage1", []),
    "stage2": out.get("stage2", []),
    "stage3": out.get("stage3", {}),
    "metadata": out.get("metadata", {}),
}

print(json.dumps(result))
