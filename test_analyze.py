from app.judge.scenario_runner import sample_clean_subset, inject_attacks
from app.api.endpoints import _APP_STATE
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

print("keys in base df:", _APP_STATE["customers"].columns.tolist() if "customers" in _APP_STATE else "No customers")
