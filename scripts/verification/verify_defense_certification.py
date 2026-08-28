import requests
import json
import time

API_URL = "http://localhost:8000"

def wait_for_api():
    print("Waiting for API to be ready...")
    for _ in range(10):
        try:
            requests.get(f"{API_URL}/docs")
            print("API is ready.")
            return True
        except requests.ConnectionError:
            time.sleep(2)
    print("API failed to start.")
    return False

def verify():
    if not wait_for_api():
        return
        
    print("Testing /api/v1/defense/certify...")
    
    payload = {
        "attack_family": "micro_structuring",
        "seed": 42,
        "rounds": 3,
        "generations_per_round": 2,
        "population_size": 3,
        "attack_scale": 100
    }
    
    start_time = time.time()
    response = requests.post(f"{API_URL}/api/v1/defense/certify", json=payload)
    end_time = time.time()
    
    print(f"Request took {end_time - start_time:.2f} seconds")
    
    if response.status_code == 200:
        data = response.json()
        print("Success! Certification Result:")
        print(json.dumps(data, indent=2))
        
        rounds = data.get("rounds", [])
        if len(rounds) > 0:
            print(f"Completed {len(rounds)} rounds.")
            for r in rounds:
                print(f"Round {r['round_number']}: Evasion {r['evasion_rate']:.3f}, Failure: {r['failure_cause']}")
        else:
            print("No rounds returned.")
    else:
        print(f"Failed with status code {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    verify()
