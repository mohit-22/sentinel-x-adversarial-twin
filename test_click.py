import sys
import json
import urllib.request
import time

def main():
    resp = urllib.request.urlopen('http://localhost:9222/json')
    tabs = json.loads(resp.read().decode('utf-8'))
    ws_url = next(t['webSocketDebuggerUrl'] for t in tabs if t['type'] == 'page')

    from websocket import create_connection
    ws = create_connection(ws_url)

    def evaluate(code):
        ws.send(json.dumps({
            "id": 1,
            "method": "Runtime.evaluate",
            "params": {"expression": code, "returnByValue": True}
        }))
        res = json.loads(ws.recv())
        return res.get("result", {}).get("result", {}).get("value")

    print(evaluate("Array.from(document.querySelectorAll('button')).find(b => b.textContent && b.textContent.includes('Run Recursive Certification'))?.outerHTML"))
    print(evaluate("typeof window.__NEXT_DATA__"))

if __name__ == '__main__':
    main()
