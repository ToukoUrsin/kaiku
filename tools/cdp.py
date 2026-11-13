"""Tiny CDP helper for the personal Chrome (127.0.0.1:9321).
usage: cdp.py open URL | cdp.py eval TARGET_ID 'js' | cdp.py list | cdp.py close TARGET_ID
"""
import asyncio, json, sys, urllib.request, urllib.parse
import websockets
BASE = "http://127.0.0.1:9321"
def http(path, method="GET"):
    req = urllib.request.Request(BASE + path, method=method)
    return json.loads(urllib.request.urlopen(req, timeout=10).read() or b"{}")
async def ev(ws_url, expr):
    async with websockets.connect(ws_url, max_size=50_000_000) as ws:
        await ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {"expression": expr, "awaitPromise": True, "returnByValue": True}}))
        while True:
            m = json.loads(await ws.recv())
            if m.get("id") == 1:
                return m["result"].get("result", {}).get("value", m["result"])
cmd = sys.argv[1]
if cmd == "list":
    for t in http("/json/list"):
        if t["type"] == "page": print(t["id"], t["url"][:120], "|", t["title"][:60])
elif cmd == "open":
    t = http("/json/new?" + urllib.parse.quote(sys.argv[2], safe=":/?=&%#"), "PUT")
    print(t["id"])
elif cmd == "eval":
    t = [x for x in http("/json/list") if x["id"] == sys.argv[2]][0]
    out = asyncio.run(ev(t["webSocketDebuggerUrl"], sys.argv[3]))
    print(out if isinstance(out, str) else json.dumps(out, indent=1))
elif cmd == "close":
    urllib.request.urlopen(BASE + "/json/close/" + sys.argv[2], timeout=10).read(); print("closed")
