import subprocess
import time
import json
import base64
from pathlib import Path
import urllib.request

try:
    import websocket
except ImportError:
    websocket = None

def main():
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    user_data = r"C:\Users\parid\AppData\Local\Temp\chrome-cdp"
    target_url = "http://127.0.0.1:8765"
    out_path = Path(r"C:\Users\parid\Downloads\Agentic AI\adaptive-agentic-semantic-cache\docs\images\workbench_screenshot.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        chrome_path,
        "--headless=new",
        "--disable-gpu",
        "--remote-debugging-port=9222",
        "--remote-allow-origins=*",
        f"--user-data-dir={user_data}",
        "--window-size=1366,1150",
        target_url
    ]
    proc = subprocess.Popen(cmd)
    try:
        # Wait for debugging port
        ws_url = None
        for _ in range(20):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen("http://127.0.0.1:9222/json") as resp:
                    pages = json.loads(resp.read().decode())
                    for p in pages:
                        if p.get("type") == "page" and "webSocketDebuggerUrl" in p:
                            ws_url = p["webSocketDebuggerUrl"]
                            break
                    if ws_url:
                        break
            except Exception:
                pass

        if not ws_url:
            print("Failed to get WebSocketDebuggerUrl from Chrome")
            return False

        print(f"Connecting to CDP: {ws_url}")
        ws = websocket.create_connection(ws_url, timeout=15)
        
        # Wait for page JS to load, then click the top history button and scroll to show the agent stages timeline
        script = """
        (async () => {
            await new Promise(r => setTimeout(r, 1500));
            const btn = document.querySelector('#history button');
            if (btn) btn.click();
            await new Promise(r => setTimeout(r, 1500));
            window.scrollBy(0, 320);
            await new Promise(r => setTimeout(r, 500));
        })()
        """
        ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {"expression": script, "awaitPromise": True}}))
        ws.recv()

        # Capture screenshot
        ws.send(json.dumps({"id": 2, "method": "Page.captureScreenshot", "params": {"format": "png"}}))
        msg = ws.recv()
        data = json.loads(msg)
        img_b64 = data["result"]["data"]
        out_path.write_bytes(base64.b64decode(img_b64))
        print(f"Screenshot successfully saved to {out_path} ({out_path.stat().st_size} bytes)")
        ws.close()
        return True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    main()
