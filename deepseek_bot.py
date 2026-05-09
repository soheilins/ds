import os
import time
import requests
import subprocess
import json

# -------------------- Configuration --------------------
TOKEN = os.environ.get("RUBIKA_TOKEN", "").strip()
if not TOKEN:
    print("FATAL: RUBIKA_TOKEN is empty.", flush=True)
    exit(1)

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
if not DEEPSEEK_API_KEY:
    print("FATAL: DEEPSEEK_API_KEY is empty.", flush=True)
    exit(1)

POLL_INTERVAL = 3                # seconds between getUpdates
RUN_DURATION = 5 * 3600 + 55 * 60  # 5h 55m
COMMIT_INTERVAL = 20 * 60        # push offset every 20 min
OFFSET_FILE = "offset.txt"

# Optional HTTP proxy (for GitHub runners that can't reach Rubika)
PROXY_URL = os.environ.get("PROXY_URL")
proxies = {"http": PROXY_URL, "https": PROXY_URL} if PROXY_URL else None

# Maximum conversation history (last N user+assistant pairs)
MAX_HISTORY = 20

BASE_RUBIKA = f"https://botapi.rubika.ir/v3/{TOKEN}"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# --------------------------------------------------------

# ---------- Git helpers (for periodic offset push) ----------
def setup_git():
    subprocess.run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"], check=False)
    subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=False)

def git_commit_and_push():
    try:
        subprocess.run(["git", "add", OFFSET_FILE], check=True)
        r = subprocess.run(["git", "diff", "--staged", "--quiet"], capture_output=True)
        if r.returncode == 0:
            return   # no changes
        subprocess.run(["git", "commit", "-m", "Update offset"], check=True)
        subprocess.run(["git", "push"], check=True)
        print("[GIT] Offset committed and pushed.", flush=True)
    except Exception as e:
        print(f"[GIT] Push failed: {e}", flush=True)

# ---------- Rubika API ----------
def api_call(method, payload=None):
    url = f"{BASE_RUBIKA}/{method}"
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, proxies=proxies, timeout=10)
            if "application/json" in resp.headers.get("content-type", ""):
                return resp.status_code, resp.json()
            return resp.status_code, resp.text
        except Exception as e:
            if attempt == 2:
                print(f"[Rubika] {method} failed: {e}", flush=True)
                return None, str(e)
            time.sleep(2)
    return None, "unknown"

def send_rubika_message(chat_id, text):
    # Rubika has a message length limit (~4096 chars). Split if needed.
    CHUNK_SIZE = 4000
    if len(text) <= CHUNK_SIZE:
        code, _ = api_call("sendMessage", {"chat_id": chat_id, "text": text})
        if code == 200:
            print(f"[OK] Reply sent → {chat_id}", flush=True)
        else:
            print(f"[FAIL] sendMessage {code}", flush=True)
    else:
        # Split into chunks (simple, could break words, but acceptable)
        chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
        for idx, chunk in enumerate(chunks):
            code, _ = api_call("sendMessage", {"chat_id": chat_id, "text": chunk})
            if code != 200:
                print(f"[FAIL] chunk {idx} send failed", flush=True)
                break
            time.sleep(0.5)  # slight delay to avoid flooding

# ---------- DeepSeek API ----------
def query_deepseek(messages):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.7,
        "stream": False
    }
    for attempt in range(2):
        try:
            resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            else:
                print(f"[DeepSeek] Error {resp.status_code}: {resp.text}", flush=True)
                return f"❌ DeepSeek API error ({resp.status_code})."
        except Exception as e:
            print(f"[DeepSeek] Exception: {e}", flush=True)
            if attempt == 1:
                return "⚠️ DeepSeek is not responding. Try again."
            time.sleep(3)
    return "⚠️ Unexpected error."

# ---------- Conversation storage ----------
conversations = {}   # chat_id -> list of {"role": ..., "content": ...}

def get_history(chat_id):
    if chat_id not in conversations:
        # Start with a system message only when first seen
        conversations[chat_id] = [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."}
        ]
    return conversations[chat_id]

def trim_history(history):
    """Keep only the last MAX_HISTORY messages (excluding system)."""
    system = [m for m in history if m["role"] == "system"]
    rest = [m for m in history if m["role"] != "system"]
    if len(rest) > MAX_HISTORY:
        rest = rest[-MAX_HISTORY:]
    return system + rest

# ---------- Main loop ----------
def main():
    start_time = time.time()
    last_commit = start_time

    setup_git()
    print("DeepSeek Rubika bot starting.", flush=True)

    # Test token
    code, info = api_call("getMe")
    if code == 200 and isinstance(info, dict):
        bot = info.get("data", {}).get("bot", {})
        print(f"Bot alive: {bot.get('bot_title', '?')} (@{bot.get('username', '?')})", flush=True)
    else:
        print("Warning: getMe failed, continuing.", flush=True)

    # Load latest offset
    next_offset_str = None
    if os.path.exists(OFFSET_FILE):
        with open(OFFSET_FILE) as f:
            next_offset_str = f.read().strip() or None
    if not next_offset_str:
        code, data = api_call("getUpdates", {"limit": 1})
        if code == 200 and isinstance(data, dict):
            next_offset_str = data.get("data", {}).get("next_offset_id")

    print(f"Offset: {next_offset_str}", flush=True)

    # Main polling loop
    try:
        while time.time() - start_time < RUN_DURATION:
            payload = {"limit": 10}
            if next_offset_str:
                payload["offset_id"] = next_offset_str

            code, data = api_call("getUpdates", payload)
            if code == 200 and isinstance(data, dict):
                inner = data.get("data", {})
                updates = inner.get("updates", [])
                new_offset_str = inner.get("next_offset_id")

                if updates:
                    print(f"Received {len(updates)} update(s).", flush=True)
                    for upd in updates:
                        if upd.get("type") != "NewMessage":
                            continue
                        msg = upd.get("new_message", {})
                        if msg.get("sender_type") != "User":
                            continue
                        text = msg.get("text", "").strip()
                        chat_id = upd.get("chat_id")
                        if not text or not chat_id:
                            continue

                        # --- Special commands ---
                        if text == "/reset":
                            conversations.pop(chat_id, None)
                            send_rubika_message(chat_id, "🧹 Conversation history cleared.")
                            continue

                        # --- Normal chat ---
                        # Get conversation history
                        history = get_history(chat_id)
                        # Append user message
                        history.append({"role": "user", "content": text})
                        # Trim to manageable length
                        history = trim_history(history)
                        conversations[chat_id] = history   # update ref

                        # Send to DeepSeek
                        reply = query_deepseek(history)

                        # Append assistant response
                        history.append({"role": "assistant", "content": reply})
                        conversations[chat_id] = history

                        # Reply to user
                        send_rubika_message(chat_id, reply)

                # Save offset immediately after poll
                if new_offset_str:
                    next_offset_str = new_offset_str
                    with open(OFFSET_FILE, "w") as f:
                        f.write(next_offset_str)

                # Periodic git push
                now = time.time()
                if now - last_commit >= COMMIT_INTERVAL:
                    git_commit_and_push()
                    last_commit = now

            else:
                print(f"[Poll] getUpdates error: {code} {data}", flush=True)
                time.sleep(5)
                continue

            elapsed = time.time() - start_time
            sleep_time = max(0, min(POLL_INTERVAL, RUN_DURATION - elapsed))
            time.sleep(sleep_time)
    finally:
        with open(OFFSET_FILE, "w") as f:
            f.write(next_offset_str or "")
        git_commit_and_push()
        print("Bot shutting down.", flush=True)

if __name__ == "__main__":
    main()
