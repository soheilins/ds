import os
import time
import requests
import subprocess
from llama_cpp import Llama

# -------------------- Configuration --------------------
TOKEN = os.environ.get("RUBIKA_TOKEN", "").strip()
if not TOKEN:
    print("FATAL: RUBIKA_TOKEN is empty.", flush=True)
    exit(1)

POLL_INTERVAL = 3                # seconds between getUpdates
RUN_DURATION = 5 * 3600 + 55 * 60  # 5h 55m
COMMIT_INTERVAL = 20 * 60        # push offset every 20 min
OFFSET_FILE = "offset.txt"
MODEL_FILE = "model.gguf"

# Optional proxy for Rubika (not for model, model is local)
PROXY_URL = os.environ.get("PROXY_URL")
proxies = {"http": PROXY_URL, "https": PROXY_URL} if PROXY_URL else None

# How many chat turns to remember (system + user + assistant)
MAX_HISTORY = 10

BASE_RUBIKA = f"https://botapi.rubika.ir/v3/{TOKEN}"

# -------------------- Git helpers --------------------
def setup_git():
    subprocess.run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"], check=False)
    subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=False)

def git_commit_and_push():
    try:
        subprocess.run(["git", "add", OFFSET_FILE], check=True)
        r = subprocess.run(["git", "diff", "--staged", "--quiet"], capture_output=True)
        if r.returncode == 0:
            return
        subprocess.run(["git", "commit", "-m", "Update offset"], check=True)
        subprocess.run(["git", "push"], check=True)
        print("[GIT] Offset committed and pushed.", flush=True)
    except Exception as e:
        print(f"[GIT] Push failed: {e}", flush=True)

# -------------------- Rubika API --------------------
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
    CHUNK_SIZE = 4000
    if len(text) <= CHUNK_SIZE:
        code, _ = api_call("sendMessage", {"chat_id": chat_id, "text": text})
        if code == 200:
            print(f"[OK] Reply sent → {chat_id}", flush=True)
        else:
            print(f"[FAIL] sendMessage {code}", flush=True)
    else:
        for i in range(0, len(text), CHUNK_SIZE):
            chunk = text[i:i+CHUNK_SIZE]
            code, _ = api_call("sendMessage", {"chat_id": chat_id, "text": chunk})
            if code != 200:
                print(f"[FAIL] chunk send failed", flush=True)
                break
            time.sleep(0.5)

# -------------------- Conversation store --------------------
conversations = {}   # chat_id -> list of {"role": ..., "content": ...}

def get_history(chat_id):
    if chat_id not in conversations:
        conversations[chat_id] = [
            {"role": "system", "content": "You are a helpful, polite assistant. Answer concisely."}
        ]
    return conversations[chat_id]

def trim_history(history):
    system = [m for m in history if m["role"] == "system"]
    rest = [m for m in history if m["role"] != "system"]
    if len(rest) > MAX_HISTORY * 2:   # paired user+assistant
        rest = rest[-(MAX_HISTORY * 2):]
    return system + rest

# -------------------- LLM --------------------
def load_model():
    if not os.path.exists(MODEL_FILE):
        print(f"ERROR: Model file '{MODEL_FILE}' not found.", flush=True)
        exit(1)
    print("Loading model (this may take a moment)...", flush=True)
    return Llama(
        model_path=MODEL_FILE,
        n_ctx=2048,        # context window
        n_threads=2,       # GitHub has 2 cores
        verbose=False
    )

model = load_model()

def generate_reply(messages):
    # Format messages into TinyLlama chat format (it uses <|user|> and <|assistant|>)
    # The template from TheBloke: " <|user|>\n{user_msg} </s> <|assistant|>\n{assistant_msg}"
    prompt_parts = []
    for m in messages:
        if m["role"] == "system":
            # TinyLlama doesn't have system, prepend as user note
            prompt_parts.append(f"<|system|>\n{m['content']}</s>")
        elif m["role"] == "user":
            prompt_parts.append(f"<|user|>\n{m['content']}</s>")
        elif m["role"] == "assistant":
            prompt_parts.append(f"<|assistant|>\n{m['content']}</s>")
    prompt = "\n".join(prompt_parts) + "\n<|assistant|>\n"

    output = model(
        prompt,
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        stop=["</s>", "<|user|>"],
        echo=False
    )
    reply = output["choices"][0]["text"].strip()
    return reply

# -------------------- Main loop --------------------
def main():
    start_time = time.time()
    last_commit = start_time

    setup_git()
    print("Local chatbot starting.", flush=True)

    # Verify token
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

    print("Bot is ready.", flush=True)

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

                        # Reset command
                        if text == "/reset":
                            conversations.pop(chat_id, None)
                            send_rubika_message(chat_id, "🧹 Conversation cleared.")
                            continue

                        # Build history
                        history = get_history(chat_id)
                        history.append({"role": "user", "content": text})
                        history = trim_history(history)
                        conversations[chat_id] = history

                        # Generate reply
                        try:
                            reply = generate_reply(history)
                        except Exception as e:
                            print(f"Generation error: {e}", flush=True)
                            reply = "⚠️ Model error, please try again."

                        history.append({"role": "assistant", "content": reply})
                        conversations[chat_id] = history

                        send_rubika_message(chat_id, reply)

                if new_offset_str:
                    next_offset_str = new_offset_str
                    with open(OFFSET_FILE, "w") as f:
                        f.write(next_offset_str)

                # Periodic push
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
