import os
import json
import re
from datetime import datetime, timezone
import gradio as gr
from uuid import uuid4
from openai import OpenAI
from pathlib import Path
from huggingface_hub import CommitScheduler

# --- API and Logging Setup ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

DATA_DIR = Path("json_dataset")
DATA_DIR.mkdir(parents=True, exist_ok=True)
JSON_PATH = DATA_DIR / f"messages-{uuid4()}.jsonl"

scheduler = CommitScheduler(
    repo_id="talhahkk/atlas-data",
    repo_type="dataset",
    folder_path=DATA_DIR,
    path_in_repo="data",
    token=os.getenv("HF_TOKEN")
)

def log_interaction(username, role, content, intent=None):
    payload = {
        "user": username,
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    if intent:
        payload["intent"] = intent

    with scheduler.lock:
        with JSON_PATH.open("a") as f:
            f.write(json.dumps(payload) + "\n")

# --- Deterministic Greeting Detection ---
def is_greeting(text):
    greetings = {
        "hi", "hello", "hey", "yo", "good morning",
        "good afternoon", "good evening", "greetings", "sup"
    }
    # Strip out all punctuation so "Hi!" or "Hello." matches cleanly
    text_clean = re.sub(r'[^\w\s]', '', text.lower()).strip()
    
    # ONLY return true if the entire message is just a greeting word
    return text_clean in greetings

# --- Intent Classification ---
def verify_travel_topic(user_input, chat_history):
    # HARD RULE: greetings handled without LLM
    if is_greeting(user_input):
        return "GREETING"

    context_str = "No prior context."
    if chat_history:
        recent_msgs = chat_history[-4:]
        context_str = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_msgs]
        )

    system_prompt = f"""
You are a strict classification system.

Your ONLY output must be EXACTLY ONE WORD from this list: GREETING, TRAVEL, OTHER.
Do not add punctuation. Do not add explanations. Do not use markdown.

Rules:
- If user shares travel OR answers a travel question → TRAVEL
- If unrelated → OTHER
- NEVER output anything else

Context:
{context_str}
"""

    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.0,
        max_completion_tokens=5
    )

    intent = response.choices[0].message.content.strip().upper()

    # Safety fallback
    if intent not in ["GREETING", "TRAVEL", "OTHER"]:
        intent = "OTHER"

    return intent

# --- Response Generator ---
def generate_facilitator_response(user_input, persona, username):
    if persona == "Empathetic":
        tone = "You are a warm, curious listener."
    else:
        tone = "You are a neutral, monotone listener."

    system_prompt = f"""
{tone}

You are Atlas. User: {username}

RULES:
- Ask ONLY travel-related questions
- ONE sentence only (max 15 words)
- NO filler
- NO advice
- NO repeating user input

Examples:
"What was your favorite place there?"
"What did you enjoy most during that trip?"
"""

    response = client.chat.completions.create(
        model="gpt-5.4-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.7,
        max_completion_tokens=40
    )

    return response.choices[0].message.content.strip()

# --- Chat Logic ---
def chat_step(user_message, username, persona, chat_history):
    history = list(chat_history) if chat_history else []
    msg = user_message.strip()

    if not msg:
        return history, history, ""

    if not username or not username.strip():
        history.append({"role": "user", "content": msg})
        history.append({
            "role": "assistant",
            "content": "Error: Please enter your name in configuration."
        })
        return history, history, ""

    clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', username).strip()[:20]

    intent = verify_travel_topic(msg, history)

    print(f"[DEBUG] Input: {msg} → Intent: {intent}")

    history.append({"role": "user", "content": msg})
    log_interaction(clean_name, "user", msg)

    try:
        if intent == "GREETING":
            if persona == "Empathetic":
                reply = f"Hello {clean_name}, I am Atlas, your dedicated travel listener. Where did your most recent journey take you?"
            else:
                reply = f"User {clean_name} recognized. I am Atlas. Awaiting input regarding your travel experiences."

        elif intent == "OTHER":
            if persona == "Empathetic":
                reply = f"I can only discuss travel experiences, {clean_name}."
            else:
                reply = f"Travel topics only, {clean_name}."

        elif intent == "TRAVEL":
            reply = generate_facilitator_response(msg, persona, clean_name)

        history.append({"role": "assistant", "content": reply})
        log_interaction(clean_name, "assistant", reply, intent=intent)

    except Exception as e:
        history.append({
            "role": "assistant",
            "content": f"System error: {str(e)}"
        })

    return history, history, ""

# --- UI ---
custom_theme = gr.themes.Soft(
    spacing_size="sm", 
    radius_size="md",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
)

with gr.Blocks(theme=custom_theme) as demo:
    gr.Markdown("<h1 style='text-align: center; font-weight: 300; margin-bottom: 0;'>Atlas</h1>")
    gr.Markdown("<p style='text-align: center; color: gray; margin-top: 0;'>I am here to listen. Share your travel experiences.</p>")

    with gr.Accordion("⚙️ Configuration", open=False):
        name_input = gr.Textbox(label="Identification", placeholder="Enter your name to begin...")
        persona_selector = gr.Radio(
            ["Empathetic", "Robotic"],
            value="Empathetic",
            label="Persona"
        )

    chatbot = gr.Chatbot(show_label=False)
    
    msg = gr.Textbox(placeholder="I visited Miami...", show_label=False)
    
    with gr.Row():
        send = gr.Button("Send", variant="primary")
        reset = gr.Button("Wipe Memory", variant="secondary")

    state = gr.State([])

    send.click(
        chat_step,
        inputs=[msg, name_input, persona_selector, state],
        outputs=[chatbot, state, msg]
    )

    msg.submit(
        chat_step,
        inputs=[msg, name_input, persona_selector, state],
        outputs=[chatbot, state, msg]
    )

    reset.click(lambda: ([], []), None, [chatbot, state])

if __name__ == "__main__":
    demo.launch()