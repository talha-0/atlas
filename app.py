import os
import json
from datetime import datetime
import gradio as gr
from uuid import uuid4
from openai import OpenAI
from pathlib import Path
from huggingface_hub import CommitScheduler

# --- API and Logging Setup ---
# The environment variables OPENAI_API_KEY and HF_TOKEN must be set in your Space secrets.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

DATA_DIR = Path("json_dataset")
DATA_DIR.mkdir(parents=True, exist_ok=True)
JSON_PATH = DATA_DIR / f"messages-{uuid4()}.jsonl"

# The Vault: Syncs local JSONL to the Hugging Face dataset
scheduler = CommitScheduler(
    repo_id="talhahkk/travel-void-data", 
    repo_type="dataset",
    folder_path=DATA_DIR,
    path_in_repo="data",
    token=os.getenv("HF_TOKEN")
)

def log_interaction(username, role, content):
    """Logs the interaction with the user's identity attached."""
    with scheduler.lock:
        with JSON_PATH.open("a") as f:
            f.write(json.dumps({
                "user": username,
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow().isoformat()
            }) + "\n")

# --- Agent Logic ---
def verify_travel_topic(user_input):
    """The Bouncer: Evaluates if the input is exclusively about travel."""
    system_prompt = """
    You are a strict binary classification system. 
    Analyze the user input. If it is about a travel experience, tourism, or visiting a place, respond ONLY with 'PASS'.
    If it is about any other topic, respond ONLY with 'FAIL'.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.0,
        max_tokens=5
    )
    return response.choices[0].message.content.strip() == "PASS"

def generate_facilitator_response(user_input, persona):
    """The Mirror: Reflects validation without adding new information."""
    if persona == "Empathetic":
        tone_instructions = "You are extremely warm, empathetic, and excited for the user."
    else:
        tone_instructions = "You are completely robotic, neutral, and monotone."

    system_prompt = f"""
    {tone_instructions}
    You are a listener. The user is sharing a travel experience.
    YOUR STRICT RULES:
    1. You MUST NOT provide any information, facts, or tips.
    2. You MUST NOT change the topic.
    3. You MUST ONLY endorse the user's message and encourage them to say more.
    4. Keep it to one short sentence. 
    Examples: "Tell me more.", "What did you see?", "That sounds interesting, please continue."
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.7,
        max_tokens=20
    )
    return response.choices[0].message.content.strip()

# --- Gradio UI & Routing ---
def chat_step(username, user_message, persona, chat_history):
    # 1. Enforce Identity
    if not username.strip():
        raise gr.Error("Identify yourself. The void requires a name.")

    # 2. Handle Empty Messages (Safely rebuild UI state)
    if not user_message.strip():
        ui_history = []
        for i in range(0, len(chat_history), 2):
            u_msg = chat_history[i]["content"]
            a_msg = chat_history[i+1]["content"] if i + 1 < len(chat_history) else None
            ui_history.append([u_msg, a_msg])
        return ui_history, chat_history, ""
    
    # 3. Process Valid Input
    history = chat_history or []
    history.append({"role": "user", "content": user_message})
    log_interaction(username, "user", user_message)

    is_travel = verify_travel_topic(user_message)

    # 4. Route based on Bouncer verification
    if not is_travel:
        rejection_msg = "I am only authorized to listen to travel experiences. Please return to the topic of travel."
        history.append({"role": "assistant", "content": rejection_msg})
        log_interaction(username, "assistant", rejection_msg)
    else:
        response = generate_facilitator_response(user_message, persona)
        history.append({"role": "assistant", "content": response})
        log_interaction(username, "assistant", response)

    # 5. Reconstruct the visual UI tuple format from the pristine internal state
    ui_history = []
    for i in range(0, len(history), 2):
        u_msg = history[i]["content"]
        a_msg = history[i+1]["content"] if i + 1 < len(history) else None
        ui_history.append([u_msg, a_msg])

    return ui_history, history, ""

# --- Build the Interface ---
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# The Listening Terminal")
    gr.Markdown("Share your travel experiences. We only listen.")
    
    with gr.Row():
        name_input = gr.Textbox(label="Traveler Identification", placeholder="Who are you?", scale=1)
        persona_selector = gr.Radio(["Empathetic", "Robotic"], label="Select Listener Persona", value="Empathetic", scale=2)
    
    # Standard Chatbot initialization (no 'type' parameter to avoid v6.0 conflicts)
    chatbot = gr.Chatbot(height=500)
    
    with gr.Row():
        msg = gr.Textbox(placeholder="I visited Hong Kong last week...", show_label=False, scale=8)
        send = gr.Button("Submit", scale=1)
        
    clear = gr.Button("Wipe Memory")
    
    # Internal state holds the strict dicts, Chatbot holds the visual tuples
    state_history = gr.State([])

    send.click(chat_step, inputs=[name_input, msg, persona_selector, state_history], outputs=[chatbot, state_history, msg])
    msg.submit(chat_step, inputs=[name_input, msg, persona_selector, state_history], outputs=[chatbot, state_history, msg])
    
    clear.click(lambda: ([], []), None, [chatbot, state_history])

if __name__ == "__main__":
    demo.launch()