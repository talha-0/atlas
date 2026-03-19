import os
import json
from datetime import datetime
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

# The Vault
scheduler = CommitScheduler(
    repo_id="talhahkk/travel-void-data", 
    repo_type="dataset",
    folder_path=DATA_DIR,
    path_in_repo="data",
    token=os.getenv("HF_TOKEN")
)

def log_interaction(role, content):
    with scheduler.lock:
        with JSON_PATH.open("a") as f:
            f.write(json.dumps({
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow().isoformat()
            }) + "\n")

# --- Agent Logic ---
def verify_travel_topic(user_input):
    """The Bouncer"""
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
    """The Mirror"""
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
    Examples: "Tell me more.", "What did you see.", "That sounds interesting, please continue."
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
def chat_step(user_message, persona, chat_history):
    if not user_message.strip():
        # Rebuild UI state safely if empty input
        ui_history = []
        for i in range(0, len(chat_history), 2):
            u_msg = chat_history[i]["content"]
            a_msg = chat_history[i+1]["content"] if i + 1 < len(chat_history) else None
            ui_history.append([u_msg, a_msg])
        return ui_history, chat_history, ""
    
    history = chat_history or []
    history.append({"role": "user", "content": user_message})
    log_interaction("user", user_message)

    is_travel = verify_travel_topic(user_message)

    if not is_travel:
        rejection_msg = "I am only authorized to listen to travel experiences. Please return to the topic of travel."
        history.append({"role": "assistant", "content": rejection_msg})
        log_interaction("assistant", rejection_msg)
    else:
        response = generate_facilitator_response(user_message, persona)
        history.append({"role": "assistant", "content": response})
        log_interaction("assistant", response)

    # Reconstruct the UI tuple format from the pristine internal state
    ui_history = []
    for i in range(0, len(history), 2):
        u_msg = history[i]["content"]
        a_msg = history[i+1]["content"] if i + 1 < len(history) else None
        ui_history.append([u_msg, a_msg])

    return ui_history, history, ""

with gr.Blocks() as demo:
    gr.Markdown("# The Listening Terminal")
    gr.Markdown("Share your travel experiences. We only listen.")
    
    persona_selector = gr.Radio(["Empathetic", "Robotic"], label="Select Listener Persona", value="Empathetic")
    chatbot = gr.Chatbot(height=500)
    
    with gr.Row():
        msg = gr.Textbox(placeholder="I visited Hong Kong last week...", show_label=False, scale=8)
        send = gr.Button("Submit", scale=1)
        
    clear = gr.Button("Wipe Memory")
    
    # Internal state holds the dicts, Chatbot holds the visual tuples
    state_history = gr.State([])

    send.click(chat_step, inputs=[msg, persona_selector, state_history], outputs=[chatbot, state_history, msg])
    msg.submit(chat_step, inputs=[msg, persona_selector, state_history], outputs=[chatbot, state_history, msg])
    
    clear.click(lambda: ([], []), None, [chatbot, state_history])

if __name__ == "__main__":
    demo.launch()