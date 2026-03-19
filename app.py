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

scheduler = CommitScheduler(
    repo_id="talhahkk/travel-void-data", 
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
        "timestamp": datetime.utcnow().isoformat()
    }
    if intent:
        payload["intent"] = intent

    with scheduler.lock:
        with JSON_PATH.open("a") as f:
            f.write(json.dumps(payload) + "\n")

def flag_last_interaction(username, chat_history):
    """The Surveillance Camera: Logs a manual flag to the dataset."""
    if not username.strip():
        raise gr.Error("Identity required to submit a flag.")
        
    if not chat_history or len(chat_history) < 2:
        gr.Info("There is nothing to flag yet.")
        return

    last_user_msg = chat_history[-2]["content"]
    last_bot_msg = chat_history[-1]["content"]
    
    with scheduler.lock:
        with JSON_PATH.open("a") as f:
            f.write(json.dumps({
                "user": username,
                "action": "FLAGGED_MISCLASSIFICATION",
                "flagged_user_input": last_user_msg,
                "flagged_bot_response": last_bot_msg,
                "timestamp": datetime.utcnow().isoformat()
            }) + "\n")
    
    gr.Info("Misclassification flagged and permanently logged to the vault.")

# --- Agent Logic ---
def verify_travel_topic(user_input):
    """The Bouncer: Now upgraded to handle basic pleasantries."""
    system_prompt = """
    You are a strict classification system. Analyze the user input.
    1. If it is a basic greeting, introduction, or pleasantry (e.g., "hi", "hello", "how are you"), respond ONLY with 'GREETING'.
    2. If it is about a travel experience, tourism, or visiting a place, respond ONLY with 'TRAVEL'.
    3. If it is about ANY other topic, respond ONLY with 'OTHER'.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.0,
        max_tokens=5
    )
    return response.choices[0].message.content.strip()

def generate_facilitator_response(user_input, persona):
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
        model="gpt-4o-mini",
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
    history = chat_history or []
    
    # Safely handle empty messages without crashing
    if not user_message.strip():
        return history, history, ""

    history.append({"role": "user", "content": user_message})

    # Graceful Identity Enforcement (In-Chat instead of UI Error)
    if not username.strip():
        sys_msg = "Error: Identity required. Please enter your Traveler Identification in the box above before speaking."
        history.append({"role": "assistant", "content": sys_msg})
        return history, history, ""

    log_interaction(username, "user", user_message)

    try:
        # Route through the upgraded Bouncer
        intent = verify_travel_topic(user_message)

        if intent == "GREETING":
            sys_msg = "Hello. I am here to listen. Tell me about your travels."
            history.append({"role": "assistant", "content": sys_msg})
            log_interaction(username, "assistant", sys_msg, intent="GREETING")
            
        elif intent == "OTHER":
            sys_msg = "I am only authorized to listen to travel experiences. Please return to the topic of travel."
            history.append({"role": "assistant", "content": sys_msg})
            log_interaction(username, "assistant", sys_msg, intent="OTHER")
            
        else: # TRAVEL
            response = generate_facilitator_response(user_message, persona)
            history.append({"role": "assistant", "content": response})
            log_interaction(username, "assistant", response, intent="TRAVEL")
            
    except Exception as e:
        # Graceful API Failure Handling
        err_msg = f"System fault: The connection to the cognitive engine failed. Please try again. ({str(e)})"
        history.append({"role": "assistant", "content": err_msg})

    return history, history, ""

# Moved the theme parameter to demo.launch() to fix the Gradio 6.0 warning
with gr.Blocks() as demo:
    gr.Markdown("# The Listening Terminal")
    gr.Markdown("Share your travel experiences. We only listen.")
    
    with gr.Row():
        name_input = gr.Textbox(label="Traveler Identification", placeholder="Who are you?", scale=1)
        persona_selector = gr.Radio(["Empathetic", "Robotic"], label="Select Listener Persona", value="Empathetic", scale=2)
    
    chatbot = gr.Chatbot(height=500)
    
    with gr.Row():
        msg = gr.Textbox(placeholder="I visited Hong Kong last week...", show_label=False, scale=8)
        send = gr.Button("Submit", scale=1)
        
    with gr.Row():
        flag_btn = gr.Button("⚠️ Report Misclassification", variant="secondary")
        clear = gr.Button("Wipe Memory")
    
    state_history = gr.State([])

    send.click(chat_step, inputs=[name_input, msg, persona_selector, state_history], outputs=[chatbot, state_history, msg])
    msg.submit(chat_step, inputs=[name_input, msg, persona_selector, state_history], outputs=[chatbot, state_history, msg])
    
    flag_btn.click(flag_last_interaction, inputs=[name_input, state_history], outputs=[])
    clear.click(lambda: ([], []), None, [chatbot, state_history])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Monochrome())