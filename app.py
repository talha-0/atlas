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
def verify_travel_topic(user_input, chat_history):
    context_str = "No prior context."
    if chat_history:
        recent_msgs = chat_history[-4:]
        context_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_msgs])

    system_prompt = f"""
    You are a strict classification system with short-term memory. Analyze the user's latest input within the context of the conversation.
    
    Recent Conversation Context:
    {context_str}
    
    Classification Rules:
    1. If the latest input is a basic greeting, introduction, or pleasantry, respond ONLY with 'GREETING'.
    2. If the latest input is about a travel experience, tourism, OR is a logical, contextual continuation of a travel story they just mentioned (like describing an activity at a destination), respond ONLY with 'TRAVEL'.
    3. If the latest input is a hard pivot to an unrelated topic (e.g., math, general science, non-travel daily tasks), respond ONLY with 'OTHER'.
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

def generate_facilitator_response(user_input, persona, username):
    """The Mirror: Now aware of the user's name and its own identity (Atlas)."""
    if persona == "Empathetic":
        tone_instructions = "You are extremely warm, highly empathetic, and genuinely excited for the user. Speak like a close friend."
    else:
        tone_instructions = "You are completely robotic, neutral, and monotone. Speak like a data processor."

    system_prompt = f"""
    {tone_instructions}
    You are Atlas, a dedicated listener. The user's name is {username}. They are sharing a travel experience.
    
    YOUR STRICT RULES:
    1. You MUST NOT provide any outside information, facts, recommendations, or tips.
    2. You MUST NOT change the topic or talk about yourself, other than introducing yourself as Atlas if asked.
    3. ACTIVE LISTENING: You MUST explicitly mention a specific detail {username} just shared to prove you are listening, and then encourage them to continue. 
    4. Keep it concise (1 to 2 sentences max).
    5. Occasionally use the user's name to make it feel personal.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.7,
        max_tokens=40
    )
    return response.choices[0].message.content.strip()

# --- Gradio UI & Routing ---
def chat_step(user_message, username, persona, chat_history):
    history = chat_history or []
    msg = user_message.strip()
    
    if not msg:
        return history, history, ""

    if not username or not username.strip():
        history.append({"role": "user", "content": msg})
        sys_msg = "Error: Identity required. Please enter your Traveler Identification in the box above before speaking."
        history.append({"role": "assistant", "content": sys_msg})
        return history, history, ""

    # Clean the username for personalized messages
    clean_name = username.strip()

    intent = verify_travel_topic(msg, history)

    history.append({"role": "user", "content": msg})
    log_interaction(clean_name, "user", msg)

    try:
        if intent == "GREETING":
            if persona == "Empathetic":
                sys_msg = f"Hello {clean_name}, I am Atlas. It's wonderful to meet you. Please, tell me about your travels."
            else:
                sys_msg = f"User {clean_name} recognized. Designation: Atlas. Awaiting travel itinerary or experience logs."
            
            history.append({"role": "assistant", "content": sys_msg})
            log_interaction(clean_name, "assistant", sys_msg, intent="GREETING")
            
        elif intent == "OTHER":
            if persona == "Empathetic":
                sys_msg = f"I'm sorry {clean_name}, but I am only able to listen to travel experiences. Could we go back to talking about your trips?"
            else:
                sys_msg = f"Topic violation detected, {clean_name}. I am strictly authorized for travel experiences only. Revert to approved topic."
                
            history.append({"role": "assistant", "content": sys_msg})
            log_interaction(clean_name, "assistant", sys_msg, intent="OTHER")
            
        else: # TRAVEL
            response = generate_facilitator_response(msg, persona, clean_name)
            history.append({"role": "assistant", "content": response})
            log_interaction(clean_name, "assistant", response, intent="TRAVEL")
            
    except Exception as e:
        err_msg = f"System fault: The connection to the cognitive engine failed. Please try again. ({str(e)})"
        history.append({"role": "assistant", "content": err_msg})

    return history, history, ""

with gr.Blocks() as demo:
    gr.Markdown("## Atlas")
    gr.Markdown("Share your travel experiences. I am here to listen.")
    
    with gr.Row():
        name_input = gr.Textbox(label="Traveler Identification", placeholder="Who are you?", scale=1)
        persona_selector = gr.Radio(["Empathetic", "Robotic"], label="Select Listener Persona", value="Empathetic", scale=2)
    
    chatbot = gr.Chatbot()
    
    with gr.Row():
        msg = gr.Textbox(placeholder="I visited Hong Kong last week...", show_label=False, scale=8)
        send = gr.Button("Submit", scale=1)
        
    with gr.Row():
        flag_btn = gr.Button("⚠️ Report Misclassification", variant="secondary")
        clear = gr.Button("Wipe Memory")
    
    state_history = gr.State([])

    send.click(
        chat_step, 
        inputs=[msg, name_input, persona_selector, state_history], 
        outputs=[chatbot, state_history, msg]
    )
    msg.submit(
        chat_step, 
        inputs=[msg, name_input, persona_selector, state_history], 
        outputs=[chatbot, state_history, msg]
    )
    
    flag_btn.click(flag_last_interaction, inputs=[name_input, state_history], outputs=[])
    clear.click(lambda: ([], []), None, [chatbot, state_history])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Monochrome())