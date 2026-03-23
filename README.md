# 🌍 Atlas: The Dedicated Travel Listener

[![Atlas - Hugging Face Space](https://img.shields.io/badge/🤗_Open_in_Spaces-Atlas-blue.svg)](https://huggingface.co/spaces/talhahkk/atlas)

Atlas is a highly constrained, multi-agent AI conversational interface built to do one thing exclusively: **listen to your travel experiences.** Unlike standard conversational models that try to be helpful tour guides or Wikipedia engines, Atlas is designed as a pure facilitator. It enforces strict topic boundaries, asks open-ended questions, and encourages the user to drive the narrative.

## ✨ Key Features

* **Strict Topic Enforcement:** Atlas refuses to discuss coding, math, general chatting, or any topic outside of personal travel experiences.
* **Anti-Injection Protocol:** User inputs are heavily sanitized and quarantined within XML tags (`<user_input>`) to prevent jailbreaks, system overrides, or roleplay injections.
* **Context-Aware Memory:** Maintains a rolling 7-message short-term memory, allowing it to correctly parse conversational fragments (e.g., understanding that the answer "the sun" is a travel detail when asked about a beach).
* **Dual Personas:** Users can toggle between an "Empathetic" (warm, validating) and "Robotic" (neutral, highly concise) listener.
* **Continuous Data Logging:** Seamlessly integrated with Hugging Face `CommitScheduler` to log conversation interactions securely into a dedicated JSONL dataset.

## 🧠 System Architecture

Atlas operates using a dual-LLM routing system to separate cognitive logic from conversational generation:

### 1. The Bouncer (Intent Classifier)
Powered by **GPT-4o-mini** at `temperature=0.0`. Every user input passes through the Bouncer first. It acts as a rigid gatekeeper, analyzing the text alongside the recent chat history to classify the intent into one of three strict categories:
* `GREETING`: Triggers a hardcoded, identity-establishing welcome message.
* `OTHER`: Triggers a hardcoded rejection, forcing the user back on topic.
* `TRAVEL`: Approves the input and passes the data to the Response Generator.

### 2. The Mirror (Response Generator)
Powered by a highly constrained generation model (e.g., **GPT-4.1**). The Mirror is bound by extreme rules:
* **Listen, Don't Lecture:** It is strictly forbidden from offering facts, trivia, or advice about locations. 
* **Brevity:** Capped at 12-20 words per response to ensure the user speaks more than the AI.
* **Open-Ended Facilitation:** Trained to avoid multiple-choice questions, instead using phrases like *"That's interesting, tell me more"* to prompt deeper storytelling.
---

---
title: Atlas
emoji: 🌍
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: false
short_description: A multi-agent chatbot that listens to travel experiences.
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference