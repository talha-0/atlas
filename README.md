# 🛍️ Aura: The Dedicated Shopping Listener

[![Aura - Hugging Face Space](https://img.shields.io/badge/🤗_Open_in_Spaces-Aura-blue.svg)](https://huggingface.co/spaces/talhahkk/atlas)

Aura is a highly constrained, multi-agent AI conversational interface built to do one thing exclusively: **listen to your shopping experiences.** Unlike standard conversational models that act as personal shoppers or product reviewers, Aura is designed as a pure facilitator. It enforces strict topic boundaries, asks open-ended questions, and encourages the user to drive the narrative about their purchases and store visits.

## ✨ Key Features

* **Strict Topic Enforcement:** Aura refuses to discuss coding, math, general chatting, or any topic outside of personal shopping experiences.
* **Anti-Injection Protocol:** User inputs are heavily sanitized and quarantined within XML tags (`<user_input>`) to prevent jailbreaks, system overrides, or roleplay injections.
* **Context-Aware Memory:** Maintains a rolling 7-message short-term memory, allowing it to correctly parse conversational fragments.
* **Dual Personas:** Users can toggle between an "Empathetic" (warm, validating) and "Robotic" (neutral, highly concise) listener.
* **Continuous Data Logging:** Seamlessly integrated with Hugging Face `CommitScheduler` to log conversation interactions securely into a dedicated JSONL dataset.

## 🧠 System Architecture

Aura operates using a dual-LLM routing system to separate cognitive logic from conversational generation:

### 1. The Bouncer (Intent Classifier)
Powered by **GPT-4o-mini** at `temperature=0.0`. Every user input passes through the Bouncer first. It acts as a rigid gatekeeper, analyzing the text alongside the recent chat history to classify the intent:
* `GREETING`: Triggers a hardcoded, identity-establishing welcome message.
* `OTHER`: Triggers a hardcoded rejection, forcing the user back on topic.
* `SHOPPING`: Approves the input and passes the data to the Response Generator.

### 2. The Mirror (Response Generator)
Powered by a highly constrained generation model. The Mirror is bound by extreme rules:
* **Listen, Don't Lecture:** It is strictly forbidden from offering facts, trivia, or reviews about brands and products. 
* **Brevity:** Capped at 15 words per response to ensure the user speaks more than the AI.
* **Open-Ended Facilitation:** Avoids specific product interrogations, using phrases like *"That sounds like a great find, what else caught your eye?"* to prompt deeper storytelling.