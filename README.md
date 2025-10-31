---
title: Agent's Co-pilot
sdk: fastapi
---
# Agent's Co-pilot 

**A Full-Stack AI application that serves as a smart assistant for customer support teams, providing real-time, high-quality reply suggestions.**

This project was built from scratch, covering everything from data cleaning and model fine-tuning to backend API development and a responsive React frontend.


---

## Problem It Solves
Customer support teams often face high volumes of repetitive queries. This leads to agent burnout and inconsistent reply quality. "Agent's Co-pilot" solves this by:
* **Increasing Efficiency:** Instantly suggests 3 relevant replies, reducing response time.
* **Ensuring Quality:** Uses a fine-tuned AI model to provide professional and consistent answers.
* **Reducing Training:** Acts as an expert assistant for new agents from day one.

---

##  Tech Stack

This project is built using two main components: a Python backend and a React frontend.

### Backend (The Brain)
* **Python 3.11**
* **Hugging Face `transformers`:** For loading and fine-tuning the AI model.
* **PyTorch:** The deep learning framework used for training.
* **Pandas:** For cleaning and preparing the Twitter customer support dataset.
* **FastAPI:** For creating the high-speed, lightweight API to serve the model.
* **Uvicorn:** As the server to run the FastAPI application.

### Frontend (The Face)
* **React.js (with Vite):** For building the fast, interactive user interface.
* **Axios:** For making requests from the frontend to the backend API.
* **CSS3:** For modern, responsive, and eye-catching styling.

---

##  The AI Model

The core of this application is a fine-tuned version of Google's **`flan-t5-small`** model.

1.  **Data:** The model was trained on the [Customer Support on Twitter dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).
2.  **Cleaning:** The raw dataset was cleaned using Pandas to create 690,000+ high-quality `prompt` (customer query) and `completion` (agent reply) pairs.
3.  **Fine-Tuning:** The `flan-t5-small` model was fine-tuned on a subset of this data (2,000 examples) using the Hugging Face `Trainer` API to specialize it for customer support conversations.
4.  **Inference:** The API uses **Sampling (`do_sample=True`)** with `top_k=50` and `temperature=0.9` to generate diverse and creative suggestions, rather than repetitive ones.

---

## How to Run This Project

### 1. Backend Server
*(Requires Python 3.11+)*

```bash
# 1. Clone the repository and go to the backend folder
git clone [your-github-repo-link]
cd Agent-s-Copilot

# 2. Create a virtual environment and activate it
python -m venv .venv
.\.venv\Scripts\Activate

# 3. Install required libraries
pip install -r requirements.txt
# (You will need: fastapi, uvicorn, torch, transformers, pandas)

# 4. Run the API server
uvicorn api:app
# Server will run on [http://127.0.0.1:8000](http://127.0.0.1:8000)

# 1. Open a new terminal and go to the frontend folder
cd frontend

# 2. Install dependencies
npm install

# 3. Run the development server
npm run dev
# App will run on http://localhost:5173
