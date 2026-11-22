# Agentic-chatbot-using-LangGraph
# 🤖 Agentic AI Chatbot Using LangGraph

## 📌 Project Overview

The **Agentic AI Chatbot Using LangGraph** is an AI-powered chatbot that can understand user queries, make decisions, and use external tools to provide useful responses.

Unlike a normal chatbot, this project uses Agentic AI concepts to perform tasks such as web searching and weather information retrieval. It uses LangGraph to manage the workflow and Human-in-the-Loop (HITL) to allow human intervention when required.

## 🎯 Objectives

- Build an intelligent AI chatbot.
- Understand and respond to user queries.
- Integrate external tools and APIs.
- Implement Agentic AI using LangGraph.
- Add Human-in-the-Loop functionality.
- Monitor AI workflows using LangSmith.

## 🚀 Features

- **AI Chat:** Conversational interaction with users.
- **LangGraph:** Manages the agent workflow.
- **Groq LLM:** Provides fast AI-powered responses.
- **Web Search:** Retrieves information from the internet using Tavily.
- **Weather Information:** Retrieves weather information using a weather API.
- **Human-in-the-Loop (HITL):** Allows human intervention in the workflow.
- **LangSmith:** Helps monitor and debug AI workflows.
- **Streamlit:** Provides an interactive chatbot interface.
- **Docker:** Supports containerized deployment.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| LangChain | LLM and tool integration |
| LangGraph | Agent workflow management |
| Groq | Large Language Model |
| Tavily | Web search |
| OpenWeather API | Weather information |
| Streamlit | User interface |
| LangSmith | Monitoring and debugging |
| Git & GitHub

## 📂 Project Structure

```text
Agentic-chatbot-using-LangGraph/
│
├── app.py
├── agentic_chatbot_hitl_backend.py
├── requirements.txt
├── README.md
├── .env
```

*Note: The actual files and folders may vary depending on the repository version.*

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the Project Folder

```bash
cd Agentic-chatbot-using-LangGraph
```

### 3. Create a Conda Environment

```bash
conda create -n langagent python=3.11 -y
```

### 4. Activate Environment

```bash
conda activate langagent
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure Environment Variables

Create a `.env` file in the project directory.

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
OPENWEATHER_API_KEY=your_openweather_api_key

LANGSMITH_TR