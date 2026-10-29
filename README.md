# 🗳️ LLM Political Debate

A Gradio app where **two free LLMs** hold a political debate about a country you choose:

- **LLM 1 – Government:** defends the current government's record.
- **LLM 2 – Opposition:** criticises it and proposes alternatives.
- Both are instructed (in their system prompts) to back every argument with valid data and statistics and to cite the source and year.
- The country (and an optional topic) is entered by the user.
- The debate is limited to **10 rounds** maximum (one round = government turn + opposition turn).

## Free LLMs used
Served by [Groq](https://console.groq.com) on its free tier through its OpenAI-compatible API:

| Side | Default model |
|---|---|
| Government | `openai/gpt-oss-120b` |
| Opposition | `openai/gpt-oss-20b` |

You can switch models in `.env` (see `.env.example`).

## Run it locally
```bash
git clone https://github.com/<your-username>/llm-political-debate.git
cd llm-political-debate
python -m venv venv
# Windows: venv\Scripts\activate    |   Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
# open .env and paste your free Groq API key
python app.py
```
Open http://127.0.0.1:7860, type a country, pick the number of rounds and click **Start debate**.

## Project structure
```
app.py            # Gradio UI + debate logic
requirements.txt  # dependencies
.env.example      # template for your API key (the real .env is git-ignored)
.gitignore
```

## Note
The API key is read from an environment variable and is **not** included in this repository.
LLM output may contain outdated or inaccurate statistics — verify the cited sources.
