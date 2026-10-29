import os
import re
import time

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
BASE_URL = "https://api.groq.com/openai/v1"

GOV_MODEL = os.getenv("GOV_MODEL", "openai/gpt-oss-120b")
OPP_MODEL = os.getenv("OPP_MODEL", "openai/gpt-oss-20b")

MAX_ROUNDS = 10

client = OpenAI(api_key=API_KEY, base_url=BASE_URL) if API_KEY else None


def government_prompt(country: str, topic: str) -> str:
    return f"""You are a senior spokesperson for the CURRENT GOVERNMENT of {country}.
You are in a televised political debate with a member of the opposition.
Debate focus: {topic or "the overall performance of the government"}.

Rules you must follow:
1. Defend the government's record and policies with VALID, VERIFIABLE data and statistics
   (GDP growth, inflation, unemployment, poverty rate, debt-to-GDP, budgets, investment, etc.).
2. Name the source and year for every number (e.g. "World Bank, 2024", "IMF WEO April 2025",
   "national statistics office, 2023"). Never invent figures. If you are not sure of an exact
   value, say "approximately" or give a range rather than making one up.
3. Directly answer the opposition's last point before making your own new point.
4. Be respectful and professional - no personal insults.
5. Keep each turn under 180 words. Do not speak for the opposition."""


def opposition_prompt(country: str, topic: str) -> str:
    return f"""You are a leading member of the main OPPOSITION in {country}.
You are in a televised political debate with a government spokesperson.
Debate focus: {topic or "the overall performance of the government"}.

Rules you must follow:
1. Criticise the government's record and propose alternatives, using VALID, VERIFIABLE data
   and statistics (inflation, unemployment, poverty, debt, corruption indices, inequality, etc.).
2. Name the source and year for every number (e.g. "World Bank, 2024", "IMF WEO April 2025",
   "Transparency International CPI 2024"). Never invent figures. If you are not sure of an
   exact value, say "approximately" or give a range rather than making one up.
3. Directly rebut the government's last point before making your own new point.
4. Be respectful and professional - no personal insults.
5. Keep each turn under 180 words. Do not speak for the government."""


HISTORY_TURNS_KEPT = 6
MAX_RETRIES = 5
WAIT_RE = re.compile(r"try again in ([\d.]+)(ms|s)", re.IGNORECASE)


def _extract_wait_seconds(error_message: str, attempt: int) -> float:
    match = WAIT_RE.search(error_message)
    if match:
        value, unit = float(match.group(1)), match.group(2).lower()
        seconds = value / 1000 if unit == "ms" else value
        return min(max(seconds, 1.0) + 0.5, 20.0)
    return min(2 ** attempt, 20.0)


def ask_llm(model: str, system_prompt: str, history: list, me: str) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    if not history:
        messages.append({"role": "user", "content": "The moderator opens the debate. Please give your opening statement."})
    for speaker, text in history[-HISTORY_TURNS_KEPT:]:
        role = "assistant" if speaker == me else "user"
        messages.append({"role": role, "content": text})

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=700,
            )
            return (response.choices[0].message.content or "").strip()
        except RateLimitError as e:
            last_error = e
            time.sleep(_extract_wait_seconds(str(e), attempt))
    raise last_error


def run_debate(country: str, topic: str, rounds: int):
    if client is None:
        raise gr.Error("GROQ_API_KEY is missing. Put it in a .env file (see README).")
    country = (country or "").strip()
    if not country:
        raise gr.Error("Please type a country name.")
    rounds = max(1, min(int(rounds), MAX_ROUNDS))

    gov_sys = government_prompt(country, topic)
    opp_sys = opposition_prompt(country, topic)

    history = []
    chat = []

    for r in range(1, rounds + 1):
        try:
            gov = ask_llm(GOV_MODEL, gov_sys, history, me="gov")
        except Exception as e:
            raise gr.Error(f"Government model error: {e}")
        history.append(("gov", gov))
        chat.append({"role": "user", "content": f"**🏛️ Government — Round {r}**\n\n{gov}"})
        yield chat, f"Round {r}/{rounds}: government spoke…"

        try:
            opp = ask_llm(OPP_MODEL, opp_sys, history, me="opp")
        except Exception as e:
            raise gr.Error(f"Opposition model error: {e}")
        history.append(("opp", opp))
        chat.append({"role": "assistant", "content": f"**⚖️ Opposition — Round {r}**\n\n{opp}"})
        yield chat, f"Round {r}/{rounds} finished."

    yield chat, f"✅ Debate on {country} finished after {rounds} round(s)."


with gr.Blocks(title="LLM Political Debate") as demo:
    gr.Markdown(
        "# 🗳️ LLM Political Debate\n"
        f"Two free LLMs debate a country's politics. **Government:** `{GOV_MODEL}` · "
        f"**Opposition:** `{OPP_MODEL}` (via Groq)."
    )
    with gr.Row():
        country_in = gr.Textbox(label="Country", placeholder="e.g. Egypt, Brazil, India", scale=2)
        topic_in = gr.Textbox(label="Topic (optional)", placeholder="e.g. economy, healthcare, education", scale=2)
        rounds_in = gr.Slider(1, MAX_ROUNDS, value=3, step=1, label="Rounds (max 10)", scale=1)
    start_btn = gr.Button("Start debate", variant="primary")
    status = gr.Markdown()
    chatbot = gr.Chatbot(type="messages", height=600, label="Debate")
    gr.Markdown("*AI-generated content. Statistics may be outdated or wrong — always verify the cited sources.*")

    start_btn.click(run_debate, inputs=[country_in, topic_in, rounds_in], outputs=[chatbot, status])

if __name__ == "__main__":
    demo.launch()
