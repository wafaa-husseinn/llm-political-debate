"""
LLM Political Debate
--------------------
Two free LLMs (served by Groq's free tier) debate the politics of a country:
  * LLM 1 speaks for the GOVERNMENT
  * LLM 2 speaks for the OPPOSITION
The user picks the country and the number of rounds (max 10) in a Gradio UI.
"""

import os
import re
import time

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()  # reads GROQ_API_KEY from a local .env file (never committed)

API_KEY = os.getenv("GROQ_API_KEY")
BASE_URL = "https://api.groq.com/openai/v1"  # Groq is OpenAI-compatible

# Free models on Groq. You can override them in .env if Groq changes its list:
# https://console.groq.com/docs/models
GOV_MODEL = os.getenv("GOV_MODEL", "openai/gpt-oss-120b")
OPP_MODEL = os.getenv("OPP_MODEL", "openai/gpt-oss-20b")

MAX_ROUNDS = 10  # hard limit required by the assignment

client = OpenAI(api_key=API_KEY, base_url=BASE_URL) if API_KEY else None


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
# Groq's free tier has a low tokens-per-minute (TPM) limit. Keeping the model's
# own context short and retrying on 429 keeps long debates (up to 10 rounds)
# running reliably without the user having to do anything.
HISTORY_TURNS_KEPT = 6   # last 6 messages (~3 rounds) of context sent to each model
MAX_RETRIES = 5
WAIT_RE = re.compile(r"try again in ([\d.]+)(ms|s)", re.IGNORECASE)


def _extract_wait_seconds(error_message: str, attempt: int) -> float:
    """Read Groq's own 'try again in Xms/Xs' hint from the error, or fall back
    to exponential backoff (2, 4, 8, 16... seconds, capped at 20)."""
    match = WAIT_RE.search(error_message)
    if match:
        value, unit = float(match.group(1)), match.group(2).lower()
        seconds = value / 1000 if unit == "ms" else value
        return min(max(seconds, 1.0) + 0.5, 20.0)  # small safety margin, cap at 20s
    return min(2 ** attempt, 20.0)


def ask_llm(model: str, system_prompt: str, history: list, me: str) -> str:
    """Send the debate so far to one LLM, retrying automatically on rate limits.

    `history` is a list of (speaker, text). The speaker's own past turns are sent
    as 'assistant' messages and the other side's turns as 'user' messages, so each
    model sees the debate from its own point of view. Only the most recent turns
    are sent (HISTORY_TURNS_KEPT) to stay well under the free tier's token limit.
    """
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
            wait_s = _extract_wait_seconds(str(e), attempt)
            time.sleep(wait_s)
    raise last_error


# ---------------------------------------------------------------------------
# Debate loop (generator -> the UI updates after every turn)
# ---------------------------------------------------------------------------
def run_debate(country: str, topic: str, rounds: int):
    if client is None:
        raise gr.Error("GROQ_API_KEY is missing. Put it in a .env file (see README).")
    country = (country or "").strip()
    if not country:
        raise gr.Error("Please type a country name.")
    rounds = max(1, min(int(rounds), MAX_ROUNDS))  # enforce the 10-round limit

    gov_sys = government_prompt(country, topic)
    opp_sys = opposition_prompt(country, topic)

    history = []  # (speaker, text)
    chat = []     # what the Gradio Chatbot shows

    for r in range(1, rounds + 1):
        # Government speaks
        try:
            gov = ask_llm(GOV_MODEL, gov_sys, history, me="gov")
        except Exception as e:
            raise gr.Error(f"Government model error: {e}")
        history.append(("gov", gov))
        chat.append({"role": "user", "content": f"**🏛️ Government — Round {r}**\n\n{gov}"})
        yield chat, f"Round {r}/{rounds}: government spoke…"

        # Opposition replies
        try:
            opp = ask_llm(OPP_MODEL, opp_sys, history, me="opp")
        except Exception as e:
            raise gr.Error(f"Opposition model error: {e}")
        history.append(("opp", opp))
        chat.append({"role": "assistant", "content": f"**⚖️ Opposition — Round {r}**\n\n{opp}"})
        yield chat, f"Round {r}/{rounds} finished."

    yield chat, f"✅ Debate on {country} finished after {rounds} round(s)."


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
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
