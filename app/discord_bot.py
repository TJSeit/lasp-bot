"""
discord_bot.py — Discord interface for the LASP bot.

Connects to Discord and exposes a !ask command (also aliased as !lasp) that
runs the same RAG pipeline used by the FastAPI app.

Environment variables (see ../.env.example):
    DISCORD_TOKEN           — Discord bot token (required)
    DISCORD_COMMAND_PREFIX  — Command prefix (default: !)

All other RAG configuration (FAISS_INDEX_DIR, OLLAMA_MODEL, …) is read from
the same environment variables as the FastAPI app (see rag.py).
"""

import asyncio
import os

import discord
from discord.ext import commands
from dotenv import load_dotenv

from rag import answer_query, build_rag_chain

load_dotenv()

DISCORD_TOKEN: str | None = os.getenv("DISCORD_TOKEN")
DISCORD_COMMAND_PREFIX: str = os.getenv("DISCORD_COMMAND_PREFIX", "!")

# Discord messages are capped at 2 000 characters; leave a small safety margin.
DISCORD_MAX_MESSAGE_LENGTH = 1900

# Shared RAG state initialised once in on_ready.
_rag_state: dict = {}

intents = discord.Intents.default()
intents.message_content = True  # Required to read message text.

bot = commands.Bot(command_prefix=DISCORD_COMMAND_PREFIX, intents=intents)


@bot.event
async def on_ready() -> None:
    """Load the FAISS index and initialise the Ollama client on login."""
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    try:
        retriever, llm_client = await asyncio.to_thread(build_rag_chain)
        _rag_state["retriever"] = retriever
        _rag_state["llm_client"] = llm_client
        print("RAG chain initialised successfully.")
    except Exception as exc:  # pragma: no cover
        print(f"Failed to initialise RAG chain: {exc}")


@bot.command(name="ask", aliases=["lasp"])
async def ask(ctx: commands.Context, *, question: str) -> None:
    """Ask a question about LASP.

    Usage:
        !ask What missions does LASP operate?
        !lasp What is the MAVEN mission?
    """
    if not _rag_state.get("retriever"):
        await ctx.send(
            "❌ The RAG chain is not ready yet. Please try again in a moment."
        )
        return

    async with ctx.typing():
        try:
            result = await asyncio.to_thread(
                answer_query,
                _rag_state["retriever"],
                _rag_state["llm_client"],
                question,
            )
        except Exception as exc:
            await ctx.send(f"❌ An error occurred while processing your question: {exc}")
            return

    answer = result["answer"]
    sources = result.get("sources", [])

    response = f"**Answer:**\n{answer}"

    if sources:
        source_lines = []
        for s in sources[:3]:  # Show at most three sources to keep the message tidy.
            label = s.get("source", "source")
            url = s.get("source_url", "")
            if url:
                source_lines.append(f"• [{label}](<{url}>)")
            else:
                source_lines.append(f"• {label}")
        response += "\n\n**Sources:**\n" + "\n".join(source_lines)

    # Truncate to stay within Discord's character limit.
    if len(response) > DISCORD_MAX_MESSAGE_LENGTH:
        response = response[: DISCORD_MAX_MESSAGE_LENGTH - 1] + "…"

    await ctx.send(response)


def main() -> None:
    """Entry point for running the Discord bot."""
    if not DISCORD_TOKEN:
        raise ValueError(
            "DISCORD_TOKEN environment variable is not set. "
            "Set it in your .env file or as an environment variable."
        )
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
