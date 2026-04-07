"""
Tests for the LASP Discord bot.

All Discord and RAG I/O is mocked so the tests run without any running
services, a real Discord connection, or a GPU.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_doc(content: str, source: str = "lasp_doc.pdf", source_url: str = ""):
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"source": source, "page": 1, "source_url": source_url}
    return doc


def _fresh_bot_module():
    """Remove the cached discord_bot module so each test gets a clean import.

    Only discord_bot needs to be cleared: its module-level _rag_state dict
    holds the live retriever/LLM references and must be empty at the start of
    each test. The rag module only contains read-only constants derived from
    environment variables and has no mutable module-level state.
    """
    for mod in list(sys.modules.keys()):
        if mod == "discord_bot":
            del sys.modules[mod]


# ---------------------------------------------------------------------------
# on_ready
# ---------------------------------------------------------------------------


class TestOnReady:
    """bot.on_ready initialises the RAG chain and stores state."""

    def test_rag_chain_stored_on_ready(self):
        _fresh_bot_module()

        fake_retriever = MagicMock()
        fake_llm = MagicMock()

        with patch("rag.build_rag_chain", return_value=(fake_retriever, fake_llm)), \
             patch("lasp_mcp.run_in_background"):
            import discord_bot

            fake_user = MagicMock()
            fake_user.id = 123456
            discord_bot.bot._connection.user = fake_user

            asyncio.run(discord_bot.on_ready())

            assert discord_bot._rag_state["retriever"] is fake_retriever
            assert discord_bot._rag_state["llm_client"] is fake_llm

    def test_on_ready_handles_build_failure_gracefully(self, capsys):
        _fresh_bot_module()

        with patch("rag.build_rag_chain", side_effect=RuntimeError("index not found")), \
             patch("lasp_mcp.run_in_background"):
            import discord_bot

            fake_user = MagicMock()
            fake_user.id = 123456
            discord_bot.bot._connection.user = fake_user

            asyncio.run(discord_bot.on_ready())

            # State should remain empty — no crash.
            assert not discord_bot._rag_state
            captured = capsys.readouterr()
            assert "Failed to initialise RAG chain" in captured.out


# ---------------------------------------------------------------------------
# !ask command
# ---------------------------------------------------------------------------


def _make_ctx():
    """Return a minimal mock of discord.ext.commands.Context."""
    ctx = MagicMock()
    ctx.send = AsyncMock()
    ctx.typing = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=None),
        __aexit__=AsyncMock(return_value=False),
    ))
    return ctx


class TestAskCommand:
    """!ask forwards the question to the RAG pipeline and sends the answer."""

    def _setup_rag_state(self, discord_bot, fake_retriever, fake_llm):
        """Directly populate _rag_state to skip the on_ready flow."""
        discord_bot._rag_state["retriever"] = fake_retriever
        discord_bot._rag_state["llm_client"] = fake_llm

    def test_successful_query_sends_answer(self):
        _fresh_bot_module()

        fake_retriever = MagicMock()
        fake_llm = MagicMock()
        rag_result = {
            "answer": "LASP studies space weather.",
            "sources": [
                {"source": "overview.pdf", "page": 1, "source_url": "https://lasp.colorado.edu/overview"},
            ],
        }

        with patch("rag.answer_query", return_value=rag_result):
            import discord_bot
            self._setup_rag_state(discord_bot, fake_retriever, fake_llm)
            ctx = _make_ctx()
            asyncio.run(discord_bot.ask(ctx, question="What does LASP study?"))

        ctx.send.assert_awaited_once()
        sent_text = ctx.send.call_args[0][0]
        assert "LASP studies space weather." in sent_text
        assert "overview.pdf" in sent_text

    def test_includes_source_url_as_link(self):
        _fresh_bot_module()

        fake_retriever = MagicMock()
        fake_llm = MagicMock()
        rag_result = {
            "answer": "Answer.",
            "sources": [
                {"source": "doc.pdf", "page": 2, "source_url": "https://example.com/doc"},
            ],
        }

        with patch("rag.answer_query", return_value=rag_result):
            import discord_bot
            self._setup_rag_state(discord_bot, fake_retriever, fake_llm)
            ctx = _make_ctx()
            asyncio.run(discord_bot.ask(ctx, question="question?"))

        sent_text = ctx.send.call_args[0][0]
        assert "https://example.com/doc" in sent_text

    def test_source_without_url_shows_plain_label(self):
        _fresh_bot_module()

        fake_retriever = MagicMock()
        fake_llm = MagicMock()
        rag_result = {
            "answer": "Answer.",
            "sources": [
                {"source": "plain.pdf", "page": 1, "source_url": ""},
            ],
        }

        with patch("rag.answer_query", return_value=rag_result):
            import discord_bot
            self._setup_rag_state(discord_bot, fake_retriever, fake_llm)
            ctx = _make_ctx()
            asyncio.run(discord_bot.ask(ctx, question="question?"))

        sent_text = ctx.send.call_args[0][0]
        assert "plain.pdf" in sent_text

    def test_rag_not_ready_sends_error_message(self):
        _fresh_bot_module()

        import discord_bot  # _rag_state is empty — RAG not initialised.
        ctx = _make_ctx()
        asyncio.run(discord_bot.ask(ctx, question="anything?"))

        ctx.send.assert_awaited_once()
        sent_text = ctx.send.call_args[0][0]
        assert "not ready" in sent_text.lower()

    def test_answer_query_exception_sends_error_message(self):
        _fresh_bot_module()

        fake_retriever = MagicMock()
        fake_llm = MagicMock()

        with patch("rag.answer_query", side_effect=RuntimeError("LLM unavailable")):
            import discord_bot
            self._setup_rag_state(discord_bot, fake_retriever, fake_llm)
            ctx = _make_ctx()
            asyncio.run(discord_bot.ask(ctx, question="question?"))

        ctx.send.assert_awaited_once()
        sent_text = ctx.send.call_args[0][0]
        assert "error" in sent_text.lower()
        assert "LLM unavailable" in sent_text

    def test_long_answer_is_truncated(self):
        _fresh_bot_module()

        fake_retriever = MagicMock()
        fake_llm = MagicMock()
        # Produce an answer that far exceeds the Discord character limit.
        rag_result = {
            "answer": "A" * 3000,
            "sources": [],
        }

        with patch("rag.answer_query", return_value=rag_result):
            import discord_bot
            self._setup_rag_state(discord_bot, fake_retriever, fake_llm)
            ctx = _make_ctx()
            asyncio.run(discord_bot.ask(ctx, question="question?"))

        sent_text = ctx.send.call_args[0][0]
        assert len(sent_text) <= discord_bot.DISCORD_MAX_MESSAGE_LENGTH
        assert sent_text.endswith("…")

    def test_at_most_three_sources_shown(self):
        _fresh_bot_module()

        fake_retriever = MagicMock()
        fake_llm = MagicMock()
        rag_result = {
            "answer": "Answer.",
            "sources": [
                {"source": f"doc{i}.pdf", "page": i, "source_url": ""} for i in range(5)
            ],
        }

        with patch("rag.answer_query", return_value=rag_result):
            import discord_bot
            self._setup_rag_state(discord_bot, fake_retriever, fake_llm)
            ctx = _make_ctx()
            asyncio.run(discord_bot.ask(ctx, question="question?"))

        sent_text = ctx.send.call_args[0][0]
        # Only the first three source filenames should appear.
        assert "doc0.pdf" in sent_text
        assert "doc1.pdf" in sent_text
        assert "doc2.pdf" in sent_text
        assert "doc3.pdf" not in sent_text
        assert "doc4.pdf" not in sent_text


# ---------------------------------------------------------------------------
# Conversation history (multi-turn)
# ---------------------------------------------------------------------------


class TestConversationHistory:
    """Per-channel history is built up across !ask calls."""

    def _setup_rag_state(self, discord_bot, fake_retriever, fake_llm):
        discord_bot._rag_state["retriever"] = fake_retriever
        discord_bot._rag_state["llm_client"] = fake_llm

    def _make_ctx(self, channel_id: int = 42):
        ctx = MagicMock()
        ctx.send = AsyncMock()
        ctx.typing = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=None),
            __aexit__=AsyncMock(return_value=False),
        ))
        ctx.channel.id = channel_id
        return ctx

    def test_history_accumulates_across_calls(self):
        _fresh_bot_module()

        fake_retriever = MagicMock()
        fake_llm = MagicMock()
        rag_results = iter([
            {"answer": "First answer.", "sources": []},
            {"answer": "Second answer.", "sources": []},
        ])

        with patch("rag.answer_query", side_effect=lambda *a, **kw: next(rag_results)):
            import discord_bot
            self._setup_rag_state(discord_bot, fake_retriever, fake_llm)

            ctx = self._make_ctx(channel_id=1)
            asyncio.run(discord_bot.ask(ctx, question="First question?"))
            asyncio.run(discord_bot.ask(ctx, question="Second question?"))

        history = discord_bot._conversation_history[1]
        assert len(history) == 4  # 2 user + 2 assistant messages
        assert history[0] == {"role": "user", "content": "First question?"}
        assert history[1] == {"role": "assistant", "content": "First answer."}
        assert history[2] == {"role": "user", "content": "Second question?"}
        assert history[3] == {"role": "assistant", "content": "Second answer."}

    def test_history_passed_to_answer_query(self):
        _fresh_bot_module()

        fake_retriever = MagicMock()
        fake_llm = MagicMock()
        captured_calls: list = []

        def fake_answer_query(retriever, llm_client, question, history=None):
            captured_calls.append({"question": question, "history": history})
            return {"answer": f"Answer to: {question}", "sources": []}

        with patch("rag.answer_query", side_effect=fake_answer_query):
            import discord_bot
            self._setup_rag_state(discord_bot, fake_retriever, fake_llm)

            ctx = self._make_ctx(channel_id=2)
            asyncio.run(discord_bot.ask(ctx, question="First?"))
            asyncio.run(discord_bot.ask(ctx, question="Second?"))

        # First call should have no history.
        assert captured_calls[0]["history"] == []
        # Second call should include the first exchange.
        assert captured_calls[1]["history"] == [
            {"role": "user", "content": "First?"},
            {"role": "assistant", "content": "Answer to: First?"},
        ]

    def test_history_is_per_channel(self):
        _fresh_bot_module()

        fake_retriever = MagicMock()
        fake_llm = MagicMock()

        with patch("rag.answer_query", return_value={"answer": "Answer.", "sources": []}):
            import discord_bot
            self._setup_rag_state(discord_bot, fake_retriever, fake_llm)

            ctx_a = self._make_ctx(channel_id=10)
            ctx_b = self._make_ctx(channel_id=20)
            asyncio.run(discord_bot.ask(ctx_a, question="Channel A question?"))
            asyncio.run(discord_bot.ask(ctx_b, question="Channel B question?"))

        assert 10 in discord_bot._conversation_history
        assert 20 in discord_bot._conversation_history
        history_a = discord_bot._conversation_history[10]
        history_b = discord_bot._conversation_history[20]
        assert history_a[0]["content"] == "Channel A question?"
        assert history_b[0]["content"] == "Channel B question?"

    def test_history_trimmed_when_exceeds_max_turns(self):
        _fresh_bot_module()

        fake_retriever = MagicMock()
        fake_llm = MagicMock()

        with patch("rag.answer_query", return_value={"answer": "Ans.", "sources": []}):
            import discord_bot
            self._setup_rag_state(discord_bot, fake_retriever, fake_llm)

            ctx = self._make_ctx(channel_id=99)
            # Send MAX_HISTORY_TURNS + 2 messages to trigger trimming.
            for i in range(discord_bot.MAX_HISTORY_TURNS + 2):
                asyncio.run(discord_bot.ask(ctx, question=f"Q{i}?"))

        history = discord_bot._conversation_history[99]
        assert len(history) == discord_bot.MAX_HISTORY_MESSAGES

    def test_history_not_updated_on_error(self):
        _fresh_bot_module()

        fake_retriever = MagicMock()
        fake_llm = MagicMock()

        with patch("rag.answer_query", side_effect=RuntimeError("LLM down")):
            import discord_bot
            self._setup_rag_state(discord_bot, fake_retriever, fake_llm)

            ctx = self._make_ctx(channel_id=55)
            asyncio.run(discord_bot.ask(ctx, question="Will fail?"))

        # History should be untouched when the query raised an error.
        assert discord_bot._conversation_history.get(55) is None

    def test_existing_history_preserved_on_error(self):
        _fresh_bot_module()

        fake_retriever = MagicMock()
        fake_llm = MagicMock()
        answers = iter([
            {"answer": "Good answer.", "sources": []},
            RuntimeError("LLM down"),
        ])

        def fake_answer_query(*args, **kwargs):
            val = next(answers)
            if isinstance(val, Exception):
                raise val
            return val

        with patch("rag.answer_query", side_effect=fake_answer_query):
            import discord_bot
            self._setup_rag_state(discord_bot, fake_retriever, fake_llm)

            ctx = self._make_ctx(channel_id=66)
            asyncio.run(discord_bot.ask(ctx, question="First question?"))
            # Record history after the first successful call.
            history_after_first = list(discord_bot._conversation_history[66])
            asyncio.run(discord_bot.ask(ctx, question="Will fail?"))

        # History should be the same as after the first successful call.
        assert discord_bot._conversation_history[66] == history_after_first


# ---------------------------------------------------------------------------
# main() / DISCORD_TOKEN validation
# ---------------------------------------------------------------------------


class TestMain:
    def test_raises_if_discord_token_missing(self):
        _fresh_bot_module()

        with patch.dict("os.environ", {}, clear=True):
            import discord_bot
            discord_bot.DISCORD_TOKEN = None

            with pytest.raises(ValueError, match="DISCORD_TOKEN"):
                discord_bot.main()

    def test_calls_bot_run_with_token(self):
        _fresh_bot_module()

        with patch.dict("os.environ", {"DISCORD_TOKEN": "fake-token"}):
            import discord_bot
            discord_bot.DISCORD_TOKEN = "fake-token"

            with patch.object(discord_bot.bot, "run") as mock_run:
                discord_bot.main()

            mock_run.assert_called_once_with("fake-token")
