"""
conftest.py — Shared pytest configuration for the app test suite.

Adds the app directory to sys.path so that test modules can import app source
files (rag, discord_bot, main, lasp_mcp) directly without package qualification.
"""

import os
import sys

# Ensure the app/ directory is on sys.path for all test modules.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Ensure the tests/ directory is on sys.path so shared helpers in _helpers.py
# can be imported directly by test modules.
sys.path.insert(0, os.path.dirname(__file__))
