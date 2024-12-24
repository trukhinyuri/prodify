#!/usr/bin/env bash
#
# ------------------------------------------------------------------------------
# This script sets up a Python virtual environment for running index.py and ask.py
# from "Prodify: Product assistant in coding".
# It creates a venv, activates it, updates pip, and installs all required packages.
# (C) IURII TRUKHIN, yuri@trukhin.com, 2024
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------------

set -e  # Exit immediately on error

# 1) Create a virtual environment (named "venv").
python3 -m venv venv

# 2) Activate the newly created venv.
# NOTE: For Windows, use "venv\Scripts\activate.bat" instead of below.
source venv/bin/activate

# 3) Upgrade pip inside the venv.
pip install --upgrade pip

# 4) Install the required dependencies for index.py and ask.py.
#    If any package is missing, add it here.
pip install \
    openai \
    prompt_toolkit \
    tqdm \
    rich \
    tiktoken \
    langchain_openai \
    langchain_chroma \
    langchain_community

echo ""
echo "============================================================"
echo "Virtual environment 'venv' is set up and all dependencies"
echo "for ask.py and index.py are installed."
echo "To activate the environment again later, run:"
echo "    source venv/bin/activate"
echo "Then you can run:"
echo "    python index.py"
echo " or python ask.py"
echo "============================================================"

