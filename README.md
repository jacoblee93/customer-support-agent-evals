# Customer support agent evals

This repo demonstrates some techniques for evaluating agent trajectories.

The example agent is found in `customer_support/agent.py` and models a customer support agent for an airline.
The agent has access to several tools for tasks such as checking flight information, rebooking, and looking up company policy, found in `customer_support/tools.py`. The `tests/` folder contains the evaluation suite.

## Quickstart

First, install required dependencies as specified in this project's `pyproject.toml`:

```bash
pip install .
```

Next, you'll need to set some required environment variables:

```
# The agent itself uses gemini-2.0-flash through Google AI Studio
# https://aistudio.google.com/
export GOOGLE_API_KEY="YOUR_KEY_HERE"

# The LLM-as-judge evaluators use OpenAI's o3-mini model
# https://platform.openai.com/
export OPENAI_API_KEY="YOUR_KEY_HERE"

# Optional for LangSmith tracing and experiment tracking
# export LANGSMITH_API_KEY="YOUR_KEY_HERE"
# export LANGSMITH_TRACING=true
```

Once you've done that, you can start a chat session with:

```python
python repl.py
```

You can run the evaluation suite using `pytest`:

```python
pytest
```

> [!IMPORTANT]
> If you are using LangSmith, comment back in the `@pytest.mark.langsmith` decorator above the test to enable tracking.
