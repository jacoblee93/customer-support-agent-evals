import pytest
from langsmith import testing as t
from langgraph.checkpoint.memory import MemorySaver
from customer_support.db import update_dates, db
import uuid
from langchain_core.messages.utils import convert_to_openai_messages
from customer_support.agent import initialize_graph

from agentevals.trajectory.match import create_trajectory_match_evaluator

from tests.data import (
    policy_check_inputs_trajectory,
    policy_check_reference_trajectory,
    test_date,
)

thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's flight information
        "passenger_id": "3442 587242",
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}


@pytest.fixture(scope="session")
def setup_db():
    update_dates(db, now=test_date)


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "inputs,reference_outputs",
    [(policy_check_inputs_trajectory, policy_check_reference_trajectory)],
)
def test_checks_policies(inputs, reference_outputs) -> None:
    checkpointer = MemorySaver()
    graph = initialize_graph(checkpointer, test_date)
    res = graph.invoke(inputs, config)

    # for nicer display convert to OpenAI format when logging
    outputs = {"messages": convert_to_openai_messages(res["messages"])}
    t.log_outputs(outputs)

    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode="superset",
        tool_args_match_mode="ignore",
    )
    evaluator_result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert evaluator_result["score"]
