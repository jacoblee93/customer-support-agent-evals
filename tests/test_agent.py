import pytest
from langsmith import testing as t
from langgraph.checkpoint.memory import MemorySaver
from customer_support.db import update_dates, db
import uuid
from langchain_core.messages.utils import convert_to_openai_messages
from customer_support.agents.swarm import initialize_swarm_agent
from agentevals.trajectory.match import create_trajectory_match_evaluator
from agentevals.trajectory.llm import create_trajectory_llm_as_judge

from langchain_google_genai import ChatGoogleGenerativeAI

from tests.data import (
    policy_check_inputs_trajectory,
    policy_check_reference_trajectory,
    handoff_check_inputs_trajectory,
    test_date,
)
from tests.prompts import TRAJECTORY_EFFICIENCY_PROMPT

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


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


@pytest.mark.langsmith
@pytest.mark.parametrize(
    "inputs,reference_outputs",
    [(policy_check_inputs_trajectory, policy_check_reference_trajectory)],
)
def test_checks_policies_before_upgrade(inputs, reference_outputs) -> None:
    checkpointer = MemorySaver()
    graph = initialize_swarm_agent(llm, checkpointer, test_date)
    res = graph.invoke(inputs, config)

    # for nicer display convert to OpenAI format when logging
    outputs = {"messages": convert_to_openai_messages(res["messages"])}
    t.log_outputs(outputs)

    def lookup_policy_query_matcher(tool_args: dict, reference_tool_args: dict):
        if reference_tool_args.get("query") and "upgrade" in reference_tool_args.get(
            "query"
        ):
            return "upgrade" in tool_args.get("query")
        # Ignore for other policy query matches
        return True

    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode="superset",
        tool_args_match_mode="ignore",
        tool_args_match_overrides={
            "lookup_policy": lookup_policy_query_matcher,
        },
    )

    evaluator_result = evaluator(outputs=outputs, reference_outputs=reference_outputs)
    assert evaluator_result["score"]


# @pytest.mark.langsmith
# @pytest.mark.parametrize(
#     "inputs",
#     [handoff_check_inputs_trajectory],
# )
# def test_efficient_handoff(inputs) -> None:
#     checkpointer = MemorySaver()
#     graph = initialize_swarm_agent(llm, checkpointer, test_date)
#     res = graph.invoke(inputs, config)

#     # for nicer display convert to OpenAI format when logging
#     outputs = {"messages": convert_to_openai_messages(res["messages"])}
#     t.log_outputs(outputs)

#     evaluator = create_trajectory_llm_as_judge(
#         model="openai:o3-mini",
#         prompt=TRAJECTORY_EFFICIENCY_PROMPT,
#         choices=[0, 0.3, 0.5, 0.7, 1],
#     )
#     evaluator_result = evaluator(outputs=outputs)
#     print(evaluator_result)
