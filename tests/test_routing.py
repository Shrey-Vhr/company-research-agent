"""Routing tests: each routing function gets a small state dict and must
return the right next node. No model or API calls."""

from research_agent.config import MAX_REVISIONS
from research_agent.graph.build import (
    after_critic,
    after_executor,
    after_human,
    after_planner,
    after_replanner,
    after_writer,
)


# after_planner: scope check
def test_out_of_scope_goes_to_refuse():
    assert after_planner({"in_scope": False, "plan": []}) == "refuse"


def test_in_scope_with_plan_goes_to_executor():
    assert after_planner({"in_scope": True, "plan": ["Get TCS financials"]}) == "executor"


def test_in_scope_but_empty_plan_goes_to_refuse():
    assert after_planner({"in_scope": True, "plan": []}) == "refuse"


# after_executor: one step at a time
def test_steps_left_loops_back_to_executor():
    assert after_executor({"plan": ["a", "b", "c"], "current_step": 1}) == "executor"


def test_all_steps_done_goes_to_writer():
    assert after_executor({"plan": ["a", "b", "c"], "current_step": 3}) == "writer"


# after_writer: skip_critic is only for the eval comparison
def test_skip_critic_with_auto_approve_goes_to_finalize():
    assert after_writer({"skip_critic": True, "auto_approve": True}) == "finalize"


def test_skip_critic_alone_goes_to_human_review():
    assert after_writer({"skip_critic": True}) == "human_review"


def test_normal_run_goes_to_critic():
    assert after_writer({}) == "critic"


# after_critic: revise, review or finalize
def test_failed_draft_with_revisions_left_goes_to_writer():
    assert after_critic({"critic_passed": False, "revision_count": 0}) == "writer"


def test_failed_draft_at_limit_goes_to_human_review():
    state = {"critic_passed": False, "revision_count": MAX_REVISIONS}
    assert after_critic(state) == "human_review"


def test_failed_draft_at_limit_with_auto_approve_goes_to_finalize():
    state = {"critic_passed": False, "revision_count": MAX_REVISIONS, "auto_approve": True}
    assert after_critic(state) == "finalize"


def test_passed_draft_goes_to_human_review():
    assert after_critic({"critic_passed": True, "revision_count": 0}) == "human_review"


def test_passed_draft_with_auto_approve_goes_to_finalize():
    state = {"critic_passed": True, "revision_count": 0, "auto_approve": True}
    assert after_critic(state) == "finalize"


# after_human: approve or change request
def test_change_request_goes_to_replanner():
    assert after_human({"human_feedback": "Also give the CEO"}) == "replanner"


def test_approve_goes_to_finalize():
    assert after_human({"human_feedback": ""}) == "finalize"


# after_replanner: new facts needed, or wording only
def test_new_steps_go_to_executor():
    assert after_replanner({"plan": ["a", "b", "c"], "current_step": 2}) == "executor"


def test_no_new_steps_go_to_writer():
    assert after_replanner({"plan": ["a", "b"], "current_step": 2}) == "writer"