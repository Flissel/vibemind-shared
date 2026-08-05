"""Offline validation for the provenance-bound Brain-to-OpenFang V1 closure.

This module packages the normative schemas and relevant closed validator logic
from Outer commit f0001e40f306745f60818a5bbd3065377dacd443.  It validates
continuity only; it neither dispatches work nor proves runtime execution.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from importlib import resources

from jsonschema import Draft202012Validator, FormatChecker


SCHEMA_FILES = {
    "channel-intent": "channel-intent-v1.schema.json",
    "brain-plan": "brain-plan-v1.schema.json",
    "space-execution-contract": "space-execution-contract-v1.schema.json",
    "brain-plan-lifecycle": "brain-plan-lifecycle-v1.schema.json",
    "brain-openfang-handoff": "brain-openfang-handoff-v1.schema.json",
}

# The snapshot's canonical values are embedded because the packaged closure has
# no external registry dependency.  They originate from the normative outer
# commit identified in schema-manifest-v1.json.
CANONICAL_SPACE_IDS = frozenset(
    {
        "agentfarm",
        "bubbles",
        "coding",
        "desktop",
        "flowzen",
        "ideas",
        "minibook",
        "mirofish",
        "n8n",
        "research",
        "rowboat",
        "schedule",
        "video",
    }
)


class ContractValidationError(ValueError):
    """Raised when a document is not valid for a packaged shared contract."""


def load_schema(document_type: str) -> dict[str, object]:
    """Load one schema by its stable document-type key."""
    try:
        filename = SCHEMA_FILES[document_type]
    except KeyError as error:
        known_types = ", ".join(sorted(SCHEMA_FILES))
        raise ContractValidationError(
            f"Unknown document type {document_type!r}; expected one of: {known_types}."
        ) from error

    value = json.loads(
        resources.files("vibemind_shared.contracts")
        .joinpath("schemas", filename)
        .read_text(encoding="utf-8")
    )
    if not isinstance(value, dict):
        raise ContractValidationError(f"Schema for {document_type!r} is not an object.")
    return value


def validate_document(document_type: str, document: Mapping[str, object]) -> None:
    """Validate one parsed document against its packaged V1 schema."""
    if document_type == "brain-plan":
        _reject_brain_plan_application_targets(document)

    schema = load_schema(document_type)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(document), key=lambda error: list(error.path))
    if errors:
        first_error = errors[0]
        location = ".".join(str(part) for part in first_error.path) or "<root>"
        raise ContractValidationError(
            f"{document_type} is invalid at {location}: {first_error.message}"
        )

    if document_type == "brain-plan":
        _validate_brain_plan_graph(document)
    elif document_type == "brain-plan-lifecycle":
        _validate_brain_plan_lifecycle(document)


def validate_brain_plan_admission_bundle(
    channel_intent: Mapping[str, object],
    brain_plan: Mapping[str, object],
    space_execution_contracts: list[Mapping[str, object]],
) -> None:
    """Validate that an offline Brain plan bundle is admissible, never executed."""
    validate_document("channel-intent", channel_intent)
    validate_document("brain-plan", brain_plan)
    for contract in space_execution_contracts:
        validate_document("space-execution-contract", contract)

    participating_spaces = brain_plan["participating_spaces"]
    tasks = brain_plan["tasks"]
    assert isinstance(participating_spaces, list) and isinstance(tasks, list)
    participating_space_ids = {
        space["space_id"]
        for space in participating_spaces
        if isinstance(space, Mapping) and isinstance(space["space_id"], str)
    }
    noncanonical_space_ids = participating_space_ids - CANONICAL_SPACE_IDS
    if noncanonical_space_ids:
        raise ContractValidationError(
            "Brain plan participating spaces must be canonical; "
            f"not canonical: {sorted(noncanonical_space_ids)!r}."
        )

    requested_space_id = channel_intent.get("requested_space_id")
    if requested_space_id is not None and requested_space_id not in participating_space_ids:
        raise ContractValidationError(
            "channel-intent requested_space_id must be a Brain plan participating space."
        )

    contract_space_ids = [contract["space_id"] for contract in space_execution_contracts]
    assert all(isinstance(space_id, str) for space_id in contract_space_ids)
    unparticipating_contract_space_ids = set(contract_space_ids) - participating_space_ids
    if unparticipating_contract_space_ids:
        raise ContractValidationError(
            "Brain plan admission bundle contains unparticipating SpaceExecutionContracts: "
            f"{sorted(unparticipating_contract_space_ids)!r}."
        )
    for space_id in participating_space_ids:
        if contract_space_ids.count(space_id) != 1:
            raise ContractValidationError(
                "Brain plan admission bundle requires exactly one "
                f"SpaceExecutionContract for participating space {space_id!r}."
            )

    covered_space_ids = set(contract_space_ids)
    for task in tasks:
        assert isinstance(task, Mapping)
        task_space_ids = task["space_ids"]
        evidence_requirements = task["evidence_requirements"]
        assert isinstance(task_space_ids, list) and isinstance(evidence_requirements, list)
        if not set(task_space_ids).issubset(participating_space_ids):
            raise ContractValidationError("Brain plan task spaces must be participating spaces.")
        if not set(task_space_ids).issubset(covered_space_ids):
            raise ContractValidationError(
                "Brain plan task spaces must be covered by SpaceExecutionContracts."
            )
        if not evidence_requirements:
            raise ContractValidationError(
                "Brain plan tasks must retain at least one evidence requirement."
            )


def validate_brain_plan_lifecycle_bundle(
    channel_intent: Mapping[str, object],
    brain_plan: Mapping[str, object],
    space_execution_contracts: list[Mapping[str, object]],
    lifecycle: Mapping[str, object],
) -> None:
    """Validate a persisted planning lifecycle after offline admission checks."""
    validate_brain_plan_admission_bundle(
        channel_intent, brain_plan, space_execution_contracts
    )
    validate_document("brain-plan-lifecycle", lifecycle)

    if lifecycle["correlation_id"] != channel_intent["correlation_id"]:
        raise ContractValidationError(
            "Brain plan lifecycle correlation_id must exactly match the admitted channel intent."
        )
    if lifecycle["plan_id"] != brain_plan["plan_id"]:
        raise ContractValidationError(
            "Brain plan lifecycle plan_id must exactly match the admitted Brain plan."
        )

    participating_spaces = brain_plan["participating_spaces"]
    lifecycle_space_ids = lifecycle["participating_space_ids"]
    assert isinstance(participating_spaces, list) and isinstance(lifecycle_space_ids, list)
    plan_space_ids = [space["space_id"] for space in participating_spaces]
    if lifecycle_space_ids != plan_space_ids:
        raise ContractValidationError(
            "Brain plan lifecycle participating_space_ids must exactly match the admitted Brain plan."
        )


def validate_brain_openfang_handoff_bundle(
    channel_intent: Mapping[str, object],
    brain_plan: Mapping[str, object],
    space_execution_contracts: list[Mapping[str, object]],
    lifecycle: Mapping[str, object],
    handoff: Mapping[str, object],
) -> None:
    """Validate a deferred Brain plan node handoff without execution authority."""
    validate_brain_plan_lifecycle_bundle(
        channel_intent, brain_plan, space_execution_contracts, lifecycle
    )
    validate_document("brain-openfang-handoff", handoff)

    if lifecycle["status"] != "execution_deferred":
        raise ContractValidationError(
            "Brain OpenFang handoff requires an execution_deferred Brain plan lifecycle."
        )
    for field_name in ("correlation_id", "plan_id"):
        if handoff[field_name] != lifecycle[field_name]:
            raise ContractValidationError(
                f"Brain OpenFang handoff {field_name} must match the deferred lifecycle."
            )
    if handoff["lifecycle_revision"] != lifecycle["revision"]:
        raise ContractValidationError(
            "Brain OpenFang handoff lifecycle_revision must match the deferred lifecycle revision."
        )

    tasks = brain_plan["tasks"]
    assert isinstance(tasks, list)
    selected_tasks = [
        task
        for task in tasks
        if isinstance(task, Mapping) and task["node_id"] == handoff["brain_task_node_id"]
    ]
    if len(selected_tasks) != 1:
        raise ContractValidationError(
            "Brain OpenFang handoff brain_task_node_id must select exactly one admitted Brain task."
        )
    selected_task_space_ids = selected_tasks[0]["space_ids"]
    assert isinstance(selected_task_space_ids, list)
    if handoff["space_id"] not in selected_task_space_ids:
        raise ContractValidationError(
            "Brain OpenFang handoff space_id must be declared by the selected Brain task."
        )

    participating_spaces = brain_plan["participating_spaces"]
    assert isinstance(participating_spaces, list)
    selected_spaces = [
        space
        for space in participating_spaces
        if isinstance(space, Mapping) and space["space_id"] == handoff["space_id"]
    ]
    if len(selected_spaces) != 1:
        raise ContractValidationError(
            "Brain OpenFang handoff space_id must select exactly one admitted canonical Space."
        )
    if handoff["roles"] != selected_spaces[0]["roles"]:
        raise ContractValidationError(
            "Brain OpenFang handoff roles must exactly match the admitted Space roles."
        )


def _reject_brain_plan_application_targets(document: Mapping[str, object]) -> None:
    tasks = document.get("tasks")
    if not isinstance(tasks, list):
        return
    for index, task in enumerate(tasks):
        if isinstance(task, Mapping) and "application_target" in task:
            raise ContractValidationError(
                "brain-plan application targets are forbidden; "
                f"found one at tasks.{index}."
            )


def _validate_brain_plan_graph(document: Mapping[str, object]) -> None:
    tasks = document["tasks"]
    participating_spaces = document["participating_spaces"]
    assert isinstance(tasks, list) and isinstance(participating_spaces, list)
    declared_space_ids: set[str] = set()
    for space in participating_spaces:
        assert isinstance(space, Mapping)
        space_id = space["space_id"]
        assert isinstance(space_id, str)
        if space_id in declared_space_ids:
            raise ContractValidationError(
                f"brain-plan contains duplicate participating space_id {space_id!r}."
            )
        declared_space_ids.add(space_id)

    dependencies: dict[str, list[str]] = {}
    node_orders: dict[str, int] = {}
    used_orders: set[int] = set()
    for task in tasks:
        assert isinstance(task, Mapping)
        node_id = task["node_id"]
        order = task["order"]
        space_ids = task["space_ids"]
        depends_on = task["depends_on"]
        assert isinstance(node_id, str) and isinstance(order, int)
        assert isinstance(space_ids, list) and isinstance(depends_on, list)
        if node_id in dependencies:
            raise ContractValidationError(
                f"brain-plan contains duplicate task node_id {node_id!r}."
            )
        if order in used_orders:
            raise ContractValidationError(
                f"brain-plan contains duplicate task order {order}."
            )
        for space_id in space_ids:
            assert isinstance(space_id, str)
            if space_id not in declared_space_ids:
                raise ContractValidationError(
                    f"brain-plan task {node_id!r} references undeclared space {space_id!r}."
                )
        dependencies[node_id] = [item for item in depends_on if isinstance(item, str)]
        node_orders[node_id] = order
        used_orders.add(order)

    for node_id, dependency_ids in dependencies.items():
        for dependency_id in dependency_ids:
            if dependency_id not in dependencies:
                raise ContractValidationError(
                    f"brain-plan task {node_id!r} depends on unknown task node {dependency_id!r}."
                )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ContractValidationError(
                f"brain-plan task graph contains a cycle at {node_id!r}."
            )
        if node_id in visited:
            return
        visiting.add(node_id)
        for dependency in dependencies[node_id]:
            visit(dependency)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in dependencies:
        visit(node_id)
    for node_id, dependency_ids in dependencies.items():
        for dependency_id in dependency_ids:
            if node_orders[dependency_id] >= node_orders[node_id]:
                raise ContractValidationError(
                    f"brain-plan dependency {dependency_id!r} must precede task {node_id!r}."
                )


def _validate_brain_plan_lifecycle(document: Mapping[str, object]) -> None:
    event_history = document["event_history"]
    assert isinstance(event_history, list)
    events = [event for event in event_history if isinstance(event, Mapping)]
    if len(events) != len(event_history):
        raise ContractValidationError("Brain plan lifecycle event history must contain objects.")
    if events[0]["status"] != "planned":
        raise ContractValidationError("Brain plan lifecycle must start in planned status.")

    allowed_transitions = {
        "planned": {"awaiting_clarification", "admitted", "cancelled"},
        "awaiting_clarification": {"planned", "cancelled"},
        "admitted": {"execution_deferred", "cancelled"},
        "execution_deferred": set(),
        "cancelled": set(),
    }
    terminal_statuses = {"execution_deferred", "cancelled"}
    previous_status: str | None = None
    previous_timestamp: datetime | None = None
    for expected_revision, event in enumerate(events, start=1):
        revision = event["revision"]
        status = event["status"]
        occurred_at = event["occurred_at"]
        assert isinstance(revision, int) and isinstance(status, str) and isinstance(occurred_at, str)
        if revision != expected_revision:
            raise ContractValidationError(
                "Brain plan lifecycle event revisions must be contiguous from revision 1."
            )
        timestamp = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ContractValidationError(
                "Brain plan lifecycle timestamps must be timezone-aware."
            )
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise ContractValidationError(
                "Brain plan lifecycle timestamps must be monotonically non-decreasing."
            )
        if previous_status is not None:
            if previous_status in terminal_statuses:
                raise ContractValidationError(
                    "Brain plan lifecycle terminal status cannot be reopened."
                )
            if status not in allowed_transitions[previous_status]:
                raise ContractValidationError(
                    f"Brain plan lifecycle transition {previous_status!r} -> {status!r} is forbidden."
                )
        previous_status = status
        previous_timestamp = timestamp

    final_event = events[-1]
    if document["revision"] != final_event["revision"]:
        raise ContractValidationError(
            "Brain plan lifecycle revision must match the final event revision."
        )
    if document["status"] != final_event["status"]:
        raise ContractValidationError(
            "Brain plan lifecycle final status must match the final event status."
        )
    if document["updated_at"] != final_event["occurred_at"]:
        raise ContractValidationError(
            "Brain plan lifecycle updated_at must match the final event occurred_at."
        )
