from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from importlib import resources
from pathlib import Path
import sys
import tomllib

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import vibemind_shared
from vibemind_shared.contracts import (
    ContractValidationError,
    validate_brain_openfang_handoff_bundle,
)


OUTER_COMMIT = "f0001e40f306745f60818a5bbd3065377dacd443"
SCHEMAS = {
    "channel-intent-v1.schema.json": "c2e839018f9b5ed98d84088ce49769fc2584ce6354b65945542a55161e006a10",
    "brain-plan-v1.schema.json": "488541303e51935a1b6d38d5fc9c666376bf8a743bd8dc26d046979b3162c632",
    "space-execution-contract-v1.schema.json": "2b0ad853d80840488b83e4f88946531f059f39789e9cc85b4debff7efd526351",
    "brain-plan-lifecycle-v1.schema.json": "efafbaa06b228e07b976d9ff293d244931af4cafdc694026d4c8dcbcdd66f1e3",
    "brain-openfang-handoff-v1.schema.json": "e69afd00058af8f101aa60886f8cd26e9d3bd04979e0f67e0d9ed2968d6d2239",
}


def valid_bundle() -> tuple[dict[str, object], dict[str, object], list[dict[str, object]], dict[str, object], dict[str, object]]:
    channel_intent: dict[str, object] = {
        "contract_version": "v1",
        "correlation_id": "event_v1_01J8Q3Z4R5T6V7W8X9Y0ABCDEF",
        "channel_kind": "desktop-chat",
        "actor_context": {"actor_id": "user:local-42", "locale": "de-DE"},
        "session_context": {"session_id": "session-20260731-01", "conversation_id": "desktop-chat-42"},
        "message": "Bitte fasse die Projektlage zusammen.",
        "received_at": "2026-07-31T10:00:00Z",
        "requested_space_id": "research",
        "user_facing_expectations": {"reply": "required", "evidence": "summary"},
    }
    brain_plan: dict[str, object] = {
        "contract_version": "v1",
        "plan_id": "plan_v1_01J8Q3Z4R5T6V7W8X9Y0ABCDEF",
        "intent": {"summary": "Prepare a verified cross-space research brief.", "context": ["The source material is already available offline."], "requested_by": "product-owner"},
        "participating_spaces": [
            {"space_id": "research", "roles": [{"role": "researcher", "required_agent_count": 1}]},
            {"space_id": "coding", "roles": [{"role": "developer", "required_agent_count": 1}]},
        ],
        "tasks": [
            {"node_id": "research-sources", "order": 1, "summary": "Extract source findings.", "space_ids": ["research"], "depends_on": [], "success_criteria": ["Findings are traceable to the supplied source material."], "evidence_requirements": [{"evidence_type": "artifact", "description": "A source finding artifact is available."}]},
            {"node_id": "write-brief", "order": 2, "summary": "Create the brief from the findings.", "space_ids": ["coding"], "depends_on": ["research-sources"], "success_criteria": ["The brief answers the stated intent."], "evidence_requirements": [{"evidence_type": "artifact", "description": "A reviewable brief artifact is available."}]},
        ],
    }
    contracts = [
        {"contract_version": "v1", "contract_id": "space_execution_contract_v1_01J8Q3Z4R5T6V7W8X9Y0ABCDEF", "space_id": space_id, "executor_id": "executor:brain-orchestrator", "approval_policy_ref": "approval-policy:standard", "cost_policy_ref": "cost-policy:bounded", "healthcheck_ref": "healthcheck:bubbles-structural", "golden_path_ref": "golden-path:bubbles-promote"}
        for space_id in ("research", "coding")
    ]
    lifecycle: dict[str, object] = {
        "contract_version": "v1", "correlation_id": channel_intent["correlation_id"], "plan_id": brain_plan["plan_id"], "participating_space_ids": ["research", "coding"], "revision": 3, "status": "execution_deferred", "updated_at": "2026-08-01T10:02:00Z",
        "event_history": [
            {"revision": 1, "status": "planned", "occurred_at": "2026-08-01T10:00:00Z"},
            {"revision": 2, "status": "admitted", "reason_code": "admission-validated", "occurred_at": "2026-08-01T10:01:00Z"},
            {"revision": 3, "status": "execution_deferred", "reason_code": "execution-engine-deferred", "occurred_at": "2026-08-01T10:02:00Z"},
        ],
    }
    handoff: dict[str, object] = {
        "contract_version": "v1", "execution_mode": "cognitive", "execution_boundary": "openfang", "correlation_id": channel_intent["correlation_id"], "plan_id": brain_plan["plan_id"], "lifecycle_revision": 3, "brain_task_node_id": "research-sources", "space_id": "research", "roles": [{"role": "researcher", "required_agent_count": 1}], "approval_ref": "approval:opaque-openfang-boundary", "cost_ref": "cost:opaque-openfang-boundary", "retry": {"classification": "transient", "max_attempts": 3},
    }
    return channel_intent, brain_plan, contracts, lifecycle, handoff


def test_accepts_complete_deferred_bundle() -> None:
    validate_brain_openfang_handoff_bundle(*valid_bundle())


def test_resources_match_manifest_and_outer_provenance() -> None:
    contracts = resources.files("vibemind_shared.contracts")
    manifest = json.loads(contracts.joinpath("schema-manifest-v1.json").read_text(encoding="utf-8"))
    assert manifest["outer_commit"] == OUTER_COMMIT
    assert manifest["schemas"] == SCHEMAS
    for filename, expected_digest in SCHEMAS.items():
        content = contracts.joinpath("schemas", filename).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected_digest


def test_runtime_metadata_declares_requests_for_public_package_import() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    dependencies = metadata["project"]["dependencies"]
    assert "requests>=2.31,<3" in dependencies


def test_project_metadata_version_matches_module_version() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert metadata["project"]["version"] == vibemind_shared.__version__


@pytest.mark.parametrize("field_name", ("correlation_id", "plan_id", "lifecycle_revision", "brain_task_node_id", "space_id", "roles"))
def test_rejects_handoff_continuity_drift(field_name: str) -> None:
    bundle = list(valid_bundle())
    handoff = bundle[4]
    assert isinstance(handoff, dict)
    mutations: dict[str, object] = {
        "correlation_id": "event_v1_01J8Q3Z4R5T6V7W8X9Y0ABCDEG",
        "plan_id": "plan_v1_01J8Q3Z4R5T6V7W8X9Y0ABCDEG",
        "lifecycle_revision": 2,
        "brain_task_node_id": "unknown-node",
        "space_id": "coding",
        "roles": [{"role": "researcher", "required_agent_count": 2}],
    }
    handoff[field_name] = mutations[field_name]
    with pytest.raises(ContractValidationError):
        validate_brain_openfang_handoff_bundle(*bundle)  # type: ignore[arg-type]


@pytest.mark.parametrize("field_name", ("runtime", "provider", "application", "success"))
def test_rejects_unknown_runtime_provider_and_application_fields(field_name: str) -> None:
    bundle = list(valid_bundle())
    handoff = bundle[4]
    assert isinstance(handoff, dict)
    handoff[field_name] = "forbidden"
    with pytest.raises(ContractValidationError):
        validate_brain_openfang_handoff_bundle(*bundle)  # type: ignore[arg-type]


@pytest.mark.parametrize(("field_name", "value"), (("execution_mode", "direct"), ("execution_boundary", "provider")))
def test_requires_cognitive_openfang_execution_contract(field_name: str, value: str) -> None:
    bundle = list(valid_bundle())
    handoff = bundle[4]
    assert isinstance(handoff, dict)
    handoff[field_name] = value
    with pytest.raises(ContractValidationError):
        validate_brain_openfang_handoff_bundle(*bundle)  # type: ignore[arg-type]


@pytest.mark.parametrize(("classification", "max_attempts"), (("none", 2), ("transient", 1), ("transient", 4)))
def test_rejects_invalid_retry_bounds(classification: str, max_attempts: int) -> None:
    bundle = list(valid_bundle())
    handoff = bundle[4]
    assert isinstance(handoff, dict)
    handoff["retry"] = {"classification": classification, "max_attempts": max_attempts}
    with pytest.raises(ContractValidationError):
        validate_brain_openfang_handoff_bundle(*bundle)  # type: ignore[arg-type]


def test_rejects_non_deferred_lifecycle_and_preserves_input() -> None:
    bundle = list(valid_bundle())
    lifecycle = bundle[3]
    assert isinstance(lifecycle, dict)
    before = deepcopy(lifecycle)
    lifecycle["event_history"] = lifecycle["event_history"][:2]  # type: ignore[index]
    lifecycle.update({"revision": 2, "status": "admitted", "updated_at": "2026-08-01T10:01:00Z"})
    with pytest.raises(ContractValidationError):
        validate_brain_openfang_handoff_bundle(*bundle)  # type: ignore[arg-type]
    assert before["status"] == "execution_deferred"
