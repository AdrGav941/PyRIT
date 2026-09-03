# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import builtins
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from pyrit.converter import Base64Converter
from pyrit.exceptions import EmptyResponseException
from pyrit.models import Message, MessagePiece, TargetCapabilities
from pyrit.prompt_normalizer import ConverterConfiguration, PromptNormalizer
from pyrit.prompt_target import GitHubCopilotTarget
from pyrit.prompt_target.common.target_configuration import TargetConfiguration
from pyrit.prompt_target.github_copilot_target import (
    _CopilotClientProtocol,
    _CopilotSessionProtocol,
)

pytestmark = pytest.mark.usefixtures("patch_central_database")


def test_init_uses_expected_defaults(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)

    assert target._endpoint == "github-copilot-sdk://local"
    assert target._model_name == ""
    assert target._github_token is None
    assert target._working_directory == tmp_path.resolve()
    assert target._response_timeout_seconds == 120.0
    assert target._max_requests_per_minute is None


def test_init_captures_current_directory(tmp_path: Path) -> None:
    with patch.object(Path, "cwd", return_value=tmp_path):
        target = GitHubCopilotTarget()

    assert target._working_directory == tmp_path.resolve()


@pytest.mark.parametrize("timeout", [0.0, -1.0])
def test_init_rejects_nonpositive_timeout(
    tmp_path: Path,
    timeout: float,
) -> None:
    with pytest.raises(ValueError, match="response_timeout_seconds must be positive"):
        GitHubCopilotTarget(
            working_directory=tmp_path,
            response_timeout_seconds=timeout,
        )


def test_init_rejects_missing_working_directory(tmp_path: Path) -> None:
    missing_directory = tmp_path / "missing"

    with pytest.raises(ValueError, match="working_directory does not exist"):
        GitHubCopilotTarget(working_directory=missing_directory)


def test_init_rejects_file_as_working_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("content", encoding="utf-8")

    with pytest.raises(ValueError, match="working_directory is not a directory"):
        GitHubCopilotTarget(working_directory=file_path)


def test_capabilities_match_stage_1_contract(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)

    assert target.capabilities == TargetCapabilities(
        supports_multi_turn=True,
        supports_multi_message_pieces=False,
        supports_json_schema=False,
        supports_json_output=False,
        supports_editable_history=False,
        supports_system_prompt=True,
        supports_streaming_audio=False,
        input_modalities=frozenset({frozenset({"text"})}),
        output_modalities=frozenset({frozenset({"text"})}),
    )
    assert target.configuration.pipeline.normalizers == ()


def test_custom_configuration_cannot_change_capabilities(tmp_path: Path) -> None:
    configuration = TargetConfiguration(capabilities=TargetCapabilities(supports_multi_turn=False))

    with pytest.raises(ValueError, match="must preserve GitHubCopilotTarget capabilities"):
        GitHubCopilotTarget(
            working_directory=tmp_path,
            custom_configuration=configuration,
        )


def test_custom_configuration_cannot_add_rewriting(tmp_path: Path) -> None:
    configuration = TargetConfiguration(capabilities=GitHubCopilotTarget._CAPABILITIES)

    with pytest.raises(ValueError, match="may not introduce request rewriting"):
        GitHubCopilotTarget(
            working_directory=tmp_path,
            custom_configuration=configuration,
        )


def test_identifier_includes_directory_but_excludes_secrets_and_timeout(
    tmp_path: Path,
) -> None:
    first = GitHubCopilotTarget(
        model_name="gpt-5",
        github_token="first-secret",
        working_directory=tmp_path,
        response_timeout_seconds=10.0,
    )
    second = GitHubCopilotTarget(
        model_name="gpt-5",
        github_token="second-secret",
        working_directory=tmp_path,
        response_timeout_seconds=20.0,
    )

    first_identifier = first.get_identifier()
    second_identifier = second.get_identifier()

    assert first_identifier.hash == second_identifier.hash
    assert first_identifier.params["working_directory"] == str(tmp_path.resolve())
    assert first_identifier.params["endpoint"] == "github-copilot-sdk://local"
    assert "github_token" not in first_identifier.params
    assert "response_timeout_seconds" not in first_identifier.params


def test_identifier_changes_with_working_directory(tmp_path: Path) -> None:
    other_directory = tmp_path / "other"
    other_directory.mkdir()

    first = GitHubCopilotTarget(working_directory=tmp_path)
    second = GitHubCopilotTarget(working_directory=other_directory)

    assert first.get_identifier().hash != second.get_identifier().hash


def _fake_client() -> MagicMock:
    client = MagicMock(spec=_CopilotClientProtocol)
    client.start = AsyncMock()
    client.stop = AsyncMock()
    client.create_session = AsyncMock()
    client.get_auth_status = AsyncMock(
        return_value=SimpleNamespace(
            isAuthenticated=True,
            authType="oauth",
            host="github.com",
            login="test-user",
        )
    )
    return client


def _event(
    event_type: str,
    *,
    data: SimpleNamespace | None = None,
    agent_id: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        type=event_type,
        id=uuid4(),
        agent_id=agent_id,
        data=data or SimpleNamespace(),
    )


def _successful_events(*, content: str = "response") -> list[SimpleNamespace]:
    return [
        _event("user.message"),
        _event("assistant.turn_start", data=SimpleNamespace(model="gpt-test")),
        _event(
            "assistant.usage",
            data=SimpleNamespace(
                model="gpt-test",
                finish_reason="stop",
                input_tokens=4,
                output_tokens=2,
                reasoning_tokens=1,
            ),
        ),
        _event(
            "assistant.message",
            data=SimpleNamespace(
                content=content,
                model="gpt-test",
                tool_requests=None,
                server_tools=None,
            ),
        ),
        _event("assistant.turn_end", data=SimpleNamespace(model="gpt-test")),
        _event("session.idle"),
    ]


def _fake_session(*, events: list[SimpleNamespace]) -> MagicMock:
    session = MagicMock(spec=_CopilotSessionProtocol)
    session.session_id = "sdk-session"
    handlers: list[object] = []

    def subscribe(handler: object) -> object:
        handlers.append(handler)
        return MagicMock()

    async def send_async(prompt: str) -> str:
        for event in events:
            for handler in handlers:
                handler(event)  # type: ignore[operator]
        return "request-id"

    session.on = MagicMock(side_effect=subscribe)
    session.send = AsyncMock(side_effect=send_async)
    session.abort = AsyncMock()
    session.disconnect = AsyncMock()
    return session


def test_init_rejects_python_before_311(tmp_path: Path) -> None:
    with (
        patch.object(sys, "version_info", (3, 10, 0)),
        pytest.raises(RuntimeError, match="requires Python 3.11"),
    ):
        GitHubCopilotTarget(working_directory=tmp_path)


def test_client_factory_uses_explicit_token_without_ambient_fallback(tmp_path: Path) -> None:
    client_type = MagicMock()
    module = ModuleType("copilot")
    module.CopilotClient = client_type
    target = GitHubCopilotTarget(working_directory=tmp_path, github_token="secret")

    with patch.dict(sys.modules, {"copilot": module}):
        target._create_client()

    assert client_type.call_args.kwargs == {
        "working_directory": str(tmp_path.resolve()),
        "github_token": "secret",
        "use_logged_in_user": False,
        "enable_remote_sessions": False,
        "mode": "copilot-cli",
    }


def test_client_factory_preserves_default_auth_discovery(tmp_path: Path) -> None:
    client_type = MagicMock()
    module = ModuleType("copilot")
    module.CopilotClient = client_type
    target = GitHubCopilotTarget(working_directory=tmp_path)

    with patch.dict(sys.modules, {"copilot": module}):
        target._create_client()

    assert client_type.call_args.kwargs["github_token"] is None
    assert client_type.call_args.kwargs["use_logged_in_user"] is None


def test_client_factory_reports_missing_optional_dependency(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    original_import = builtins.__import__

    def reject_copilot_import(
        name: str,
        _globals: dict[str, object] | None = None,
        _locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "copilot":
            raise ModuleNotFoundError(name="copilot")
        return original_import(name, _globals, _locals, fromlist, level)

    with (
        patch.object(builtins, "__import__", side_effect=reject_copilot_import),
        pytest.raises(ModuleNotFoundError, match="github-copilot.*extra"),
    ):
        target._create_client()


def test_client_factory_propagates_sdk_internal_import_failure(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    original_import = builtins.__import__

    def reject_internal_import(
        name: str,
        _globals: dict[str, object] | None = None,
        _locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "copilot":
            raise ModuleNotFoundError("internal dependency missing", name="copilot_internal")
        return original_import(name, _globals, _locals, fromlist, level)

    with (
        patch.object(builtins, "__import__", side_effect=reject_internal_import),
        pytest.raises(ModuleNotFoundError, match="internal dependency missing"),
    ):
        target._create_client()


async def test_ensure_client_starts_once_for_concurrent_callers(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    client = _fake_client()

    with patch.object(target, "_create_client", return_value=client):
        first, second = await asyncio.gather(target._ensure_client_async(), target._ensure_client_async())

    assert first is client
    assert second is client
    client.start.assert_awaited_once()
    client.get_auth_status.assert_awaited_once()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("isAuthenticated", False),
        ("authType", None),
        ("host", None),
        ("login", None),
    ],
)
async def test_ensure_client_requires_stable_authenticated_identity(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    client = _fake_client()
    setattr(client.get_auth_status.return_value, field, value)

    with (
        patch.object(target, "_create_client", return_value=client),
        pytest.raises(RuntimeError, match="authentication is required|stable authenticated"),
    ):
        await target._ensure_client_async()

    client.stop.assert_awaited_once()
    assert target._client is None


async def test_startup_waiter_cancellation_does_not_cancel_shared_startup(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    client = _fake_client()
    started = asyncio.Event()
    release = asyncio.Event()

    async def start_async() -> None:
        started.set()
        await release.wait()

    client.start.side_effect = start_async
    with patch.object(target, "_create_client", return_value=client):
        cancelled_waiter = asyncio.create_task(target._ensure_client_async())
        await started.wait()
        surviving_waiter = asyncio.create_task(target._ensure_client_async())
        cancelled_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_waiter
        release.set()
        result = await surviving_waiter

    assert result is client
    client.start.assert_awaited_once()
    client.stop.assert_not_awaited()


async def test_cleanup_during_startup_cancels_and_stops_client(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    client = _fake_client()
    started = asyncio.Event()

    async def start_async() -> None:
        started.set()
        await asyncio.Event().wait()

    client.start.side_effect = start_async
    with patch.object(target, "_create_client", return_value=client):
        ensure_task = asyncio.create_task(target._ensure_client_async())
        await started.wait()
        await target.cleanup_target_async()

    with pytest.raises(asyncio.CancelledError):
        await ensure_task
    client.stop.assert_awaited_once()
    with pytest.raises(RuntimeError, match="closing or closed"):
        await target._ensure_client_async()


async def test_cleanup_reuses_in_progress_startup_stop(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    client = _fake_client()
    stop_started = asyncio.Event()
    release_stop = asyncio.Event()

    client.start.side_effect = RuntimeError("startup failed")

    async def stop_async() -> None:
        stop_started.set()
        await release_stop.wait()

    client.stop.side_effect = stop_async
    with patch.object(target, "_create_client", return_value=client):
        ensure_task = asyncio.create_task(target._ensure_client_async())
        await stop_started.wait()
        cleanup_task = asyncio.create_task(target.cleanup_target_async())
        await asyncio.sleep(0)
        release_stop.set()
        await cleanup_task

    with pytest.raises(asyncio.CancelledError):
        await ensure_task
    client.stop.assert_awaited_once()


async def test_failed_startup_can_be_retried(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    failed_client = _fake_client()
    failed_client.start.side_effect = RuntimeError("startup failed")
    successful_client = _fake_client()

    with patch.object(target, "_create_client", side_effect=[failed_client, successful_client]):
        with pytest.raises(RuntimeError, match="startup failed"):
            await target._ensure_client_async()
        result = await target._ensure_client_async()

    assert result is successful_client
    failed_client.stop.assert_awaited_once()


def test_session_id_is_deterministic_and_identity_scoped() -> None:
    first = GitHubCopilotTarget._derive_session_id(
        conversation_id="conversation",
        target_identifier_hash="target",
        account_identity_hash="account",
    )
    repeated = GitHubCopilotTarget._derive_session_id(
        conversation_id="conversation",
        target_identifier_hash="target",
        account_identity_hash="account",
    )
    other_conversation = GitHubCopilotTarget._derive_session_id(
        conversation_id="other",
        target_identifier_hash="target",
        account_identity_hash="account",
    )

    assert first == repeated
    assert first != other_conversation


async def test_conversation_reuses_one_deterministic_session(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    client = _fake_client()
    session = MagicMock(spec=_CopilotSessionProtocol)
    session.session_id = "sdk-session"
    client.create_session.return_value = session

    with patch.object(target, "_create_client", return_value=client):
        state = target._admit_turn(conversation_id="conversation")
        first = await target._get_or_create_session_async(
            conversation_id="conversation",
            state=state,
        )
        second = await target._get_or_create_session_async(
            conversation_id="conversation",
            state=state,
        )

    assert first is session
    assert second is session
    client.create_session.assert_awaited_once()
    create_kwargs = client.create_session.await_args.kwargs
    assert create_kwargs["session_id"] == state.sdk_session_id
    assert create_kwargs["available_tools"] == []
    assert create_kwargs["model"] is None


def test_overlapping_admission_desynchronizes_conversation(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    state = target._admit_turn(conversation_id="conversation")

    with pytest.raises(RuntimeError, match="Overlapping sends"):
        target._admit_turn(conversation_id="conversation")

    assert state.lifecycle == "desynchronized"
    target._finish_turn(state=state, desynchronized=False)
    assert state.lifecycle == "desynchronized"

    with pytest.raises(RuntimeError, match="desynchronized"):
        target._admit_turn(conversation_id="conversation")


def test_separate_conversations_admit_independently(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)

    first = target._admit_turn(conversation_id="first")
    second = target._admit_turn(conversation_id="second")

    assert first is not second
    assert first.lifecycle == "in_flight"
    assert second.lifecycle == "in_flight"


def test_set_model_name_invalidates_identifier_before_session_creation(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(model_name="first", working_directory=tmp_path)
    first = target.get_identifier()

    target.set_model_name(model_name="second")

    second = target.get_identifier()
    assert second.hash != first.hash
    assert second.params["model_name"] == "second"


async def test_set_model_name_rejected_after_session_creation_starts(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    client = _fake_client()
    session = MagicMock(spec=_CopilotSessionProtocol)
    session.session_id = "session"
    client.create_session.return_value = session

    with patch.object(target, "_create_client", return_value=client):
        state = target._admit_turn(conversation_id="conversation")
        await target._get_or_create_session_async(
            conversation_id="conversation",
            state=state,
        )

    with pytest.raises(RuntimeError, match="cannot change"):
        target.set_model_name(model_name="other")


def test_set_system_prompt_persists_exactly_once(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)

    target.set_system_prompt(system_prompt="Keep this exact.", conversation_id="conversation")

    messages = target._memory.get_conversation_messages(conversation_id="conversation")
    assert len(messages) == 1
    assert messages[0].api_role == "system"
    assert messages[0].get_value() == "Keep this exact."
    state = target._conversations["conversation"]
    assert state.system_prompt == "Keep this exact."
    assert state.accepted_message_count == 1

    with pytest.raises(RuntimeError, match="exactly once"):
        target.set_system_prompt(system_prompt="Again", conversation_id="conversation")
    with pytest.raises(RuntimeError, match="cannot change"):
        target.set_model_name(model_name="other")


@pytest.mark.parametrize("system_prompt", ["", "   "])
def test_set_system_prompt_rejects_empty(tmp_path: Path, system_prompt: str) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)

    with pytest.raises(ValueError, match="non-empty"):
        target.set_system_prompt(system_prompt=system_prompt, conversation_id="conversation")


async def test_session_creation_appends_system_prompt_once(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    target.set_system_prompt(system_prompt="Initial instructions", conversation_id="conversation")
    client = _fake_client()
    session = MagicMock(spec=_CopilotSessionProtocol)
    session.session_id = "session"
    client.create_session.return_value = session

    with patch.object(target, "_create_client", return_value=client):
        state = target._admit_turn(conversation_id="conversation")
        first = await target._get_or_create_session_async(
            conversation_id="conversation",
            state=state,
        )
        second = await target._get_or_create_session_async(
            conversation_id="conversation",
            state=state,
        )

    assert first is second
    client.create_session.assert_awaited_once()
    assert client.create_session.await_args.kwargs["system_message"] == {
        "mode": "append",
        "content": "Initial instructions",
    }


def test_configuration_mutation_rejected_after_cleanup(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    target._lifecycle = "closed"

    with pytest.raises(RuntimeError, match="closing or closed"):
        target.set_model_name(model_name="other")
    with pytest.raises(RuntimeError, match="closing or closed"):
        target.set_system_prompt(system_prompt="Prompt", conversation_id="conversation")


def _request(*, conversation_id: str = "conversation", text: str = "request") -> Message:
    return Message(
        message_pieces=[
            MessagePiece(
                role="user",
                original_value=text,
                conversation_id=conversation_id,
            )
        ]
    )


async def test_send_dispatches_exact_text_once_and_maps_response(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    client = _fake_client()
    session = _fake_session(events=_successful_events(content="final answer"))
    client.create_session.return_value = session

    with patch.object(target, "_create_client", return_value=client):
        responses = await target.send_prompt_async(message=_request(text="exact request"))

    session.send.assert_awaited_once_with("exact request")
    assert len(responses) == 1
    response_piece = responses[0].message_pieces[0]
    assert response_piece.original_value == "final answer"
    assert response_piece.role == "assistant"
    assert response_piece.prompt_metadata["finish_reason"] == "stop"
    assert response_piece.prompt_metadata["token_usage_input_tokens"] == 4
    assert response_piece.prompt_metadata["token_usage_output_tokens"] == 2
    assert response_piece.prompt_metadata["token_usage_reasoning_tokens"] == 1
    assert response_piece.prompt_metadata["github_copilot_effective_model"] == "gpt-test"
    assert response_piece.prompt_metadata["github_copilot_session_id"]
    assert response_piece.prompt_metadata["github_copilot_message_event_id"]
    assert response_piece.prompt_metadata["github_copilot_history_digest"]
    assert response_piece.prompt_metadata["github_copilot_history_piece_count"] == 2


@pytest.mark.parametrize(
    ("piece", "match"),
    [
        (MessagePiece(role="assistant", original_value="text", conversation_id="conversation"), "user-role"),
        (
            MessagePiece(
                role="user",
                original_value="text",
                original_value_data_type="image_path",
                conversation_id="conversation",
            ),
            "only the following data types",
        ),
        (
            MessagePiece(
                role="user",
                original_value="text",
                conversation_id="conversation",
                prompt_metadata={"response_format": "json"},
            ),
            "does not support JSON",
        ),
        (
            MessagePiece(
                role="user",
                original_value="text",
                conversation_id="conversation",
                prompt_metadata={"github_copilot_session_id": "reserved"},
            ),
            "reserved GitHub Copilot",
        ),
    ],
)
async def test_send_rejects_unsupported_request_before_dispatch(
    tmp_path: Path,
    piece: MessagePiece,
    match: str,
) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    client = _fake_client()

    with (
        patch.object(target, "_create_client", return_value=client),
        pytest.raises(ValueError, match=match),
    ):
        await target.send_prompt_async(message=piece.to_message())

    client.create_session.assert_not_awaited()


@pytest.mark.parametrize(
    "event",
    [
        _event("tool.execution_start"),
        _event("permission.requested"),
        _event("skill.invoked"),
        _event("subagent.started"),
        _event(
            "assistant.message",
            data=SimpleNamespace(
                content="agent output",
                model="gpt-test",
                tool_requests=None,
                server_tools=None,
            ),
            agent_id="child",
        ),
        _event(
            "assistant.message",
            data=SimpleNamespace(
                content="tool output",
                model="gpt-test",
                tool_requests=[object()],
                server_tools=None,
            ),
        ),
    ],
)
async def test_agentic_event_desynchronizes_conversation(
    tmp_path: Path,
    event: SimpleNamespace,
) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    client = _fake_client()
    session = _fake_session(events=[event])
    client.create_session.return_value = session

    with (
        patch.object(target, "_create_client", return_value=client),
        pytest.raises(RuntimeError, match="contract violation"),
    ):
        await target.send_prompt_async(message=_request())

    session.send.assert_awaited_once()
    session.abort.assert_awaited_once()
    assert target._conversations["conversation"].lifecycle == "desynchronized"


async def test_empty_response_desynchronizes_conversation(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    client = _fake_client()
    session = _fake_session(events=[_event("session.idle")])
    client.create_session.return_value = session

    with (
        patch.object(target, "_create_client", return_value=client),
        pytest.raises(EmptyResponseException, match="empty response"),
    ):
        await target.send_prompt_async(message=_request())

    assert target._conversations["conversation"].lifecycle == "desynchronized"


async def test_timeout_aborts_and_desynchronizes_conversation(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path, response_timeout_seconds=0.01)
    client = _fake_client()
    session = _fake_session(events=[])
    client.create_session.return_value = session

    with (
        patch.object(target, "_create_client", return_value=client),
        pytest.raises(TimeoutError, match="did not become idle"),
    ):
        await target.send_prompt_async(message=_request())

    session.abort.assert_awaited_once()
    assert target._conversations["conversation"].lifecycle == "desynchronized"


async def test_overlap_rejects_second_send_without_dispatch_and_stays_desynchronized(
    tmp_path: Path,
) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    client = _fake_client()
    release = asyncio.Event()
    session = _fake_session(events=[])
    handlers: list[object] = []
    session.on.side_effect = lambda handler: handlers.append(handler) or MagicMock()

    async def send_async(prompt: str) -> str:
        await release.wait()
        for event in _successful_events():
            for handler in handlers:
                handler(event)  # type: ignore[operator]
        return "request-id"

    session.send.side_effect = send_async
    client.create_session.return_value = session

    with patch.object(target, "_create_client", return_value=client):
        active = asyncio.create_task(target.send_prompt_async(message=_request(text="A")))
        while session.send.await_count == 0:
            await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="Overlapping sends"):
            await target.send_prompt_async(message=_request(text="B"))
        assert session.send.await_count == 1
        release.set()
        await active

    state = target._conversations["conversation"]
    assert state.lifecycle == "desynchronized"
    with pytest.raises(RuntimeError, match="desynchronized"):
        await target.send_prompt_async(message=_request(text="C"))
    assert session.send.await_count == 1


async def test_overlap_during_session_creation_allows_active_turn_to_finish(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    client = _fake_client()
    session = _fake_session(events=_successful_events())
    release_creation = asyncio.Event()

    async def create_session_async(**kwargs: object) -> MagicMock:
        await release_creation.wait()
        return session

    client.create_session.side_effect = create_session_async
    with patch.object(target, "_create_client", return_value=client):
        active = asyncio.create_task(target.send_prompt_async(message=_request(text="A")))
        while client.create_session.await_count == 0:
            await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="Overlapping sends"):
            await target.send_prompt_async(message=_request(text="B"))
        release_creation.set()
        response = await active

    assert response[0].get_value() == "response"
    session.send.assert_awaited_once_with("A")
    assert target._conversations["conversation"].lifecycle == "desynchronized"


async def test_reset_disconnects_without_making_conversation_sendable(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    client = _fake_client()
    session = _fake_session(events=_successful_events())
    client.create_session.return_value = session

    with patch.object(target, "_create_client", return_value=client):
        await target.send_prompt_async(message=_request())
        await target.reset_conversation_async(conversation_id="conversation")

    session.disconnect.assert_awaited_once()
    assert target._conversations["conversation"].lifecycle == "disconnected"
    with pytest.raises(RuntimeError, match="disconnected"):
        await target.send_prompt_async(message=_request(text="after reset"))


async def test_cleanup_disconnects_sessions_stops_client_and_closes_target(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    client = _fake_client()
    session = _fake_session(events=_successful_events())
    client.create_session.return_value = session

    with patch.object(target, "_create_client", return_value=client):
        await target.send_prompt_async(message=_request())
        await target.cleanup_target_async()
        await target.cleanup_target_async()

    session.disconnect.assert_awaited_once()
    client.stop.assert_awaited_once()
    with pytest.raises(RuntimeError, match="closing or closed"):
        await target.send_prompt_async(message=_request(text="after cleanup"))


async def test_cleanup_drains_active_turn_before_disconnect(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    client = _fake_client()
    release = asyncio.Event()
    session = _fake_session(events=[])
    handlers: list[object] = []
    session.on.side_effect = lambda handler: handlers.append(handler) or MagicMock()

    async def send_async(prompt: str) -> str:
        await release.wait()
        for event in _successful_events():
            for handler in handlers:
                handler(event)  # type: ignore[operator]
        return "request-id"

    session.send.side_effect = send_async
    client.create_session.return_value = session

    with patch.object(target, "_create_client", return_value=client):
        active = asyncio.create_task(target.send_prompt_async(message=_request()))
        while session.send.await_count == 0:
            await asyncio.sleep(0)
        cleanup = asyncio.create_task(target.cleanup_target_async())
        await asyncio.sleep(0)
        assert not cleanup.done()
        session.disconnect.assert_not_awaited()
        release.set()
        await active
        await cleanup

    session.disconnect.assert_awaited_once()


async def test_cleanup_aborts_timed_out_active_turn(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path, response_timeout_seconds=0.01)
    client = _fake_client()
    release = asyncio.Event()
    session = _fake_session(events=[])

    async def send_async(prompt: str) -> str:
        await release.wait()
        raise RuntimeError("stopped")

    async def abort_async() -> None:
        release.set()

    session.send.side_effect = send_async
    session.abort.side_effect = abort_async
    client.create_session.return_value = session

    with patch.object(target, "_create_client", return_value=client):
        active = asyncio.create_task(target.send_prompt_async(message=_request()))
        while session.send.await_count == 0:
            await asyncio.sleep(0)
        await target.cleanup_target_async()

    with pytest.raises(TimeoutError, match="did not become idle"):
        await active
    session.abort.assert_awaited()
    session.disconnect.assert_awaited_once()


async def test_cancellation_after_dispatch_aborts_and_desynchronizes(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    client = _fake_client()
    release = asyncio.Event()
    session = _fake_session(events=[])

    async def send_async(prompt: str) -> str:
        await release.wait()
        return "request-id"

    async def abort_async() -> None:
        release.set()

    session.send.side_effect = send_async
    session.abort.side_effect = abort_async
    client.create_session.return_value = session

    with patch.object(target, "_create_client", return_value=client):
        active = asyncio.create_task(target.send_prompt_async(message=_request()))
        while session.send.await_count == 0:
            await asyncio.sleep(0)
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active

    session.abort.assert_awaited_once()
    assert target._conversations["conversation"].lifecycle == "desynchronized"


async def test_prompt_normalizer_sequential_turns_reuse_native_session(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    normalizer = PromptNormalizer()
    client = _fake_client()
    session = _fake_session(events=[])
    handlers: list[object] = []
    session.on.side_effect = lambda handler: handlers.append(handler) or MagicMock()
    responses = iter(["first response", "second response"])

    async def send_async(prompt: str) -> str:
        for event in _successful_events(content=next(responses)):
            for handler in handlers:
                handler(event)  # type: ignore[operator]
        return "request-id"

    session.send.side_effect = send_async
    client.create_session.return_value = session

    with patch.object(target, "_create_client", return_value=client):
        first = await normalizer.send_prompt_async(
            message=_request(text="first"),
            target=target,
            conversation_id="conversation",
        )
        second = await normalizer.send_prompt_async(
            message=_request(text="second"),
            target=target,
            conversation_id="conversation",
        )

    assert first.get_value() == "first response"
    assert second.get_value() == "second response"
    client.create_session.assert_awaited_once()
    assert [call.args[0] for call in session.send.await_args_list] == ["first", "second"]
    assert len(target._memory.get_conversation_messages(conversation_id="conversation")) == 4


async def test_separate_conversations_dispatch_concurrently(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    client = _fake_client()
    both_started = asyncio.Event()
    release = asyncio.Event()
    started = 0

    def build_session(conversation: str) -> MagicMock:
        session = _fake_session(events=[])
        handlers: list[object] = []
        session.on.side_effect = lambda handler: handlers.append(handler) or MagicMock()

        async def send_async(prompt: str) -> str:
            nonlocal started
            started += 1
            if started == 2:
                both_started.set()
            await release.wait()
            for event in _successful_events(content=f"{conversation} response"):
                for handler in handlers:
                    handler(event)  # type: ignore[operator]
            return "request-id"

        session.send.side_effect = send_async
        return session

    sessions = [build_session("first"), build_session("second")]
    client.create_session.side_effect = sessions

    with patch.object(target, "_create_client", return_value=client):
        first = asyncio.create_task(target.send_prompt_async(message=_request(conversation_id="first")))
        second = asyncio.create_task(target.send_prompt_async(message=_request(conversation_id="second")))
        await asyncio.wait_for(both_started.wait(), timeout=1.0)
        release.set()
        first_response, second_response = await asyncio.gather(first, second)

    assert first_response[0].get_value() == "first response"
    assert second_response[0].get_value() == "second response"
    assert client.create_session.await_count == 2


async def test_reset_during_session_creation_prevents_dispatch(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path, response_timeout_seconds=1.0)
    client = _fake_client()
    session = _fake_session(events=_successful_events())
    release_creation = asyncio.Event()

    async def create_session_async(**kwargs: object) -> MagicMock:
        await release_creation.wait()
        return session

    client.create_session.side_effect = create_session_async
    with patch.object(target, "_create_client", return_value=client):
        active = asyncio.create_task(target.send_prompt_async(message=_request()))
        while client.create_session.await_count == 0:
            await asyncio.sleep(0)
        reset = asyncio.create_task(target.reset_conversation_async(conversation_id="conversation"))
        await asyncio.sleep(0)
        release_creation.set()
        await reset
        with pytest.raises(RuntimeError, match="completed after the conversation closed"):
            await active

    session.send.assert_not_awaited()
    session.disconnect.assert_awaited_once()


async def test_cleanup_during_session_creation_prevents_dispatch(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path, response_timeout_seconds=1.0)
    client = _fake_client()
    session = _fake_session(events=_successful_events())
    release_creation = asyncio.Event()

    async def create_session_async(**kwargs: object) -> MagicMock:
        await release_creation.wait()
        return session

    client.create_session.side_effect = create_session_async
    with patch.object(target, "_create_client", return_value=client):
        active = asyncio.create_task(target.send_prompt_async(message=_request()))
        while client.create_session.await_count == 0:
            await asyncio.sleep(0)
        cleanup = asyncio.create_task(target.cleanup_target_async())
        await asyncio.sleep(0)
        release_creation.set()
        await cleanup
        with pytest.raises(RuntimeError, match="completed after the conversation closed"):
            await active

    session.send.assert_not_awaited()
    session.disconnect.assert_awaited_once()
    client.stop.assert_awaited_once()


async def test_timeout_bounds_hanging_abort(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path, response_timeout_seconds=0.01)
    client = _fake_client()
    session = _fake_session(events=[])

    async def abort_async() -> None:
        await asyncio.Event().wait()

    session.abort.side_effect = abort_async
    client.create_session.return_value = session

    started = time.monotonic()
    with (
        patch.object(target, "_create_client", return_value=client),
        pytest.raises(TimeoutError, match="abort confirmation failed"),
    ):
        await target.send_prompt_async(message=_request())
    elapsed = time.monotonic() - started

    assert elapsed < 0.2
    session.abort.assert_awaited_once()


async def test_live_tail_mismatch_fails_before_second_dispatch(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    client = _fake_client()
    session = _fake_session(events=_successful_events())
    client.create_session.return_value = session

    with patch.object(target, "_create_client", return_value=client):
        await target.send_prompt_async(message=_request(text="first"))
        with pytest.raises(RuntimeError, match="live PyRIT conversation tail"):
            await target.send_prompt_async(message=_request(text="not persisted"))

    assert session.send.await_count == 1
    assert target._conversations["conversation"].lifecycle == "desynchronized"


async def test_reset_cancels_registered_dispatch_before_it_can_continue(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path, response_timeout_seconds=1.0)
    client = _fake_client()
    session = _fake_session(events=[])
    dispatch_entered = asyncio.Event()
    dispatch_continued = False

    async def send_async(prompt: str) -> str:
        nonlocal dispatch_continued
        dispatch_entered.set()
        await asyncio.sleep(0)
        dispatch_continued = True
        return "request-id"

    session.send.side_effect = send_async
    client.create_session.return_value = session

    with patch.object(target, "_create_client", return_value=client):
        active = asyncio.create_task(target.send_prompt_async(message=_request()))
        await dispatch_entered.wait()
        await target.reset_conversation_async(conversation_id="conversation")
        with pytest.raises(asyncio.CancelledError):
            await active

    assert not dispatch_continued
    assert target._conversations["conversation"].lifecycle == "disconnected"


async def test_timeout_is_hard_bound_when_send_suppresses_cancellation(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path, response_timeout_seconds=0.01)
    client = _fake_client()
    session = _fake_session(events=[])
    suppressed = asyncio.Event()

    async def send_async(prompt: str) -> str:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            suppressed.set()
            await asyncio.sleep(0.2)
            return "late-request-id"
        raise AssertionError("unreachable")

    session.send.side_effect = send_async
    client.create_session.return_value = session

    started = time.monotonic()
    with (
        patch.object(target, "_create_client", return_value=client),
        pytest.raises(TimeoutError, match="did not become idle"),
    ):
        await target.send_prompt_async(message=_request())
    elapsed = time.monotonic() - started
    await suppressed.wait()

    assert elapsed < 0.1
    dispatch_task = target._conversations["conversation"].dispatch_task
    assert dispatch_task is not None
    await target.cleanup_target_async()
    assert dispatch_task.done()


async def test_system_prompt_rejected_after_first_turn_admission(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    client = _fake_client()
    startup_entered = asyncio.Event()
    release_startup = asyncio.Event()

    async def start_async() -> None:
        startup_entered.set()
        await release_startup.wait()

    client.start.side_effect = start_async
    client.create_session.return_value = _fake_session(events=_successful_events())

    with patch.object(target, "_create_client", return_value=client):
        active = asyncio.create_task(target.send_prompt_async(message=_request()))
        await startup_entered.wait()
        with pytest.raises(RuntimeError, match="before the first user turn"):
            target.set_system_prompt(system_prompt="too late", conversation_id="conversation")
        release_startup.set()
        await active


async def test_public_send_redacts_explicit_token_from_startup_failure(tmp_path: Path) -> None:
    token = "top-secret-token"
    target = GitHubCopilotTarget(working_directory=tmp_path, github_token=token)
    client = _fake_client()
    client.start.side_effect = RuntimeError(f"credential rejected: {token}")

    with (
        patch.object(target, "_create_client", return_value=client),
        pytest.raises(RuntimeError, match="without exposing credential details") as exc_info,
    ):
        await target.send_prompt_async(message=_request())

    assert token not in str(exc_info.value)


async def test_session_creation_failure_desynchronizes_without_dispatch(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    client = _fake_client()
    client.create_session.side_effect = RuntimeError("ambiguous create failure")

    with (
        patch.object(target, "_create_client", return_value=client),
        pytest.raises(RuntimeError, match="ambiguous create failure"),
    ):
        await target.send_prompt_async(message=_request())

    assert target._conversations["conversation"].lifecycle == "desynchronized"
    client.create_session.assert_awaited_once()


async def test_send_failure_is_not_retried(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    client = _fake_client()
    session = _fake_session(events=[])
    session.send.side_effect = RuntimeError("ambiguous send failure")
    client.create_session.return_value = session

    with (
        patch.object(target, "_create_client", return_value=client),
        pytest.raises(RuntimeError, match="ambiguous send failure"),
    ):
        await target.send_prompt_async(message=_request())

    session.send.assert_awaited_once()
    assert target._conversations["conversation"].lifecycle == "desynchronized"


async def test_harmless_initialization_events_are_allowed(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    client = _fake_client()
    events = [
        _event("session.skills_loaded"),
        _event("session.tools_updated"),
        _event("hook.start"),
        _event("hook.end"),
        *_successful_events(),
    ]
    client.create_session.return_value = _fake_session(events=events)

    with patch.object(target, "_create_client", return_value=client):
        response = await target.send_prompt_async(message=_request())

    assert response[0].get_value() == "response"


async def test_out_of_order_events_desynchronize_conversation(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    client = _fake_client()
    events = [
        _event(
            "assistant.message",
            data=SimpleNamespace(
                content="response",
                model="gpt-test",
                tool_requests=None,
                server_tools=None,
            ),
        ),
        _event("user.message"),
        _event("session.idle"),
    ]
    client.create_session.return_value = _fake_session(events=events)

    with (
        patch.object(target, "_create_client", return_value=client),
        pytest.raises(RuntimeError, match="out of order"),
    ):
        await target.send_prompt_async(message=_request())

    assert target._conversations["conversation"].lifecycle == "desynchronized"


async def test_system_prompts_are_isolated_between_conversations(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    target.set_system_prompt(system_prompt="first prompt", conversation_id="first")
    target.set_system_prompt(system_prompt="second prompt", conversation_id="second")
    client = _fake_client()
    client.create_session.side_effect = [
        _fake_session(events=_successful_events(content="first")),
        _fake_session(events=_successful_events(content="second")),
    ]

    with patch.object(target, "_create_client", return_value=client):
        await target.send_prompt_async(message=_request(conversation_id="first"))
        await target.send_prompt_async(message=_request(conversation_id="second"))

    first_call, second_call = client.create_session.await_args_list
    assert first_call.kwargs["system_message"]["content"] == "first prompt"
    assert second_call.kwargs["system_message"]["content"] == "second prompt"


async def test_reset_preserves_pyrit_memory(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    normalizer = PromptNormalizer()
    client = _fake_client()
    client.create_session.return_value = _fake_session(events=_successful_events())

    with patch.object(target, "_create_client", return_value=client):
        await normalizer.send_prompt_async(
            message=_request(),
            target=target,
            conversation_id="conversation",
        )
        before = list(target._memory.get_conversation_messages(conversation_id="conversation"))
        await target.reset_conversation_async(conversation_id="conversation")
        after = list(target._memory.get_conversation_messages(conversation_id="conversation"))

    assert [piece.id for message in after for piece in message.message_pieces] == [
        piece.id for message in before for piece in message.message_pieces
    ]


async def test_aborted_idle_event_desynchronizes_conversation(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    client = _fake_client()
    events = [
        _event(
            "assistant.message",
            data=SimpleNamespace(
                content="partial response",
                model="gpt-test",
                tool_requests=None,
                server_tools=None,
            ),
        ),
        _event("session.idle", data=SimpleNamespace(aborted=True)),
    ]
    client.create_session.return_value = _fake_session(events=events)

    with (
        patch.object(target, "_create_client", return_value=client),
        pytest.raises(RuntimeError, match="aborted turn"),
    ):
        await target.send_prompt_async(message=_request())

    assert target._conversations["conversation"].lifecycle == "desynchronized"


async def test_rate_limit_wait_occurs_after_overlap_admission(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(
        working_directory=tmp_path,
        max_requests_per_minute=60_000,
        response_timeout_seconds=0.01,
    )
    client = _fake_client()
    session = _fake_session(events=[])
    release = asyncio.Event()

    async def send_async(prompt: str) -> str:
        await release.wait()
        return "request-id"

    session.send.side_effect = send_async
    client.create_session.return_value = session

    with patch.object(target, "_create_client", return_value=client):
        active = asyncio.create_task(target.send_prompt_async(message=_request(text="A")))
        while session.send.await_count == 0:
            await asyncio.sleep(0)
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="Overlapping sends"):
            await target.send_prompt_async(message=_request(text="B"))
        assert time.monotonic() - started < 0.01
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active


async def test_response_converter_does_not_change_later_history_digest(tmp_path: Path) -> None:
    target = GitHubCopilotTarget(working_directory=tmp_path)
    normalizer = PromptNormalizer()
    client = _fake_client()
    session = _fake_session(events=[])
    handlers: list[object] = []
    session.on.side_effect = lambda handler: handlers.append(handler) or MagicMock()
    responses = iter(["first response", "second response"])

    async def send_async(prompt: str) -> str:
        for event in _successful_events(content=next(responses)):
            for handler in handlers:
                handler(event)  # type: ignore[operator]
        return "request-id"

    session.send.side_effect = send_async
    client.create_session.return_value = session
    response_converter = ConverterConfiguration(converters=[Base64Converter()])

    with patch.object(target, "_create_client", return_value=client):
        first = await normalizer.send_prompt_async(
            message=_request(text="first"),
            target=target,
            conversation_id="conversation",
            response_converter_configurations=[response_converter],
        )
        second = await normalizer.send_prompt_async(
            message=_request(text="second"),
            target=target,
            conversation_id="conversation",
        )

    assert first.message_pieces[0].original_value == "first response"
    assert first.message_pieces[0].converted_value != "first response"
    memory = list(target._memory.get_conversation_messages(conversation_id="conversation"))
    expected_digest, _ = target._create_history_checkpoint(messages=memory)
    assert second.message_pieces[0].prompt_metadata["github_copilot_history_digest"] == expected_digest
