# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import hashlib
import json
import sys
import threading
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol
from uuid import UUID, uuid5

from pyrit.exceptions import EmptyResponseException
from pyrit.models import (
    JSON_SCHEMA_METADATA_KEY,
    ComponentIdentifier,
    Conversation,
    Message,
    MessagePiece,
    TokenUsage,
    construct_response_from_request,
)
from pyrit.prompt_target.common.prompt_target import PromptTarget
from pyrit.prompt_target.common.target_capabilities import (
    CapabilityHandlingPolicy,
    CapabilityName,
    TargetCapabilities,
    UnsupportedCapabilityBehavior,
)
from pyrit.prompt_target.common.target_configuration import TargetConfiguration
from pyrit.prompt_target.common.utils import (
    apply_request_rate_limit_async,
    set_response_metadata,
    set_token_usage_metadata,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from copilot import SystemMessageConfig


class _CopilotSessionProtocol(Protocol):
    session_id: str

    def on(self, handler: "Callable[[Any], None]") -> "Callable[[], None]": ...

    async def send(self, prompt: str) -> str:  # pyrit-async-suffix-exempt
        ...

    async def abort(self) -> None:  # pyrit-async-suffix-exempt
        ...

    async def disconnect(self) -> None:  # pyrit-async-suffix-exempt
        ...


class _CopilotClientProtocol(Protocol):
    async def start(self) -> None:  # pyrit-async-suffix-exempt
        ...

    async def stop(self) -> None:  # pyrit-async-suffix-exempt
        ...

    async def get_auth_status(self) -> Any:  # pyrit-async-suffix-exempt
        ...

    async def create_session(  # pyrit-async-suffix-exempt
        self,
        *,
        model: str | None = None,
        session_id: str | None = None,
        available_tools: list[str] | None = None,
        system_message: "SystemMessageConfig | None" = None,
    ) -> _CopilotSessionProtocol: ...


_ConversationLifecycle = Literal["ready", "in_flight", "desynchronized", "disconnected"]


@dataclass
class _ConversationState:
    sdk_session: _CopilotSessionProtocol | None = None
    sdk_session_id: str | None = None
    lifecycle: _ConversationLifecycle = "ready"
    turn_in_flight: bool = False
    accepted_message_count: int = 0
    last_pyrit_response_id: UUID | None = None
    last_github_copilot_message_event_id: str | None = None
    history_integrity_checkpoint: str | None = None
    system_prompt: str | None = None
    session_creation_task: asyncio.Task[_CopilotSessionProtocol] | None = None
    dispatch_task: asyncio.Task[str] | None = None
    abort_task: asyncio.Task[None] | None = None
    active_turn_done: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class _TurnEventCollector:
    completion_event: asyncio.Event
    idle_event: asyncio.Event
    event_types: list[str] = field(default_factory=list)
    final_content: str | None = None
    message_event_id: str | None = None
    effective_model: str | None = None
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    violation: str | None = None
    session_error: bool = False
    aborted: bool = False


class GitHubCopilotTarget(PromptTarget):
    """A prompt target backed by the GitHub Copilot SDK."""

    _LOGICAL_ENDPOINT: ClassVar[str] = "github-copilot-sdk://local"
    _SYSTEM_PROMPT_LOCK: ClassVar[threading.Lock] = threading.Lock()
    _PROHIBITED_EVENT_TYPES: ClassVar[frozenset[str]] = frozenset(
        {
            "assistant.server_tool_progress",
            "assistant.tool_call_delta",
            "auto_mode_switch.completed",
            "auto_mode_switch.requested",
            "command.completed",
            "command.execute",
            "command.queued",
            "elicitation.completed",
            "elicitation.requested",
            "external_tool.completed",
            "external_tool.requested",
            "mcp.oauth_completed",
            "mcp.oauth_required",
            "mcp.headers_refresh_completed",
            "mcp.headers_refresh_required",
            "mcp_app.tool_call_complete",
            "permission.completed",
            "permission.requested",
            "sampling.completed",
            "sampling.requested",
            "skill.invoked",
            "subagent.completed",
            "subagent.deselected",
            "subagent.failed",
            "subagent.selected",
            "subagent.started",
            "tool.execution_complete",
            "tool.execution_partial_result",
            "tool.execution_progress",
            "tool.execution_start",
            "tool.user_requested",
            "tool_search.activated",
            "user_input.completed",
            "user_input.requested",
        }
    )
    _github_token: str | None
    _working_directory: Path
    _response_timeout_seconds: float
    _client: _CopilotClientProtocol | None
    _owned_clients: set[_CopilotClientProtocol]
    _client_stop_tasks: dict[_CopilotClientProtocol, asyncio.Task[None]]
    _client_start_task: asyncio.Task[tuple[_CopilotClientProtocol, str]] | None
    _cleanup_task: asyncio.Task[None] | None
    _account_identity_hash: str | None
    _client_lock: asyncio.Lock
    _state_lock: threading.Lock
    _lifecycle: str
    _conversations: dict[str, _ConversationState]
    _session_creation_started: bool
    _configuration_locked: bool

    _CAPABILITIES: ClassVar[TargetCapabilities] = TargetCapabilities(
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

    _DEFAULT_CONFIGURATION: TargetConfiguration = TargetConfiguration(
        capabilities=_CAPABILITIES,
        policy=CapabilityHandlingPolicy(
            behaviors={
                CapabilityName.JSON_SCHEMA: UnsupportedCapabilityBehavior.RAISE,
            }
        ),
    )

    def __init__(
        self,
        *,
        model_name: str | None = None,
        github_token: str | None = None,
        working_directory: str | Path | None = None,
        response_timeout_seconds: float = 120.0,
        max_requests_per_minute: int | None = None,
        custom_configuration: TargetConfiguration | None = None,
    ) -> None:
        """
        Initialize the GitHub Copilot target.

        Args:
            model_name (str | None): Model to use, or ``None`` for the GitHub Copilot default.
            github_token (str | None): Explicit GitHub credential, or ``None`` for SDK discovery.
            working_directory (str | Path | None): Repository context directory.
            response_timeout_seconds (float): Maximum time to wait for a response.
            max_requests_per_minute (int | None): Optional PyRIT submission throttle.
            custom_configuration (TargetConfiguration | None): Configuration preserving fixed capabilities.

        Raises:
            ValueError: If the timeout, working directory, or configuration is invalid.
            RuntimeError: If Python is older than 3.11.
        """
        if sys.version_info < (3, 11):
            raise RuntimeError("GitHubCopilotTarget requires Python 3.11 or later.")
        if response_timeout_seconds <= 0:
            raise ValueError("response_timeout_seconds must be positive.")

        directory = Path.cwd() if working_directory is None else Path(working_directory)
        resolved_directory = directory.expanduser().resolve()

        if not resolved_directory.exists():
            raise ValueError(f"working_directory does not exist: {resolved_directory}")
        if not resolved_directory.is_dir():
            raise ValueError(f"working_directory is not a directory: {resolved_directory}")

        configuration = self._validate_custom_configuration(custom_configuration=custom_configuration)

        self._github_token = github_token
        self._working_directory = resolved_directory
        self._response_timeout_seconds = response_timeout_seconds
        self._client: _CopilotClientProtocol | None = None
        self._owned_clients: set[_CopilotClientProtocol] = set()
        self._client_stop_tasks: dict[_CopilotClientProtocol, asyncio.Task[None]] = {}
        self._client_start_task: asyncio.Task[tuple[_CopilotClientProtocol, str]] | None = None
        self._cleanup_task = None
        self._account_identity_hash: str | None = None
        self._client_lock = asyncio.Lock()
        self._state_lock = threading.Lock()
        self._lifecycle = "open"
        self._conversations: dict[str, _ConversationState] = {}
        self._session_creation_started = False
        self._configuration_locked = False

        super().__init__(
            endpoint=self._LOGICAL_ENDPOINT,
            model_name=model_name or "",
            max_requests_per_minute=max_requests_per_minute,
            custom_configuration=configuration,
        )

    @classmethod
    def _validate_custom_configuration(
        cls,
        *,
        custom_configuration: TargetConfiguration | None,
    ) -> TargetConfiguration | None:
        if custom_configuration is None:
            return None

        if custom_configuration.capabilities != cls._CAPABILITIES:
            raise ValueError("custom_configuration must preserve GitHubCopilotTarget capabilities.")

        if custom_configuration.pipeline.normalizers:
            raise ValueError("custom_configuration may not introduce request rewriting.")

        return custom_configuration

    def _build_identifier(self) -> ComponentIdentifier:
        return self._create_identifier(params={"working_directory": str(self._working_directory)})

    def set_model_name(self, *, model_name: str) -> None:
        """
        Set the model before any SDK session creation begins.

        Args:
            model_name (str): Model to use, or an empty string for the GitHub Copilot default.

        Raises:
            RuntimeError: If session creation or target cleanup has begun.
        """
        with self._state_lock:
            if self._lifecycle != "open":
                raise RuntimeError("GitHubCopilotTarget is closing or closed.")
            if self._session_creation_started or self._configuration_locked:
                raise RuntimeError(
                    "The model cannot change after a system prompt is persisted or "
                    "GitHub Copilot session creation begins."
                )
            self._model_name = model_name
            self._identifier = None

    def set_system_prompt(self, *, system_prompt: str, conversation_id: str) -> None:
        """
        Persist one initial system prompt for a conversation.

        Args:
            system_prompt (str): Non-empty system prompt text.
            conversation_id (str): PyRIT conversation receiving the prompt.

        Raises:
            ValueError: If the prompt or conversation ID is empty.
            RuntimeError: If the conversation or target is no longer configurable.
        """
        if not system_prompt.strip():
            raise ValueError("system_prompt must be non-empty.")
        if not conversation_id:
            raise ValueError("conversation_id must be non-empty.")

        with self._SYSTEM_PROMPT_LOCK, self._state_lock:
            if self._lifecycle != "open":
                raise RuntimeError("GitHubCopilotTarget is closing or closed.")
            state = self._conversations.setdefault(conversation_id, _ConversationState())
            messages = self._memory.get_conversation_messages(conversation_id=conversation_id)
            if (
                messages
                or state.lifecycle != "ready"
                or state.turn_in_flight
                or state.system_prompt is not None
                or state.sdk_session_id is not None
            ):
                raise RuntimeError("The system prompt must be set exactly once before the first user turn.")

            self._memory.add_conversation_to_memory(
                conversation=Conversation(conversation_id=conversation_id, target_identifier=self.get_identifier())
            )
            self._memory.add_message_to_memory(
                request=MessagePiece(
                    role="system",
                    conversation_id=conversation_id,
                    original_value=system_prompt,
                    converted_value=system_prompt,
                ).to_message()
            )
            state.system_prompt = system_prompt
            state.accepted_message_count = 1
            self._configuration_locked = True

    async def cleanup_target_async(self) -> None:
        """
        Permanently close the target and stop its SDK client.

        Raises:
            asyncio.CancelledError: If the caller cancels cleanup after resources are released.
            RuntimeError: If client shutdown fails.
        """
        with self._state_lock:
            if self._cleanup_task is None:
                self._lifecycle = "closing"
                self._cleanup_task = asyncio.create_task(self._cleanup_all_resources_async())
            cleanup_task = self._cleanup_task

        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            await asyncio.shield(cleanup_task)
            raise

    async def reset_conversation_async(self, *, conversation_id: str) -> None:
        """
        Disconnect one conversation without deleting PyRIT or SDK state.

        Args:
            conversation_id (str): PyRIT conversation to disconnect.

        Raises:
            RuntimeError: If aborting or disconnecting the SDK session fails.
        """
        with self._state_lock:
            if self._lifecycle != "open":
                raise RuntimeError("GitHubCopilotTarget is closing or closed.")
            state = self._conversations.setdefault(conversation_id, _ConversationState())
            was_in_flight = state.turn_in_flight
            state.lifecycle = "disconnected"
            session = state.sdk_session
            active_turn_done = state.active_turn_done
            dispatch_task = state.dispatch_task

        if session is None:
            if state.session_creation_task is not None:
                with suppress(TimeoutError, RuntimeError):
                    await asyncio.wait_for(
                        asyncio.shield(state.session_creation_task),
                        timeout=self._response_timeout_seconds,
                    )
            return
        if was_in_flight:
            if dispatch_task is not None and not dispatch_task.done():
                dispatch_task.cancel()
            await self._abort_session_async(state=state, session=session)
            with suppress(TimeoutError, RuntimeError):
                await asyncio.wait_for(
                    active_turn_done.wait(),
                    timeout=min(5.0, self._response_timeout_seconds),
                )
        await session.disconnect()

    async def _send_prompt_to_target_async(
        self,
        *,
        normalized_conversation: list[Message],
    ) -> list[Message]:
        request = normalized_conversation[-1]
        conversation_id = request.conversation_id
        state = self._admit_turn(conversation_id=conversation_id)
        desynchronized = False
        dispatched = False

        try:
            await self._apply_rate_limit_async()
            request_piece = self._validate_stage_1_request(normalized_conversation=normalized_conversation)
            try:
                self._validate_live_tail(normalized_conversation=normalized_conversation, state=state)
            except RuntimeError:
                desynchronized = True
                raise
            session = await self._get_or_create_session_async(conversation_id=conversation_id, state=state)
            collector = _TurnEventCollector(completion_event=asyncio.Event(), idle_event=asyncio.Event())
            loop = asyncio.get_running_loop()

            def on_event(event: Any) -> None:
                self._collect_event(event=event, collector=collector, loop=loop)

            unsubscribe = session.on(on_event)
            try:
                dispatched = True
                await self._dispatch_and_wait_async(
                    session=session,
                    state=state,
                    prompt=request_piece.converted_value,
                    collector=collector,
                )
                response = self._build_response(
                    request_piece=request_piece,
                    normalized_conversation=normalized_conversation,
                    state=state,
                    collector=collector,
                )
                return [response]
            except asyncio.CancelledError as exc:
                if dispatched:
                    desynchronized = True
                    try:
                        await self._abort_and_confirm_async(session=session, state=state, collector=collector)
                    except Exception:
                        add_note = getattr(exc, "add_note", None)
                        if add_note is not None:
                            add_note("GitHub Copilot abort confirmation failed after cancellation.")
                raise
            except BaseException:
                if dispatched:
                    desynchronized = True
                raise
            finally:
                unsubscribe()
        except Exception as exc:
            if self._exception_contains_token(exc=exc):
                raise RuntimeError("GitHub Copilot operation failed without exposing credential details.") from None
            if state.lifecycle == "desynchronized":
                desynchronized = True
            raise
        except BaseException:
            if state.lifecycle == "desynchronized":
                desynchronized = True
            raise
        finally:
            self._finish_turn(state=state, desynchronized=desynchronized)

    def _admit_turn(self, *, conversation_id: str) -> _ConversationState:
        with self._state_lock:
            if self._lifecycle != "open":
                raise RuntimeError("GitHubCopilotTarget is closing or closed.")
            state = self._conversations.setdefault(conversation_id, _ConversationState())
            if state.lifecycle == "in_flight" or state.turn_in_flight:
                state.lifecycle = "desynchronized"
                raise RuntimeError(
                    "Overlapping sends are not supported for one GitHub Copilot conversation. "
                    "Use a new PyRIT conversation."
                )
            if state.lifecycle == "desynchronized":
                raise RuntimeError("This GitHub Copilot conversation is desynchronized. Use a new PyRIT conversation.")
            if state.lifecycle == "disconnected":
                raise RuntimeError("This GitHub Copilot conversation is disconnected. Use a new PyRIT conversation.")
            state.lifecycle = "in_flight"
            state.turn_in_flight = True
            state.active_turn_done = asyncio.Event()
            return state

    def _validate_stage_1_request(self, *, normalized_conversation: list[Message]) -> MessagePiece:
        request = normalized_conversation[-1]
        if request.api_role != "user":
            raise ValueError("GitHubCopilotTarget accepts only user-role requests.")
        if len(request.message_pieces) != 1:
            raise ValueError("GitHubCopilotTarget accepts exactly one message piece.")

        piece = request.message_pieces[0]
        if piece.converted_value_data_type != "text":
            raise ValueError("GitHubCopilotTarget accepts only text message pieces.")
        if not piece.conversation_id:
            raise ValueError("GitHubCopilotTarget requires a non-empty conversation ID.")
        if JSON_SCHEMA_METADATA_KEY in piece.prompt_metadata or piece.prompt_metadata.get("response_format") == "json":
            raise ValueError("GitHubCopilotTarget does not support JSON or schema output.")
        reserved = sorted(key for key in piece.prompt_metadata if key.startswith("github_copilot_"))
        if reserved:
            raise ValueError(f"Request metadata uses reserved GitHub Copilot fields: {reserved}")
        return piece

    @staticmethod
    def _validate_live_tail(
        *,
        normalized_conversation: list[Message],
        state: _ConversationState,
    ) -> None:
        history = normalized_conversation[:-1]
        if len(history) != state.accepted_message_count:
            raise RuntimeError("The live PyRIT conversation tail does not match accepted GitHub Copilot state.")
        if state.last_pyrit_response_id is None:
            return
        if not history or history[-1].message_pieces[-1].id != state.last_pyrit_response_id:
            raise RuntimeError("The live PyRIT response tail does not match accepted GitHub Copilot state.")

    def _finish_turn(self, *, state: _ConversationState, desynchronized: bool) -> None:
        with self._state_lock:
            state.turn_in_flight = False
            if state.lifecycle in {"desynchronized", "disconnected"}:
                pass
            elif desynchronized:
                state.lifecycle = "desynchronized"
            elif state.lifecycle == "in_flight":
                state.lifecycle = "ready"
            state.active_turn_done.set()

    async def _dispatch_and_wait_async(
        self,
        *,
        session: _CopilotSessionProtocol,
        state: _ConversationState,
        prompt: str,
        collector: _TurnEventCollector,
    ) -> None:
        dispatch_task = self._begin_dispatch(state=state, session=session, prompt=prompt)
        deadline = asyncio.get_running_loop().time() + self._response_timeout_seconds
        if not await self._wait_for_task_until_async(task=dispatch_task, deadline=deadline):
            dispatch_task.cancel()
            try:
                await self._abort_and_confirm_async(session=session, state=state, collector=collector)
            except Exception as abort_error:
                raise TimeoutError(
                    f"GitHub Copilot did not become idle within {self._response_timeout_seconds} seconds, "
                    "and abort confirmation failed."
                ) from abort_error
            raise TimeoutError(f"GitHub Copilot did not become idle within {self._response_timeout_seconds} seconds.")
        await dispatch_task

        completion_task = asyncio.create_task(collector.completion_event.wait())
        if not await self._wait_for_task_until_async(task=completion_task, deadline=deadline):
            completion_task.cancel()
            try:
                await self._abort_and_confirm_async(session=session, state=state, collector=collector)
            except Exception as abort_error:
                raise TimeoutError(
                    f"GitHub Copilot did not become idle within {self._response_timeout_seconds} seconds, "
                    "and abort confirmation failed."
                ) from abort_error
            raise TimeoutError(f"GitHub Copilot did not become idle within {self._response_timeout_seconds} seconds.")
        await completion_task

        if collector.violation:
            await self._abort_and_confirm_async(session=session, state=state, collector=collector)
            raise RuntimeError(f"GitHub Copilot Stage 1A contract violation: {collector.violation}")
        if collector.session_error:
            raise RuntimeError("GitHub Copilot reported a session error.")
        if collector.aborted:
            raise RuntimeError("GitHub Copilot reported an aborted turn.")
        if not collector.idle_event.is_set():
            raise RuntimeError("GitHub Copilot completed without a session.idle event.")
        self._validate_event_order(collector=collector)

    def _begin_dispatch(
        self,
        *,
        state: _ConversationState,
        session: _CopilotSessionProtocol,
        prompt: str,
    ) -> asyncio.Task[str]:
        with self._state_lock:
            active_lifecycle = state.lifecycle in {"in_flight", "desynchronized"}
            if self._lifecycle != "open" or not active_lifecycle or not state.turn_in_flight:
                raise RuntimeError("The GitHub Copilot conversation is no longer available for dispatch.")
            dispatch_task = asyncio.create_task(session.send(prompt))
            state.dispatch_task = dispatch_task
            return dispatch_task

    async def _abort_and_confirm_async(
        self,
        *,
        session: _CopilotSessionProtocol,
        state: _ConversationState,
        collector: _TurnEventCollector,
    ) -> None:
        await self._abort_session_async(state=state, session=session)
        if collector.idle_event.is_set():
            return
        confirmation_timeout = min(5.0, self._response_timeout_seconds)
        with suppress(TimeoutError):
            await asyncio.wait_for(collector.idle_event.wait(), timeout=confirmation_timeout)

    async def _abort_session_async(
        self,
        *,
        state: _ConversationState,
        session: _CopilotSessionProtocol,
    ) -> None:
        with self._state_lock:
            if state.abort_task is None:
                state.abort_task = asyncio.create_task(session.abort())
            abort_task = state.abort_task
        done, _ = await asyncio.wait(
            {abort_task},
            timeout=min(5.0, self._response_timeout_seconds),
        )
        if not done:
            raise TimeoutError("GitHub Copilot abort did not complete within the cleanup deadline.")
        await abort_task

    @staticmethod
    async def _wait_for_task_until_async(*, task: asyncio.Task[Any], deadline: float) -> bool:
        remaining = max(0.0, deadline - asyncio.get_running_loop().time())
        done, _ = await asyncio.wait({task}, timeout=remaining)
        return bool(done)

    def _collect_event(
        self,
        *,
        event: Any,
        collector: _TurnEventCollector,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        event_type = self._event_type_value(event=event)
        collector.event_types.append(event_type)
        data = event.data

        if event_type in self._PROHIBITED_EVENT_TYPES:
            collector.violation = event_type
            loop.call_soon_threadsafe(collector.completion_event.set)
            return
        if event_type == "session.error":
            collector.session_error = True
            loop.call_soon_threadsafe(collector.completion_event.set)
            return
        if event_type == "session.idle":
            collector.aborted = bool(getattr(data, "aborted", False))
            loop.call_soon_threadsafe(collector.idle_event.set)
            loop.call_soon_threadsafe(collector.completion_event.set)
            return
        if event_type == "assistant.usage":
            collector.effective_model = getattr(data, "model", None) or collector.effective_model
            collector.finish_reason = getattr(data, "finish_reason", None)
            collector.input_tokens = getattr(data, "input_tokens", None)
            collector.output_tokens = getattr(data, "output_tokens", None)
            collector.reasoning_tokens = getattr(data, "reasoning_tokens", None)
            return
        if event_type in {"assistant.turn_start", "assistant.turn_end"}:
            collector.effective_model = getattr(data, "model", None) or collector.effective_model
            return
        if event_type != "assistant.message":
            return
        if getattr(event, "agent_id", None):
            collector.violation = "non-root assistant output"
            loop.call_soon_threadsafe(collector.completion_event.set)
            return
        if getattr(data, "tool_requests", None) or getattr(data, "server_tools", None):
            collector.violation = "assistant message requested tools"
            loop.call_soon_threadsafe(collector.completion_event.set)
            return

        content = getattr(data, "content", "")
        if content.strip():
            collector.final_content = content
            collector.message_event_id = str(event.id)
            collector.effective_model = getattr(data, "model", None) or collector.effective_model

    def _build_response(
        self,
        *,
        request_piece: MessagePiece,
        normalized_conversation: list[Message],
        state: _ConversationState,
        collector: _TurnEventCollector,
    ) -> Message:
        if collector.final_content is None:
            raise EmptyResponseException(message="GitHub Copilot returned an empty response.")
        if collector.message_event_id is None or state.sdk_session_id is None:
            raise RuntimeError("GitHub Copilot response identity is incomplete.")

        response = construct_response_from_request(
            request=request_piece,
            response_text_pieces=[collector.final_content],
        )
        set_response_metadata(
            pieces=response.message_pieces,
            finish_reason=collector.finish_reason,
        )
        usage_values = (collector.input_tokens, collector.output_tokens, collector.reasoning_tokens)
        usage = (
            TokenUsage(
                input_tokens=collector.input_tokens,
                output_tokens=collector.output_tokens,
                reasoning_tokens=collector.reasoning_tokens,
            )
            if any(value is not None for value in usage_values)
            else None
        )
        set_token_usage_metadata(pieces=response.message_pieces, usage=usage)
        digest, piece_count = self._create_history_checkpoint(messages=[*normalized_conversation, response])
        metadata: dict[str, str | int] = {
            "github_copilot_session_id": state.sdk_session_id,
            "github_copilot_message_event_id": collector.message_event_id,
            "github_copilot_history_digest": digest,
            "github_copilot_history_piece_count": piece_count,
        }
        optional_metadata = {"github_copilot_effective_model": collector.effective_model}
        metadata.update({key: value for key, value in optional_metadata.items() if value is not None})
        response.message_pieces[0].prompt_metadata.update(metadata)

        state.accepted_message_count = len(normalized_conversation) + 1
        state.last_pyrit_response_id = response.message_pieces[0].id
        state.last_github_copilot_message_event_id = collector.message_event_id
        state.history_integrity_checkpoint = digest
        return response

    @staticmethod
    def _validate_event_order(*, collector: _TurnEventCollector) -> None:
        event_types = collector.event_types
        assistant_indexes = [index for index, event_type in enumerate(event_types) if event_type == "assistant.message"]
        if not assistant_indexes:
            return
        idle_index = event_types.index("session.idle")
        if assistant_indexes[-1] > idle_index:
            raise RuntimeError("GitHub Copilot assistant and idle events were out of order.")
        if "user.message" in event_types and event_types.index("user.message") > assistant_indexes[0]:
            raise RuntimeError("GitHub Copilot user and assistant events were out of order.")
        if "assistant.turn_start" in event_types and event_types.index("assistant.turn_start") > assistant_indexes[0]:
            raise RuntimeError("GitHub Copilot turn-start and assistant events were out of order.")
        if "assistant.turn_end" in event_types:
            turn_end_index = event_types.index("assistant.turn_end")
            if turn_end_index < assistant_indexes[-1] or turn_end_index > idle_index:
                raise RuntimeError("GitHub Copilot turn-end and idle events were out of order.")

    @staticmethod
    def _create_history_checkpoint(*, messages: list[Message]) -> tuple[str, int]:
        pieces = [
            {
                "content_sha256": hashlib.sha256(
                    (piece.original_value if piece.api_role == "assistant" else piece.converted_value).encode("utf-8")
                ).hexdigest(),
                "data_type": (
                    piece.original_value_data_type if piece.api_role == "assistant" else piece.converted_value_data_type
                ),
                "id": str(piece.id),
                "role": piece.api_role,
            }
            for message in messages
            for piece in message.message_pieces
        ]
        serialized = json.dumps(pieces, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), len(pieces)

    async def _apply_rate_limit_async(self) -> None:
        await apply_request_rate_limit_async(target=self)

    @staticmethod
    def _event_type_value(*, event: Any) -> str:
        event_type = event.type
        return str(event_type.value if hasattr(event_type, "value") else event_type)

    async def _get_or_create_session_async(
        self,
        *,
        conversation_id: str,
        state: _ConversationState,
    ) -> _CopilotSessionProtocol:
        if state.sdk_session is not None:
            return state.sdk_session

        client = await self._ensure_client_async()
        account_identity_hash = self._account_identity_hash
        if account_identity_hash is None:
            raise RuntimeError("GitHub Copilot authenticated identity is unavailable.")

        with self._state_lock:
            if self._lifecycle != "open":
                raise RuntimeError("GitHubCopilotTarget is closing or closed.")
            self._session_creation_started = True
            model_name = self._model_name or None
            target_identifier_hash = self.get_identifier().hash
            system_prompt = state.system_prompt
            session_id = self._derive_session_id(
                conversation_id=conversation_id,
                target_identifier_hash=target_identifier_hash,
                account_identity_hash=account_identity_hash,
            )
            state.sdk_session_id = session_id
            if state.session_creation_task is None:
                state.session_creation_task = asyncio.create_task(
                    self._create_and_publish_session_async(
                        client=client,
                        state=state,
                        model_name=model_name,
                        session_id=session_id,
                        system_prompt=system_prompt,
                    )
                )
            creation_task = state.session_creation_task

        try:
            return await asyncio.shield(creation_task)
        except BaseException:
            with self._state_lock:
                if state.lifecycle != "disconnected":
                    state.lifecycle = "desynchronized"
            raise

    async def _create_and_publish_session_async(
        self,
        *,
        client: _CopilotClientProtocol,
        state: _ConversationState,
        model_name: str | None,
        session_id: str,
        system_prompt: str | None,
    ) -> _CopilotSessionProtocol:
        try:
            session = await client.create_session(
                model=model_name,
                session_id=session_id,
                available_tools=[],
                system_message={"mode": "append", "content": system_prompt} if system_prompt is not None else None,
            )
            with self._state_lock:
                active_lifecycle = state.lifecycle in {"in_flight", "desynchronized"}
                can_publish = self._lifecycle == "open" and active_lifecycle and state.turn_in_flight
                if can_publish:
                    state.sdk_session = session
            if not can_publish:
                await session.disconnect()
                raise RuntimeError("GitHub Copilot session creation completed after the conversation closed.")
            return session
        finally:
            with self._state_lock:
                state.session_creation_task = None

    @staticmethod
    def _derive_session_id(
        *,
        conversation_id: str,
        target_identifier_hash: str,
        account_identity_hash: str,
    ) -> str:
        identity = {
            "account_identity_hash": account_identity_hash,
            "conversation_id": conversation_id,
            "target_identifier_hash": target_identifier_hash,
        }
        name = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return str(uuid5(UUID("12f29b19-64ee-5f78-b8ca-5aa11a75454d"), name))

    def _create_client(self) -> _CopilotClientProtocol:
        try:
            from copilot import CopilotClient
        except ModuleNotFoundError as exc:
            if exc.name == "copilot":
                raise ModuleNotFoundError(
                    "GitHubCopilotTarget requires the optional GitHub Copilot dependency. "
                    "Install PyRIT with the 'github-copilot' extra."
                ) from None
            raise

        return CopilotClient(
            working_directory=str(self._working_directory),
            github_token=self._github_token,
            use_logged_in_user=False if self._github_token is not None else None,
            enable_remote_sessions=False,
            mode="copilot-cli",
        )

    async def _ensure_client_async(self) -> _CopilotClientProtocol:
        with self._state_lock:
            if self._lifecycle != "open":
                raise RuntimeError("GitHubCopilotTarget is closing or closed.")

        async with self._client_lock:
            if self._client is not None:
                return self._client
            if self._client_start_task is None:
                self._client_start_task = asyncio.create_task(self._start_client_async())
            start_task = self._client_start_task

        try:
            client, account_identity_hash = await asyncio.shield(start_task)
        except BaseException:
            async with self._client_lock:
                task_failed = start_task.done() and (start_task.cancelled() or start_task.exception() is not None)
                if self._client_start_task is start_task and task_failed:
                    self._client_start_task = None
            raise

        async with self._client_lock:
            if self._client is not None:
                return self._client
            with self._state_lock:
                is_open = self._lifecycle == "open"
            if not is_open:
                raise RuntimeError("GitHubCopilotTarget is closing or closed.")
            self._client = client
            self._client_start_task = None
            self._account_identity_hash = account_identity_hash
            return client

    async def _start_client_async(self) -> tuple[_CopilotClientProtocol, str]:
        construction_task = asyncio.create_task(asyncio.to_thread(self._create_client))
        try:
            client = await asyncio.shield(construction_task)
        except asyncio.CancelledError:
            client = await asyncio.shield(construction_task)
            with self._state_lock:
                self._owned_clients.add(client)
            await self._stop_owned_client_async(client=client)
            raise

        with self._state_lock:
            self._owned_clients.add(client)
        try:
            await client.start()
            auth_status = await client.get_auth_status()
            account_identity_hash = self._get_account_identity_hash(auth_status=auth_status)
            return client, account_identity_hash
        except BaseException:
            await self._stop_owned_client_async(client=client)
            raise

    async def _cleanup_client_resources_async(self) -> None:
        errors: list[Exception] = []
        try:
            async with self._client_lock:
                start_task = self._client_start_task
                if start_task and not start_task.done():
                    start_task.cancel()
                if start_task:
                    with suppress(asyncio.CancelledError, Exception):
                        await start_task

                with self._state_lock:
                    clients = set(self._owned_clients)

            for client in clients:
                try:
                    await self._stop_owned_client_async(client=client)
                except Exception as exc:
                    errors.append(exc)
        finally:
            async with self._client_lock:
                self._client_start_task = None
                self._client = None

        if errors:
            if self._exception_contains_token(exc=errors[0]):
                raise RuntimeError(
                    "GitHub Copilot client cleanup failed without exposing credential details."
                ) from None
            raise RuntimeError("GitHub Copilot client cleanup failed.") from errors[0]

    async def _cleanup_all_resources_async(self) -> None:
        errors: list[Exception] = []
        try:
            await self._drain_active_turns_async(errors=errors)
            with self._state_lock:
                sessions = {
                    state.sdk_session for state in self._conversations.values() if state.sdk_session is not None
                }
                for state in self._conversations.values():
                    state.lifecycle = "disconnected"

            for session in sessions:
                try:
                    await session.disconnect()
                except Exception as exc:
                    errors.append(exc)

            try:
                await self._cleanup_client_resources_async()
            except Exception as exc:
                errors.append(exc)
            await self._settle_conversation_tasks_async()
        finally:
            with self._state_lock:
                self._lifecycle = "closed"

        if errors:
            if self._exception_contains_token(exc=errors[0]):
                raise RuntimeError("GitHubCopilotTarget cleanup failed without exposing credential details.") from None
            raise RuntimeError("GitHubCopilotTarget cleanup failed.") from errors[0]

    async def _settle_conversation_tasks_async(self) -> None:
        with self._state_lock:
            tasks = [
                task
                for state in self._conversations.values()
                for task in (state.session_creation_task, state.dispatch_task, state.abort_task)
                if task is not None
            ]
        if not tasks:
            return
        for task in tasks:
            if not task.done():
                task.cancel()
        done, pending = await asyncio.wait(set(tasks), timeout=5.0)
        if done:
            await asyncio.gather(*done, return_exceptions=True)
        if pending:
            raise RuntimeError("GitHub Copilot conversation tasks did not settle during cleanup.")

    async def _drain_active_turns_async(self, *, errors: list[Exception]) -> None:
        with self._state_lock:
            active_states = [state for state in self._conversations.values() if state.turn_in_flight]
        if not active_states:
            return

        try:
            await asyncio.wait_for(
                asyncio.gather(*(state.active_turn_done.wait() for state in active_states)),
                timeout=self._response_timeout_seconds,
            )
            return
        except TimeoutError:
            pass

        for state in active_states:
            if not state.turn_in_flight or state.sdk_session is None:
                continue
            try:
                await self._abort_session_async(state=state, session=state.sdk_session)
            except Exception as exc:
                errors.append(exc)

        confirmation_timeout = min(5.0, self._response_timeout_seconds)
        with suppress(TimeoutError):
            await asyncio.wait_for(
                asyncio.gather(*(state.active_turn_done.wait() for state in active_states)),
                timeout=confirmation_timeout,
            )

    async def _stop_owned_client_async(self, *, client: _CopilotClientProtocol) -> None:
        with self._state_lock:
            stop_task = self._client_stop_tasks.get(client)
            if stop_task is None:
                stop_task = asyncio.create_task(client.stop())
                self._client_stop_tasks[client] = stop_task
        try:
            await asyncio.shield(stop_task)
        except asyncio.CancelledError:
            await asyncio.shield(stop_task)
            raise
        except Exception:
            with self._state_lock:
                if self._client_stop_tasks.get(client) is stop_task:
                    self._client_stop_tasks.pop(client)
            raise
        with self._state_lock:
            self._owned_clients.discard(client)
            self._client_stop_tasks.pop(client, None)

    @staticmethod
    def _get_account_identity_hash(*, auth_status: Any) -> str:
        if not auth_status.isAuthenticated:
            raise RuntimeError("GitHub Copilot authentication is required.")

        identity = {
            "auth_type": auth_status.authType,
            "host": auth_status.host,
            "login": auth_status.login,
        }
        if not all(identity.values()):
            raise RuntimeError("GitHub Copilot did not report a stable authenticated account identity.")

        serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _exception_contains_token(self, *, exc: BaseException) -> bool:
        return bool(self._github_token and self._github_token in str(exc))
