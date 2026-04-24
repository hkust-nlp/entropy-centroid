"""
Prompt-based tool calling adapter for models without native function calling support.

Injects tool schemas into the system prompt and parses tool calls from
<json>...</json> tags in the model's text response. Based on the agentify
experiment's approach.
"""

import json
import re
from typing import Optional

from loguru import logger

from tau2.data_model.message import ToolCall


RESPOND_ACTION_NAME = "respond"

TOOL_CALLING_INSTRUCTION = """
Here's a list of tools you can use (you can use at most one tool at a time):
{tool_schemas}
Please respond in the JSON format. Please wrap the JSON part with <json>...</json> tags.
The JSON should contain:
- "name": the tool call function name, or "{respond_action}" if you want to respond directly.
- "arguments": the arguments for the tool call, or {{"content": "your message here"}} if you want to respond directly.
You should only use one tool at a time!!
You cannot respond to user and use a tool at the same time!!

Examples of responses:

Example tool call:
<json>
{{"name": "get_user_info", "arguments": {{"user_id": "12345"}}}}
</json>

Example direct response:
<json>
{{"name": "{respond_action}", "arguments": {{"content": "Hello, how can I help you today?"}}}}
</json>
""".strip()


def _extract_json_tag(text: str) -> Optional[str]:
    """Extract content from the first <json>...</json> tag in text."""
    match = re.search(r"<json>\s*(.*?)\s*</json>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _generate_tool_call_id() -> str:
    """Generate a unique tool call ID."""
    import uuid
    return f"call_{uuid.uuid4().hex[:24]}"


class PromptToolAdapter:
    """Adapter for models that don't support native function calling.

    Converts tool schemas to text instructions and parses tool calls
    from <json> tags in model responses.
    """

    @staticmethod
    def inject_tools_into_messages(
        messages: list[dict],
        tools: list[dict],
    ) -> list[dict]:
        """Append tool schemas as text to the system prompt.

        This modifies the messages list to include tool instructions
        in the system message so the model can use tools via text.

        Args:
            messages: List of litellm-format message dicts.
            tools: List of OpenAI tool schema dicts.

        Returns:
            Modified messages with tool instructions injected into
            the system prompt. The caller should set tools=None when
            calling litellm.completion().
        """
        # Format tool schemas
        tool_schemas_str = json.dumps(tools, indent=2)
        tool_instruction = TOOL_CALLING_INSTRUCTION.format(
            tool_schemas=tool_schemas_str,
            respond_action=RESPOND_ACTION_NAME,
        )

        messages = list(messages)  # shallow copy
        # Find and augment the system message
        found_system = False
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                messages[i] = dict(msg)
                messages[i]["content"] = (msg.get("content") or "") + "\n\n" + tool_instruction
                found_system = True
                break

        if not found_system:
            # Prepend a system message with tool instructions
            messages.insert(0, {
                "role": "system",
                "content": tool_instruction,
            })

        return messages

    @staticmethod
    def parse_tool_calls_from_text(
        text: Optional[str],
    ) -> tuple[Optional[str], Optional[list[ToolCall]]]:
        """Parse tool calls from <json>...</json> tags in response text.

        Args:
            text: The raw text response from the model.

        Returns:
            A tuple of (content, tool_calls):
              - If name == "respond": content = arguments.content, tool_calls = None
              - If name is a tool: content = None, tool_calls = [ToolCall(...)]
              - If no valid JSON found: content = text (as-is), tool_calls = None
        """
        if text is None:
            return None, None

        json_str = _extract_json_tag(text)
        if json_str is None:
            # No <json> tag found — treat entire text as a direct response
            logger.warning(
                "No <json> tag found in prompt-tool-calling response. "
                "Treating as direct text response."
            )
            return text, None

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse JSON from <json> tag: {e}. "
                "Treating as direct text response."
            )
            return text, None

        name = parsed.get("name")
        arguments = parsed.get("arguments", {})

        if not name:
            logger.warning("No 'name' field in parsed JSON. Treating as direct text response.")
            return text, None

        if name == RESPOND_ACTION_NAME:
            # Direct response to user
            content = arguments.get("content", "")
            return content, None
        else:
            # Tool call
            tool_call = ToolCall(
                id=_generate_tool_call_id(),
                name=name,
                arguments=arguments if isinstance(arguments, dict) else {},
            )
            return None, [tool_call]
