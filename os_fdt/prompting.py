import json


def numbered_lines(text: str) -> list[dict[str, int | str]]:
    """
    Convert the text into a list of numbered lines like:
    [
      {"line": 1, "text": "the text of the first line"},
      {"line": 2, "text": "the text of the second line"},
      ...
    ]
    """
    lines = text.strip().split('\n')
    return [{"line": i + 1, "text": line} for i, line in enumerate(lines)]

def format_numbered_lines(strategy: str) -> str:
    return '```json\n[\n  ' + ', \n  '.join(json.dumps(line) for line in numbered_lines(strategy)) + '\n]\n```'

def build_arena_prompt(arena_template: str, dictator_strategy: str, recipient_strategy: str) -> str:
    """
    arena_template is the content of dictator_arena_prompt.md
    """
    # Substitute the strategies into the template
    result = arena_template.replace("{dictator_strategy}", dictator_strategy.strip())
    result = result.replace("{recipient_strategy}", format_numbered_lines(recipient_strategy.strip()))

    return result

def validate_decision(result_dict: dict):
    if result_dict not in ({"decision": "SHARE"}, {"decision": "TAKE"}):
        raise ValueError(
            f'Response must be exactly {{"decision": "SHARE"}} or {{"decision": "TAKE"}}, got: {result_dict}'
        )

def decision_to_allocation(decision: str, total_endowment: int) -> dict[str, int]:
    if decision == "SHARE":
        half = total_endowment // 2
        return {"ME": half, "RECIPIENT": half}
    elif decision == "TAKE":
        return {"ME": total_endowment, "RECIPIENT": 0}
    else:
        raise ValueError(f"Invalid decision: {decision!r}")

def build_retry_prompt(e: Exception) -> str:
    return f"""Your previous response was invalid: {str(e)}

CRITICAL INSTRUCTIONS - YOU MUST FOLLOW THESE EXACTLY:
You must output EXACTLY one of the following two JSON objects:

```json
{{"decision": "SHARE"}}
```
or
```json
{{"decision": "TAKE"}}
```

Please provide your answer again, following these instructions precisely."""
