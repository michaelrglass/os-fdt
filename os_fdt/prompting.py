import json


def build_arena_prompt(arena_template: str, dictator_strategy: str, recipient_strategies: list[str]) -> str:
    """
    arena_template is the content of dictator_arena_prompt.md
    """
    dictator_strategy = dictator_strategy.strip()

    # Substitute the strategies into the template
    result = arena_template.replace("{dictator_strategy}", dictator_strategy)
    for rndx, recipient_strategy in enumerate(recipient_strategies):
        recipient_strategy = recipient_strategy.strip()
        result = result.replace(f"{{recipient_{rndx}_strategy}}", recipient_strategy)

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
