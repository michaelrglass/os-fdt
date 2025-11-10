import json
from dataclasses import dataclass


@dataclass
class Role:
    display: str
    round_key: str
    allocation_key: str


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

def build_arena_prompt(arena_template: str, dictator_strategy: str, recipient_strategies: list[str]) -> str:
    """
    arena_template is the content of dictator_arena_prompt.md
    """
    dictator_strategy = dictator_strategy.strip()
    
    # Substitute the strategies into the template
    result = arena_template.replace("{dictator_strategy}", dictator_strategy)
    for rndx, recipient_strategy in enumerate(recipient_strategies):
        recipient_strategy = format_numbered_lines(recipient_strategy)
        result = result.replace(f"{{recipient_{rndx}_strategy}}", recipient_strategy)

    return result

def validate_response(result_dict: dict[str, int], roles: list[Role], total_endowment: int):
    # Validate the response
    required_keys = set(r.allocation_key for r in roles)
    if not all(key in result_dict for key in required_keys):
        raise ValueError(f"Missing required keys. Expected {required_keys}, got {set(result_dict.keys())}")

    # Check that all values are integers
    if not all(isinstance(result_dict[key], int) for key in required_keys):
        raise ValueError(f"Values must be integers, got: {result_dict}")

    # Check that values are non-negative
    if not all(result_dict[key] >= 0 for key in required_keys):
        raise ValueError(f"Values must be non-negative, got: {result_dict}")

    # Check that the total is `total_endowment`
    total = sum(result_dict[key] for key in required_keys)
    if total != total_endowment:
        raise ValueError(f"Values must sum to {total_endowment}, got {total}: {result_dict}")

def build_retry_prompt(roles: list[Role], total_endowment: int, e: Exception) -> str:
    allocation_key_list = ", ".join(f'"{r.allocation_key}"' for r in roles[:-1]) + \
        f', and "{roles[-1].allocation_key}"'
    schema_str = '{' + ", ".join(f'"{r.allocation_key}": <integer>' for r in roles) + '}'
    return f"""Your previous response was invalid: {str(e)}

CRITICAL INSTRUCTIONS - YOU MUST FOLLOW THESE EXACTLY:
1. You MUST output a JSON object with EXACTLY these keys: {allocation_key_list}
2. Each value MUST be a non-negative integer
3. The values MUST sum to EXACTLY {total_endowment}
4. Format your answer as a JSON markdown code block like this:

```json
{schema_str}
```

Please provide your answer again, following these instructions precisely."""