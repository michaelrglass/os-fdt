"""Validation utilities for tournament strategies."""

from pathlib import Path
from anthropic import Anthropic
from collections import Counter
import statistics
from .rounds import Round


def validate_tournament_rounds(rounds: set[Round]) -> dict:
    """
    Validate tournament rounds and check fairness properties.

    Checks:
    1. Each strategy is a dictator an equal number of times
    2. Each strategy is a recipient an equal number of times
    3. Each strategy has an equal amount of self-play
    4. Each tuple (dictator, {recipients...}) appears at most once (enforced by set[Round])
    5. All recipient sets appear "about" equally often

    Args:
        rounds: Set of Round objects to validate

    Returns:
        Dictionary with validation results and statistics
    """
    if not rounds:
        return {
            "valid": True,
            "error": "Empty round set",
            "dictator_counts": {},
            "player_counts": {},
            "self_play_counts": {},
            "total_appearance_counts": {},
            "recipient_group_stats": {}
        }

    # Extract all strategy names
    all_strategies = set()
    for round_obj in rounds:
        all_strategies.add(round_obj.dictator_name)
        all_strategies.update(round_obj.player_names)

    # Initialize counters
    dictator_counts = {name: 0 for name in all_strategies}
    player_counts = {name: 0 for name in all_strategies}
    self_play_counts = {name: 0 for name in all_strategies}
    total_appearance_counts = {name: 0 for name in all_strategies}
    recipient_group_counts = Counter()

    # Count appearances
    for round_obj in rounds:
        # Dictator count
        dictator_counts[round_obj.dictator_name] += 1

        # Player counts
        for player in round_obj.player_names:
            player_counts[player] += 1
            total_appearance_counts[player] += 1

        # Self-play count (if dictator is one of the players)
        if round_obj.is_self_play():
            self_play_counts[round_obj.dictator_name] += 1

        # Player pair count (unordered)
        recipient_group_counts[round_obj.player_names] += 1

    # Check property 1: equal dictator appearances
    dictator_values = list(dictator_counts.values())
    equal_dictator = len(set(dictator_values)) == 1

    # Check property 2: equal player appearances
    player_values = list(player_counts.values())
    equal_player = len(set(player_values)) == 1

    # Check property 3: equal self-play
    self_play_values = list(self_play_counts.values())
    equal_self_play = len(set(self_play_values)) == 1

    # Check property 4: each (dictator, {recipients}) at most once
    # This is enforced by set[Round], so always True
    unique_rounds = True

    # Property 5: player pair distribution statistics
    group_counts = list(recipient_group_counts.values())
    if group_counts:
        group_min = min(group_counts)
        group_max = max(group_counts)
        group_mean = statistics.mean(group_counts)
        group_variance = statistics.variance(group_counts) if len(group_counts) > 1 else 0.0
    else:
        group_min = group_max = group_mean = group_variance = 0

    # Overall validation
    valid = equal_dictator and equal_player and equal_self_play and unique_rounds

    result = {
        "valid": valid,
        "total_rounds": len(rounds),
        "num_strategies": len(all_strategies),
        "checks": {
            "equal_dictator_appearances": equal_dictator,
            "equal_player_appearances": equal_player,
            "equal_self_play": equal_self_play,
            "unique_rounds": unique_rounds,
        },
        "dictator_counts": dictator_counts,
        "player_counts": player_counts,
        "self_play_counts": self_play_counts,
        "total_appearance_counts": total_appearance_counts,
        "recipient_group_stats": {
            "num_unique_pairs": len(recipient_group_counts),
            "min": group_min,
            "max": group_max,
            "mean": group_mean,
            "variance": group_variance,
        }
    }

    return result


def check_round_validation_summary(result) -> bool:
    if not result['valid']:
        print(f"\nValidation result: {result['valid']}")
        print(f"Total rounds: {result['total_rounds']}")
        print(f"Number of strategies: {result['num_strategies']}")

        print("\nChecks:")
        for check_name, passed in result['checks'].items():
            status = "PASS" if passed else "FAIL"
            print(f"  {check_name}: {status}")

        print(f"\nDictator counts: {result['dictator_counts']}")
        print(f"Player counts: {result['player_counts']}")
        print(f"Self-play counts: {result['self_play_counts']}")

        print(f"\nPlayer pair statistics:")
        print(f"  Unique pairs: {result['recipient_group_stats']['num_unique_pairs']}")
        print(f"  Min appearances: {result['recipient_group_stats']['min']}")
        print(f"  Max appearances: {result['recipient_group_stats']['max']}")
        print(f"  Mean appearances: {result['recipient_group_stats']['mean']:.2f}")
        print(f"  Variance: {result['recipient_group_stats']['variance']:.2f}")
        return False
    else:
        group_min = result['recipient_group_stats']['min']
        group_max = result['recipient_group_stats']['max']
        if group_max > group_min + 1:
            print(f"  Min/max recipient group appearances: {group_min} / {group_max}")
            print(f"  Mean appearances: {result['recipient_group_stats']['mean']:.2f}")
            print(f"  Variance: {result['recipient_group_stats']['variance']:.2f}")
            # CONSIDER: could return true here too
            return False
    return True

def count_tokens(text: str, model: str = "claude-sonnet-4-5-20250929") -> int:
    """
    Count tokens in text using Claude's tokenizer.

    Args:
        text: Text to count tokens for
        model: Model to use for token counting

    Returns:
        Number of tokens
    """
    client = Anthropic()

    count = client.messages.count_tokens(
        model=model,
        messages=[
            {"role": "user", "content": text}
        ]
    )

    return count.input_tokens


def check_all_strategies() -> dict[str, int]:
    """
    Check token counts for all strategy files in the strategies directory.

    Returns:
        Dictionary mapping strategy name to token count
    """
    # Find the strategies directory
    current_file = Path(__file__)
    repo_root = current_file.parent.parent
    strategies_dir = repo_root / "strategies"

    if not strategies_dir.exists():
        raise FileNotFoundError(f"Strategies directory not found: {strategies_dir}")

    token_counts = {}

    # Process each .md file
    for md_file in sorted(strategies_dir.glob("*.md")):
        strategy_name = md_file.stem
        content = md_file.read_text()

        try:
            token_count = count_tokens(content)
            token_counts[strategy_name] = token_count
        except Exception as e:
            print(f"Error counting tokens for {strategy_name}: {e}")
            token_counts[strategy_name] = -1

    return token_counts

def check_token_counts_summary(token_counts: dict[str, int], max_tokens: int) -> bool:
    # Calculate statistics
    total_strategies = len(token_counts)
    total_tokens = sum(t for t in token_counts.values() if t >= 0)
    max_tokens_found = max(token_counts.values()) if token_counts else 0
    min_tokens_found = min(t for t in token_counts.values() if t >= 0) if token_counts else 0
    avg_tokens = total_tokens / total_strategies if total_strategies > 0 else 0

    # Display results
    print(f"{'Strategy':<30} {'Tokens':>10} {'Status':>10}")
    print("=" * 52)

    for strategy_name, token_count in sorted(token_counts.items()):
        if token_count < 0:
            status = "ERROR"
        elif max_tokens and token_count > max_tokens:
            status = "! OVER"
        else:
            status = "OK"

        print(f"{strategy_name:<30} {token_count:>10} {status:>10}")

    # Summary
    print("=" * 52)
    print(f"\nSummary:")
    print(f"  Total strategies: {total_strategies}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average tokens: {avg_tokens:.1f}")
    print(f"  Min tokens: {min_tokens_found}")
    print(f"  Max tokens: {max_tokens_found}")

    if max_tokens:
        over_limit = [name for name, count in token_counts.items()
                        if count >= 0 and count > max_tokens]
        if over_limit:
            print(f"\nWARNING: {len(over_limit)} strategies exceed {max_tokens} tokens:")
            for name in over_limit:
                print(f"  - {name}: {token_counts[name]} tokens")
            return False
        else:
            print(f"\nAll strategies are within {max_tokens} token limit")
    return True
