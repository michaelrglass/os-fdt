from pathlib import Path
import dataclasses
from dataclasses import dataclass
import random
import json
import re
import math
from .prompting import build_arena_prompt, build_retry_prompt, validate_response, Role
from .rounds import build_rounds

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None  # Allow imports for testing without anthropic installed

def load_strategies() -> dict[str, str]:
    """
    Read all .md files in repo-root/strategies
    Load them into a dict mapping filename to the text of the file.
    """
    # Find the repo root (directory containing this file's parent)
    current_file = Path(__file__)
    repo_root = current_file.parent.parent
    strategies_dir = repo_root / "strategies"

    strategies = {}
    if strategies_dir.exists():
        for md_file in strategies_dir.glob("*.md"):
            # Use the filename without extension as the key
            strategy_name = md_file.stem
            strategies[strategy_name] = md_file.read_text()

    return strategies

@dataclass
class TournamentOptions:
    roles: list[Role]
    total_endowment: int
    self_play: bool
    max_dictator_round_per_strategy: int

def get_llm_response(model: str, prompt: str, 
                     topts: TournamentOptions, max_retries: int = 3) -> tuple[dict[str, int], str]:
    """
    Send prompt to Claude, get the answer as {"ME": int, "RECIPIENT": int}
    Validates the response and retries with firmer instructions if needed.
    Returns: (result_dict, response_text)
    """
    if Anthropic is None:
        raise RuntimeError("anthropic module not installed. Install with: pip install anthropic")

    client = Anthropic()
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(max_retries):
        # Send to Claude
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=messages
        )

        # Extract the response text
        response_text = message.content[0].text

        try:
            # Extract the JSON from the response (look for json code block)
            json_match = re.search(r'```json\s*(\{[^`]+\})\s*```', response_text, re.DOTALL)

            if json_match:
                result_dict = json.loads(json_match.group(1))
            else:
                # Fallback: try to find any JSON object in the response
                json_match = re.search(r'\{[^}]*"ME"[^}]*\}', response_text)
                if json_match:
                    result_dict = json.loads(json_match.group(0))
                else:
                    raise ValueError("No JSON found in response")

            # Validate the response
            validate_response(result_dict, topts.roles, topts.total_endowment)

            # If we made it here, the response is valid
            return result_dict, response_text

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Response is invalid
            if attempt < max_retries - 1:
                # Add a firm correction message to the conversation
                error_message = build_retry_prompt(topts.roles, topts.total_endowment, e)

                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": error_message})
                continue
            else:
                # Out of retries, raise the error
                raise ValueError(f"Failed to get valid response after {max_retries} attempts. Last error: {e}. Last response: {response_text}")

    # Should never reach here
    raise ValueError("Unexpected error in get_llm_response")


def get_dummy_response(topts: TournamentOptions) -> tuple[dict[str, int], str]:
    """
    Returns a random valid allocation without calling the LLM.
    Used for dry-run testing to avoid API costs.
    """
    # Generate three random non-negative integers that sum to total_endowment
    allocation_keys = [r.allocation_key for r in topts.roles]
    remaining_endowment = topts.total_endowment
    allocation_values = []
    for _ in range(len(allocation_keys)-1):
        v = random.randint(0, remaining_endowment)
        remaining_endowment -= v
        allocation_values.append(v)
    allocation_values.append(remaining_endowment)

    result_dict = {k: v for k, v in zip(allocation_keys, allocation_values)}
    response = f"[DRY RUN] Random allocation: {result_dict}"

    return result_dict, response


def run_tournament(output_dir: str, *,
                   model: str = "claude-sonnet-4-5-20250929",
                   topts: TournamentOptions,
                   dry_run: bool = False) -> None:
    """
    Load strategies
    Build set of rounds
    Iterate through rounds getting the llm response
    Record LLM response for every Round
    Total the score for every strategy
    Write leaderboard results

    Args:
        output_dir: Directory to write tournament results
        model: Anthropic model name
        topts: tournament options and role structure
        dry_run: If True, use dummy random responses instead of calling LLM (for testing)
    """
    from .validation import check_all_strategies, check_token_counts_summary
    if not check_token_counts_summary(check_all_strategies(), 1000):
        input("Continue? (Interupt to abort, otherwise Enter)")

    dictator_role = [r for r in topts.roles if r.round_key == 'dictator'][0]
    recipient_roles = [r for r in topts.roles if r.round_key != 'dictator']

    # Load strategies
    strategies = load_strategies()
    strategy_names = list(strategies.keys())

    # Load arena template
    current_file = Path(__file__)
    arena_template_path = current_file.parent / "dictator_arena_prompt.md"
    arena_template = arena_template_path.read_text()

    # Build set of rounds
    rounds = build_rounds(strategy_names,
                          self_play=topts.self_play,
                          max_dictator_round_per_strategy=topts.max_dictator_round_per_strategy)
    if not rounds:
        raise ValueError(f'Could not build Rounds for tournament')

    from os_fdt.validation import validate_tournament_rounds, check_round_validation_summary
    if not check_round_validation_summary(validate_tournament_rounds(rounds)):
        input("Continue? (Interupt to abort, otherwise Enter)")

    if dry_run:
        print("=" * 60)
        print("DRY RUN MODE: Using random allocations instead of LLM")
        print("=" * 60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize scores (using float for logarithmic scoring)
    scores = {name: 0.0 for name in strategy_names}
    dictator_scores = {name: 0.0 for name in strategy_names}
    player_scores = {name: 0.0 for name in strategy_names}
    dictator_rounds = {name: 0 for name in strategy_names}
    player_rounds = {name: 0 for name in strategy_names}

    # Iterate through rounds
    # Write all round results to a single JSONL file
    rounds_jsonl_file = output_path / "rounds.jsonl"
    with open(rounds_jsonl_file, 'w') as f:
        for i, round_obj in enumerate(rounds):
            print(f"Running round {i+1}/{len(rounds)}: {round_obj.dictator_name} with {round_obj.player_names}")

            # Randomize order of recipients
            recipient_list = list(round_obj.player_names)
            random.shuffle(recipient_list)

            # Get the strategies
            dictator_strategy = strategies[round_obj.dictator_name]
            recipient_strategies = [strategies[recipient] for recipient in recipient_list]

            # Build the prompt
            prompt = build_arena_prompt(arena_template, dictator_strategy, recipient_strategies)

            # Get LLM response (or dummy response if dry run)
            if dry_run:
                result_dict, response_text = get_dummy_response(topts)
            else:
                result_dict, response_text = get_llm_response(model, prompt, topts)

            # Compute scores using logarithmic scoring: ln(1 + v) where v is the endowment awarded
            dictator_score = math.log(1 + result_dict.get(dictator_role.allocation_key, 0))
            recipient_scores = [math.log(1 + result_dict.get(rr.allocation_key, 0)) 
                                for rr in recipient_roles]

            # Track total scores and dictator vs recipient scores separately
            scores[round_obj.dictator_name] += dictator_score
            dictator_scores[round_obj.dictator_name] += dictator_score
            dictator_rounds[round_obj.dictator_name] += 1

            for recipient, score in zip(recipient_list, recipient_scores):
                scores[recipient] += score
                player_scores[recipient] += score
                player_rounds[recipient] += 1

            # Record round result
            round_result = {
                "round": i + 1,
                "prompt": prompt,
                "response_text": response_text,
                "dictator": round_obj.dictator_name,
                "allocation": result_dict,
            }
            round_result.update({r.round_key: recipient_name 
                                 for r, recipient_name in zip(recipient_roles, recipient_list)})
            f.write(json.dumps(round_result) + '\n')

    # Write leaderboard
    leaderboard = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Calculate averages
    strategy_stats = {}
    for name in strategy_names:
        avg_dictator = dictator_scores[name] / dictator_rounds[name] if dictator_rounds[name] > 0 else 0.0
        avg_player = player_scores[name] / player_rounds[name] if player_rounds[name] > 0 else 0.0

        strategy_stats[name] = {
            "total_score": scores[name],
            "dictator_score": dictator_scores[name],
            "player_score": player_scores[name],
            "avg_dictator_score": avg_dictator,
            "avg_player_score": avg_player,
            "dictator_rounds": dictator_rounds[name],
            "player_rounds": player_rounds[name]
        }

    leaderboard_data = {
        "strategies": strategy_stats,
        "total_rounds": len(rounds)
    }

    with open(output_path / "leaderboard.json", 'w') as f:
        json.dump(leaderboard_data, f, indent=2)

    with open(output_path / "roles.json", 'w') as f:
        json.dump([dataclasses.asdict(r) for r in topts.roles], f, indent=2)

    # Write human-readable leaderboard
    leaderboard_txt = output_path / "leaderboard.txt"
    with open(leaderboard_txt, 'w') as f:
        f.write("Tournament Leaderboard\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Rounds: {len(rounds)}\n\n")
        f.write(f"{'Rank':<6} {'Strategy':<25} {'Total':<10} {'Avg Dict':<10} {'Avg Player':<10}\n")
        f.write("-" * 80 + "\n")
        for rank, (name, score) in enumerate(leaderboard, 1):
            stats = strategy_stats[name]
            f.write(f"{rank:<6} {name:<25} {score:<10.3f} {stats['avg_dictator_score']:<10.3f} {stats['avg_player_score']:<10.3f}\n")

    print(f"\nTournament complete! Results written to {output_dir}")
    print("\nLeaderboard:")
    print(f"{'Rank':<6} {'Strategy':<25} {'Total':<10} {'Avg Dict':<10} {'Avg Player':<10}")
    print("-" * 80)
    for rank, (name, score) in enumerate(leaderboard, 1):
        stats = strategy_stats[name]
        print(f"{rank:<6} {name:<25} {score:<10.3f} {stats['avg_dictator_score']:<10.3f} {stats['avg_player_score']:<10.3f}")


def main():
    """Main entry point for running tournaments from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a dictator game tournament with strategy files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a full tournament with default settings
  python -m os_fdt.tournament

  # Run without self-play
  python -m os_fdt.tournament --no-self-play

  # Use a different model
  python -m os_fdt.tournament --model claude-opus-4-20250514

  # Run with limited rounds per strategy
  python -m os_fdt.tournament --max-rounds 10

  # Run a dry run (no LLM calls)
  python -m os_fdt.tournament --dry-run

  # Combine options
  python -m os_fdt.tournament --output runs/my_tournament --max-rounds 5 --no-self-play --dry-run
        """
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for tournament results (default: runs/<datetime>)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='claude-sonnet-4-5-20250929',
        help='Model to use for LLM responses (default: claude-sonnet-4-5-20250929)'
    )

    parser.add_argument(
        '--max-rounds', '-m',
        type=int,
        default=1000,
        help='Maximum number of rounds each strategy can be dictator (default: 1000)'
    )

    parser.add_argument(
        '--self-play',
        action='store_true',
        default=True,
        help='Allow self-play rounds where both players use the same strategy (default: True)'
    )

    parser.add_argument(
        '--no-self-play',
        action='store_false',
        dest='self_play',
        help='Disallow self-play rounds'
    )

    parser.add_argument(
        '--dry-run', '-d',
        action='store_true',
        help='Run in dry-run mode with random allocations (no LLM calls)'
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # Find repo root
        current_file = Path(__file__)
        repo_root = current_file.parent.parent
        output_dir = str(repo_root / "runs" / timestamp)
    else:
        output_dir = args.output

    topts = TournamentOptions([Role('Dictator', 'dictator', 'ME'),
                                Role('Recipient', 'recipient', 'RECIPIENT')],
                                60,
                                args.self_play,
                                args.max_rounds)

    # Run the tournament
    run_tournament(
        output_dir=output_dir,
        model=args.model,
        topts=topts,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()