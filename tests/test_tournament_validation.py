"""Tests for validate_tournament_rounds function."""

from os_fdt.rounds import build_rounds
from os_fdt.validation import validate_tournament_rounds, check_round_validation_summary
import random


def test_validate(n: int):
    """Test validation on a tournament."""
    print(f"\n=== Validating {n} strategies ===")
    strategy_names = [f"S_{i}" for i in range(n)]
    rounds = build_rounds(strategy_names, self_play=n % 2 == 0)

    result = validate_tournament_rounds(rounds)

    check_round_validation_summary(result)
    
    assert result['valid'], "Validation failed!"


def run_all_tests():
    """Run all validation tests."""
    passed = 0
    failed = 0
    for _ in range(100):
        try:
            n=random.randint(3, 100)
            test_validate(n)
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\n{'='*70}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_all_tests()
