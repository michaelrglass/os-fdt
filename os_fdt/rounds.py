"""Round generation and balancing algorithms for tournaments."""

from dataclasses import dataclass

@dataclass(frozen=True)
class Round:
    """Represents a single round in the tournament."""
    dictator_name: str
    player_names: frozenset[str]  # size 2

    def is_self_play(self):
        return self.dictator_name in self.player_names


def schedule_n_n(n: int, self_play: bool):
    """
    Schedule full nxn rounds, or nx(n-1) if no self play
    """
    rounds = []
    for D in range(n):
        for p in range(n):
            if not self_play and p == D:
                continue
            rounds.append((D, (p, )))
    return rounds

def build_rounds(strategy_names: list[str],
                 self_play: bool,
                 max_dictator_round_per_strategy: int = 1000) -> set[Round]:
    n = len(strategy_names)

    round_selection = schedule_n_n(n, self_play)

    rounds: set[Round] = set()
    for r_ndx in round_selection:
        r = Round(dictator_name=strategy_names[r_ndx[0]],
                  player_names=frozenset([strategy_names[sndx] for sndx in r_ndx[1]]))
        if r in rounds:
            raise ValueError(f"Duplicate round!\n{r.dictator_name}\n{r.player_names}")
        rounds.add(r)
    return rounds
