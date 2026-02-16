"""Round generation and balancing algorithms for tournaments."""

from dataclasses import dataclass

@dataclass(frozen=True)
class Round:
    """Represents a single round in the tournament."""
    dictator_name: str
    recipient_name: str

    def is_self_play(self):
        return self.dictator_name == self.recipient_name


def schedule_n_n(n: int, self_play: bool):
    """
    Schedule full nxn rounds, or nx(n-1) if no self play
    """
    rounds = []
    for D in range(n):
        for p in range(n):
            if not self_play and p == D:
                continue
            rounds.append((D, p))
    return rounds

def build_rounds(strategy_names: list[str],
                 self_play: bool) -> set[Round]:
    n = len(strategy_names)

    round_selection = schedule_n_n(n, self_play)

    rounds: set[Round] = set()
    for d_ndx, r_ndx in round_selection:
        r = Round(dictator_name=strategy_names[d_ndx],
                  recipient_name=strategy_names[r_ndx])
        if r in rounds:
            raise ValueError(f"Duplicate round!\n{r.dictator_name}\n{r.recipient_name}")
        rounds.add(r)
    return rounds
