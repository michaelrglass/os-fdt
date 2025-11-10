"""Round generation and balancing algorithms for tournaments."""

from dataclasses import dataclass

@dataclass(frozen=True)
class Round:
    """Represents a single round in the tournament."""
    dictator_name: str
    player_names: frozenset[str]  # size 2

    def is_self_play(self):
        return self.dictator_name in self.player_names


def round_robin_matchings(n: int, exclude_ndx: int | None):
    """
    Circle method on the player set, optionally excluding one player.
    If exclude is None: n must be even (returns n-1 perfect matchings, each n/2 pairs).
    If exclude is not None: runs the circle on n-1 players (now even).
    """
    players = list(range(n))
    if exclude_ndx is not None:
        players.remove(exclude_ndx)
        n = len(players)
    rounds = []
    left = players[: n//2]
    right = players[n//2 :][::-1]
    all_paired_strategies = set()
    for bndx in range(n-1):
        rbatch = [tuple(sorted((a,b))) for a,b in zip(left, right)]
        # ensure no repeats
        for pndx, pair in enumerate(rbatch):
            if pair in all_paired_strategies:
                raise ValueError(f"repeated pair: {pair} at {bndx} of {n-1}; {pndx}")
            all_paired_strategies.add(pair)

        rounds.append(rbatch)
        players = [players[0]] + [players[-1]] + players[1:-1]
        left = players[: n//2]
        right = players[n//2 :][::-1]
     
    return rounds  # length n-1, each has n/2 pairs

def schedule_k_n_n_over_2(n: int, k: int, self_play: bool):
    """
    Returns (D, (i,j)) with i<j, unordered pair for the two recipients.
    - If self_play=True: require n even; per block produce n*(n/2) rounds.
    - If self_play=False: require n odd; per block produce n*((n-1)/2) rounds.
    """
    if self_play:
        assert n % 2 == 0, "self play requires even number of strategies"
    else:
        assert n % 2 != 0, "no self play requires an odd number of strategies"

    # with self_play
    if self_play:
        M = round_robin_matchings(n, None)
        R = len(M)  # n-1
    else:
        M = []
        R = 0

    rounds = []
    for D in range(n):
        if not self_play:
            M = round_robin_matchings(n, D)
            R = len(M)  # n-2
        for t in range(k):
            r = (D + t) % R
            for pair in M[r]:
                rounds.append((D, pair))  # unordered pair
    return rounds  # length k * n * (n//2)

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
                 number_of_recipients: int,
                 self_play: bool, 
                 max_dictator_round_per_strategy: int = 1000, 
                 dummy_strategy: str = 'dummy') -> set[Round]:
    if number_of_recipients == 2:
        if (len(strategy_names) % 2 != 0) == self_play:
            if dummy_strategy not in strategy_names:
                raise ValueError(f'Need even number of strategies for self play / odd for no self play! Dummy strategy `{dummy_strategy}` not found!')
            strategy_names.remove(dummy_strategy)
  
    n = len(strategy_names)

    if number_of_recipients == 1:
        round_selection = schedule_n_n(n, self_play)
    elif number_of_recipients == 2:
        scale_factor_k = max_dictator_round_per_strategy // (n // 2)

        # if our scale factor is too high it would be more than exhaustive
        if scale_factor_k > n - (1 if self_play else 2):
            scale_factor_k = n - (1 if self_play else 2)

        round_selection = schedule_k_n_n_over_2(n, scale_factor_k, self_play)
    else:
        raise NotImplementedError(f'Not implemented for number of recipients = {number_of_recipients}')
    
    rounds: set[Round] = set()
    for r_ndx in round_selection:
        r = Round(dictator_name=strategy_names[r_ndx[0]], 
                  player_names=frozenset([strategy_names[sndx] for sndx in r_ndx[1]]))
        if r in rounds:
            raise ValueError(f"Duplicate round!\n{r.dictator_name}\n{r.player_names}")
        rounds.add(r)
    return rounds
