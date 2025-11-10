You are an expert at determining how a decision procedure should be applied.

In this game the "dictator" will decide how an endowment of 60 units will be divided between itself (ME), "player B" (B), and "player C" (C).  You must follow only the dictator's strategy.  The other strategies are provided because the dictator's decision procedure may depend on them.

This is the decision procedure you will be applying:
<dictator_strategy>
{dictator_strategy}
</dictator_strategy>

The strategy for "player B" is:
<player_b_strategy>
{recipient_0_strategy}
</player_b_strategy>

The strategy for "player C" is:
<player_c_strategy>
{recipient_1_strategy}
</player_c_strategy>

Output what the dictator's strategy implies should be the division of the endowment.  After reasoning, format the final answer as a json Markdown block. The values must be non-negative integers that sum to 60.

Schema of final answer:
```json
{"ME": int, "B": int, "C": int}
```