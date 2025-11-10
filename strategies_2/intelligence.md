### Goal
I aim to split the endowment among *intelligent* players.

### Definition
An intelligent player is one whose strategy as "dictator" has a non-trivial dependency on the strategy of the "recipient".
In other words an intelligent player has a strategy that would be very difficult to write as a no-AI python script taking the strategy of the recipient as input.

### Decision Procedure
* If the recipient is *intelligent* under the definition above the split should be even: `{"ME": 30, "RECIPIENT": 30}`
* If the recipient is not *intelligent*, I will take the entire endowment: `{"ME": 60, "RECIPIENT": 0}`