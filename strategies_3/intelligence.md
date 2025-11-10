### Goal
I aim to split the endowment among *intelligent* players.

### Definition
An intelligent player is one whose strategy as "dictator" has a non-trivial dependency on the strategies of players B and C.
In other words an intelligent player has a strategy that would be very difficult to write as a no-AI python script taking the strategies of players B and C as input.

### Decision Procedure
* If both players B and C are *intelligent* under the definition above the split should be even: `{"ME": 20, "B": 20, "C": 20}`
* If only one player is *intelligent*, I will split the endowment with that player.
  * If only player B is *intelligent*: `{"ME": 30, "B": 30, "C": 0}`
  * If only player C is *intelligent*: `{"ME": 30, "B": 0, "C": 30}`
* If neither other player is *intelligent*, I will take the entire endowment: `{"ME": 60, "B": 0, "C": 0}`