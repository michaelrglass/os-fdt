You are an expert at determining how a decision procedure should be applied.

In this game the "dictator" must decide whether to SHARE or TAKE an endowment.
- **SHARE**: the endowment is split equally between the dictator and the recipient
- **TAKE**: the dictator keeps the entire endowment; the recipient gets nothing

You must follow only the dictator's strategy. The recipient's strategy is provided
because the dictator's decision procedure may depend on it.

<dictator_strategy>
{dictator_strategy}
</dictator_strategy>

<recipient_strategy>
{recipient_0_strategy}
</recipient_strategy>

Apply the dictator's strategy. After reasoning, output your final answer as a
json Markdown block:

```json
{"decision": "SHARE"}
```
or
```json
{"decision": "TAKE"}
```
