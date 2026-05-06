# clbench-poker

Continual Learning Bench's `exploitable_poker` task wrapped as a
[verifiers](https://github.com/willccbb/verifiers) `MultiTurnEnv`, suitable
for Prime Intellect Hosted Training.

The agent plays heads-up Texas Hold'em against a deterministic exploitable
opponent (default `calling_station`). Reward is the per-hand chip profit
divided by the big blind. Continual-learning value comes from learning the
opponent's pattern over a sequence of hands within a single rollout.

## Args (passed via `[[env]].args` in the training TOML)

| Arg | Default | Notes |
|---|---|---|
| `task_kwargs` | `{num_instances=5, opponent_policy="calling_station", seed=0}` | Forwarded to CLBench's `Poker` constructor. |
| `max_instances_per_rollout` | `1` | Set ≥ 2 to enable continual mode; required for `use_notepad`. |
| `use_notepad` | `false` | Adds an `icl_notepad`-style `notepad_update` field to the action schema. |
| `notepad_max_chars` | `4000` | Soft cap; head-truncated when exceeded. |
| `max_turns` | `64` | Hard cap for the verifiers rollout loop. |
| `parse_failure_penalty` | `-1.0` | Per-failure reward delta. |
| `end_on_parse_failure` | `false` | If true, parse failure ends the rollout instead of re-prompting. |

See <https://github.com/sr-networks/clbench-verifiers> for the wrapper source
and a fuller architecture description.

## Reward

`mean_instance_reward` (mean per-hand chip profit / big blind) plus a
parse-failure penalty. Diagnostic-only (weight 0) signals on
`num_instances_completed`, `num_notepad_updates`, and `notepad_length_chars`.

## License

Apache-2.0
