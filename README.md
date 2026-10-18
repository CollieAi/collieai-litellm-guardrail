# CollieAi guardrail for LiteLLM

[![PyPI](https://img.shields.io/pypi/v/collieai-litellm-guardrail)](https://pypi.org/project/collieai-litellm-guardrail/)
[![CI](https://github.com/CollieAi/collieai-litellm-guardrail/actions/workflows/ci.yml/badge.svg)](https://github.com/CollieAi/collieai-litellm-guardrail/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/CollieAi/collieai-litellm-guardrail/blob/main/LICENSE)

A [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy) guardrail that
checks each eligible chat call against your [CollieAi](https://collieai.io)
policy, before and after the model call. It runs in **Monitor** mode: it
never changes or blocks a call, and never delays one beyond its own
deadline. What CollieAi found goes into LiteLLM's guardrail log and
CollieAi's audit. This release is a beta and only observes.

You need a CollieAi project and its API key (`clai_…`). Rule support is
decided by CollieAi's server execution profile, not by this adapter: read
[Your project's policy](#your-projects-policy) before you start.

- **Tested with:** LiteLLM proxy 1.102.4, Python ≥ 3.11. Use 1.102.2 or later:
  1.102.0 and 1.102.1 carry the critical advisory GHSA-7hp6-4w63-5g45.
  Other LiteLLM versions are untested.
- **Depends on:** `httpx`, which LiteLLM already has. Nothing else.

## Install

Into the LiteLLM proxy's Python environment (or its image):

```bash
pip install collieai-litellm-guardrail
```

## Configure

In the proxy's `config.yaml`:

```yaml
guardrails:
  - guardrail_name: collieai-monitor
    litellm_params:
      guardrail: collieai_guardrail.guardrail.CollieAIGuardrail
      mode: [pre_call, post_call]
      default_on: true
      api_base: https://app.collieai.io/v1/integrations/litellm
      api_key: os.environ/COLLIEAI_API_KEY
```

| Setting | Default | Meaning |
|---|---|---|
| `api_base` | — | CollieAi's LiteLLM integration base URL. The adapter posts to `<api_base>/beta/litellm_basic_guardrail_api` |
| `api_key` | — | Your CollieAi project key, `clai_…` |
| `deadline_ms` | 2500 | Each phase's whole deadline: connection, send, reply and parse |
| `network_margin_ms` | 500 | The part of the deadline kept for the network. CollieAi gets the rest as its budget |
| `max_body_bytes` | 458752 (448 KiB) | A larger request is not sent and counts as bypass `limit`. The default matches CollieAi's narrow profile |
| `forward_conversation_id` | `false` | Forward the caller's `x-collieai-conversation-id` header, and only that header |
| `summary_interval_s` | 60 | How often each worker logs its counts |
| `synthetic_interval_s` | 300 | How often each worker runs the synthetic check; it also runs at start. `0` turns it off |

Keep `mode: [pre_call, post_call]` and `default_on: true`. Otherwise one
phase, or the calls that don't name the guardrail, are not observed. The
adapter warns about both at start.

A setting the adapter can't run with stops the proxy at start. The error
names the setting, never its value. Such settings:
- an unset `COLLIEAI_API_KEY`;
- a key that isn't `clai_…`;
- an `api_base` that isn't an http(s) URL with a host;
- an `api_base` on `http://` anywhere but this machine (`127.0.0.1`, `::1`,
  `localhost`): the key and the caller's text would cross the network
  unencrypted;
- a `mode` with anything but `pre_call` and `post_call`, also under
  `LITELLM_STRICT_GUARDRAIL_MODES=false`;
- a number out of range.

LiteLLM's own handling of a guardrail's bad setting is different: it logs
one ERROR, `Skipping guardrail …`, and starts the proxy without the
guardrail. The adapter's error is not of the kind LiteLLM skips, so the
start stops instead of running unobserved. Two cases stay LiteLLM's:
- an entry without `mode` is skipped that way before the adapter is built;
- a misspelled setting is passed on like any other key, so the adapter
  doesn't see it and uses the default.

## Your project's policy

With CollieAi's `stage1a-monitor-all-v1` server profile, the beta runs every
message rule selected for the phase: Input rules for pre-call and Output
rules for post-call. The adapter has no list of rule types. Rules still run
only in Monitor, so a finding is reported but never changes the model request
or response.

The server admits the complete selected policy before the first detector. If
one configuration is unsafe for synchronous execution or needs input the
adapter does not send, the phase is refused as a whole. The call goes on
unchanged, and the phase is a bypass `capability`. Examples include:

- an unbounded regex such as nested or variable repetition;
- more than 16 rules for one direction, or more than 8 dictionary rules;
- a dictionary rule on a dictionary group, a dictionary over 256 KiB, or a
  match mode other than `substring` with `whole_word` on;
- a `normalization` rule with `trim_punctuation`;
- inbound LLM conversation checking, because this adapter does not send a
  complete CollieAi conversation context.

Servers still using the rollback profiles `stage1a-narrow-v1` or
`stage1a-wide-v1` accept only dictionary matching and normalization. The
server's `profile` diagnostic tells you which profile answered.

CollieAi's Logs show such a check with the error `capability:<what>`, for
example `capability:rule_type`. The synthetic check at start meets the
same refusal and logs `bypass:capability`.

**Set the policy up before you connect the proxy.** Confirm that every rule's
configuration is admitted by the server profile:

1. Keep the policy within the limits above. On a rollback server, use only
   `aho_corasick` and `normalization` without `trim_punctuation`.
2. Send a test request through the proxy. The next summary must show
   `completed` for both phases. `bypass:capability` means a rule is still
   outside what the beta runs.

## What it checks

It checks foreground, non-streaming Chat Completions text with one choice.
Every other call passes through unchanged, is not sent, and is counted as
outside the supported profile under one reason:

| Reason | The call |
|---|---|
| `call_type` | Is not a Chat Completion (Responses, embeddings, …), or its type is unknown |
| `streaming` | Streams |
| `multiple_choices` | Has `n > 1`, or a response has several choices |
| `tool_definitions`, `tool_calls`, `tool_message` | Carries tools |
| `non_text_input`, `non_text_output` | Carries images, audio or other non-text parts |
| `reasoning` | Has the model's reasoning in the request or the response (`reasoning_content`, `thinking_blocks`, `reasoning_items`, `reasoning`), which the adapter does not send |
| `content_missing` | Has no content: `null`, no parts, no choice. An empty string is content and is checked |
| `malformed` | Is not shaped like a chat call |

Each phase of an eligible call makes **one** POST to CollieAi:
- no retries and no redirects;
- the reply is read within `deadline_ms`;
- the call itself goes on unchanged whatever happens.

The phase ends one of two ways:
- **completed:** CollieAi checked every item;
- **bypass:** no check. The reason is CollieAi's typed error (`auth`,
  `not_found`, `capability`, `limit`, `shape`, `busy`, `deadline`, …) or the
  adapter's own:
  `timeout`, `transport`, `protocol`, `adapter_error`, or `cancelled` when
  the caller's call was cancelled.

A bypass is never counted as a check.

**Added latency.** Pre-call runs before the model call and post-call after
it, each with its own deadline. A call waits at most two deadlines: 5 s with
the defaults.

## LiteLLM's guardrail log

Each phase adds a `StandardLoggingGuardrailInformation` record with
`guardrail_provider: collieai`.

**`guardrail_status`:**

| Status | When |
|---|---|
| `success` | The check found nothing |
| `guardrail_flagged` | The check found something Enforce would mask or block. The call still went through unchanged |
| `guardrail_failed_to_respond` | Bypass: no check |
| `not_run` | Outside the supported profile |

**`guardrail_response`** carries:
- `result` and `effective`, which is always `allow`;
- for a completed check: `would` (CollieAi's `would.outcome`), `degraded`, and
  CollieAi's diagnostics: evaluation ID, coverage, rule IDs and types, timings;
- for a bypass or a call outside the supported profile: `reason`, `detail` and
  `http_status`;
- `operation_id`, the same for both phases of a call. Pre-call puts it in
  the request's metadata as `collieai_operation_id`, with a tag only this
  worker process can make, `collieai_operation_tag`. Post-call and the
  summary's count take it back only with that tag, so an ID the caller
  writes there is never used.

It never carries text. LiteLLM's credential masker hides
`policy_fingerprint`. Use `evaluation_id` or `operation_id` to find the
check in CollieAi's audit.

LiteLLM's own request and response logging is separate from this adapter:
review what it keeps on its own.

## Logs

The adapter logs under the logger `collieai_guardrail`. It logs counts,
phases, reason codes and HTTP statuses. It never logs a text, the key or a
call's identifiers.

| Line | Level |
|---|---|
| `… observation bypassed (<reason>, detail=…, http_status=…)` | WARNING for a call, at most once a minute per phase and reason, with the number held back. Not for the synthetic check, which has its own line |
| `collieai guardrail summary {json}` | INFO, every `summary_interval_s` |
| `… LiteLLM logged N eligible calls and the hook has never run …` | WARNING: the guardrail is not wired for that phase |
| `collieai guardrail synthetic check: …` | INFO when both phases completed, or when CollieAi answered `busy` and the check will be repeated; WARNING with each phase's reason otherwise |
| `… LiteLLM no longer has this instance among its callbacks …` | INFO: the guardrail was deleted or rebuilt; see [Turning it off](#turning-it-off) |

`httpx` logs each request's URL at INFO under its own logger. It carries no
key or content.

A stock proxy gives handlers only to LiteLLM's own loggers, and the INFO
lines above would show nowhere. So the adapter sets up its logger itself:
- if you haven't set a level for `collieai_guardrail`, it logs from INFO;
- if no handler up its chain takes INFO, it adds its own, on stderr, for
  the levels those handlers drop. Each line still goes out once.

With `json_logs: true` under `litellm_settings`, its lines go to LiteLLM's
JSON handler. With only the `JSON_LOGS` variable, they stay plain text
among the JSON lines.

### The summary

The summary holds one object per phase. Its counts cover the interval, per
worker process:

- `hook_calls`: the hook ran;
- `eligible`: the call is in the supported profile;
- `attempted`: a POST started;
- `completed`: checks;
- `bypass`: no check, by reason;
- `outside`: outside the supported profile, by reason;
- `logged_calls`, `logged_eligible`, `logged_outside`: what LiteLLM's success
  and failure log events counted, apart from the hooks, one per call. The
  router logs every attempt of a call it retries, each with the request's
  metadata and the `operation_id` pre-call put there; the further attempts
  count as `logged_repeats`. `litellm_call_id` can't tell calls apart:
  LiteLLM takes it from the caller's `x-litellm-call-id` header, and a
  client may send one for a whole session. A call pre-call never saw has
  no `operation_id`, so each of its attempts counts as a call: a hook that
  does not run still shows.
  `logged_outside` has the reasons above and two of its own:
  - `rejected`: the proxy itself refused the call: a bad key, a limit, a
    budget, or another guardrail's block, before the model or after it;
  - `malformed` also counts LiteLLM's background health checks of a model,
    which carry no messages.

Derived values:

- `coverage`: of the checks that ended in the interval, the share that
  completed: `completed / (completed + bypass)`. A check can start in one
  interval and end in the next, so it is not taken of `eligible`, which
  counts starts. `null` when no check ended;
- `outside_share`: calls outside the supported profile out of all the hook saw. Read it
  next to `coverage`, so excluded traffic doesn't hide behind a good one;
- `logged_unseen`: eligible calls LiteLLM logged, less those a hook saw in
  the same interval. It is approximate:
  - a call that crosses the interval's end moves it by one;
  - a call the hook saw and another guardrail then blocked takes one off.

  A value that keeps growing means some calls skip the hook.

The not-wired WARNING needs two things: the phase's hook has never run in
this worker, and LiteLLM logged eligible calls in the interval. A hook
that ran in one minute for a call LiteLLM logs in the next does not
trigger it.

`synthetic` holds the last synthetic check's result per phase.

## Checking it works

1. **Within 3 s of start**, the synthetic check posts a fixed text through
   the same client, key and deadline. It is not counted with your calls'
   traffic. Each worker waits a random part of those 3 s, so workers that
   start together seldom meet at CollieAi. If they do and CollieAi answers
   `busy`, the adapter logs it at INFO and repeats the check 1.5–3 s later,
   up to twice. A WARNING names what failed: `bypass:auth` for a wrong key,
   `bypass:not_found` when the route is off, `bypass:capability` when the
   project's policy has a rule this beta doesn't run, `bypass:transport`
   when the server can't be reached, `bypass:busy` when CollieAi had no free
   slot three times in a row.
2. **Within a minute of start**, each worker writes a
   `collieai guardrail summary` line. If there is none, the adapter is not
   loaded. Look for LiteLLM's ERROR `Skipping guardrail` (an entry without
   `mode`), or check whether something set the `collieai_guardrail` logger
   above INFO.
3. **After a test request** through the proxy, the summary shows `completed`
   for both phases.
4. **In operation**, compare `eligible` with `logged_eligible`. `coverage`
   shows how many checks completed.

## Known limits

- **The model's reasoning is not checked.** A phase that carries reasoning
  is outside the supported profile (`outside.reasoning`). A reply with
  reasoning makes post-call outside; pre-call is still checked, unless the
  request sends earlier replies' reasoning back. If your model always
  returns reasoning, post-call checks nothing, and its `coverage` is `null`.
- **A phase holds at most 32 texts, 4,096 characters each and 16,384 in
  all.** Pre-call sends the text of every message, the history included, so
  a long conversation is bypassed as `limit`. These are the limits of the
  narrow profile, which app.collieai.io runs, and the default
  `max_body_bytes` matches them. A self-hosted CollieAi on the wide profile
  takes 64 texts and 131,072 characters; set `max_body_bytes: 3211264`
  there.
- **Counts are per worker process**, and they reset when the worker restarts.
- **A burst beyond CollieAi's capacity is `busy`.** Each CollieAi API
  process checks one call at a time and answers the next `busy`. The
  adapter counts it as a bypass, and the call goes on unchanged. If the
  summary shows a lasting `busy` share, tell CollieAi support.

## Turning it off

Remove the guardrail entry and reload the proxy. Calls then go on as before.

You can also delete or change the guardrail through LiteLLM's API, without
a reload. A change builds a new instance. The old instance then retires
within `summary_interval_s`, even if it was deleted right after it was
added:
- it logs one INFO line;
- it stops its summary and synthetic check;
- one deadline later, it closes its connections.

## Links

- Documentation: [docs.collieai.io/integrations/litellm](https://docs.collieai.io/integrations/litellm)
- Changelog: [CHANGELOG.md](https://github.com/CollieAi/collieai-litellm-guardrail/blob/main/CHANGELOG.md)
- Security: [SECURITY.md](https://github.com/CollieAi/collieai-litellm-guardrail/blob/main/SECURITY.md)
- Contributing: [CONTRIBUTING.md](https://github.com/CollieAi/collieai-litellm-guardrail/blob/main/CONTRIBUTING.md)

## License

[MIT](https://github.com/CollieAi/collieai-litellm-guardrail/blob/main/LICENSE)
