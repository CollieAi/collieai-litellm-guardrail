# Changelog

## 1.0.0 — 2026-10-29

- Block mode is no longer a pilot. The README gives app.collieai.io's
  capacity in block — up to 8 checks a second with the Starter policy —
  and what a call gets past it.
- `enforcement_mode: block` does not start with a mode by tags, or with a
  LiteLLM `stream_scope` other than `both` (LiteLLM 1.106): each runs a
  phase only for some calls, and the others went through unchecked.
- In block mode a request over the limits CollieAi checks, or of a text
  shape it doesn't take (bypass `limit`, `shape`), is refused with `400`
  `unsupported_request` (`unsupported_response` for the model's reply)
  instead of `503`: the same call is refused again, so it is not one to
  retry. LiteLLM's record is `guardrail_intervened`.
- `enforcement_mode: block` needs the readiness guard,
  `collieai_guardrail.guardrail.require_guardrail`, in
  `litellm_settings.callbacks` (or `require_guardrail` in
  `litellm.callbacks` where you embed the adapter); without it the proxy
  does not start. The guard refuses every call
  (`503 guardrail_not_loaded`) while the worker has no CollieAi guardrail
  in block mode. Before, LiteLLM started without a guardrail whose entry
  it refused (no `mode`, a value of the wrong type), and a misspelled
  `enforcement_mode` left the adapter in Monitor: both let calls through
  unchecked.
- Tested with LiteLLM 1.106.5 too.

## 0.3.1 — 2026-10-27

- `enforcement_mode: block`: a refused call is counted in the summary's
  `logged_*` counts once, when it is refused. LiteLLM logs a guardrail's
  refusal differently by version (1.102.4 not at all, 1.106.4 as a call
  its proxy refused), so in 0.3.0 `logged_unseen` went below zero by the
  refused calls and a block after the model was counted as `rejected`.

## 0.3.0 — 2026-10-27

- Tested with LiteLLM 1.106.4 as well as 1.102.4.
- `enforcement_mode: block`, a pilot: the adapter refuses a call the
  CollieAi policy blocks (`400`, `content_blocked` before the model,
  `response_blocked` after it), a call it could not check (`503`,
  `guardrail_unavailable`), and a call outside what it checks (`400`,
  `unsupported_request` / `unsupported_response`) unless it is an
  embedding, rerank or moderation call. LiteLLM records a block as
  `guardrail_intervened` with `guardrail_action: "BLOCKED"`. It needs
  `mode: [pre_call, post_call]`, `default_on: true`, a CollieAi host that
  allows blocking and an Enforce policy. Masks only observe in this mode.
  `monitor`, the default, is unchanged.
- The summary counts blocks (`blocked`) and refused calls by code
  (`refused`).

## 0.2.0 — 2026-10-25

What a check found is now visible in LiteLLM, also where LiteLLM redacts the
guardrail's response.

- `guardrail_flagged` means a rule found what it looks for: a finding of kind
  `detection`. CollieAi 2026.10.23 and later name each finding's kind. A
  check in a Monitor policy that found a card number was `success` in 0.1.0,
  because Enforce of that policy would have done nothing; it is now
  `guardrail_flagged`. A rule's `allow` match and a detector that failed are
  not violations. With a server that names no kind, the status is decided as
  in 0.1.0. A reply where some findings name a kind and some don't, or with a
  kind this version doesn't know, is a bypass `protocol`.
- A check that found nothing while a rule did not run is
  `guardrail_failed_to_respond`, not `success`: it gave no full answer. One
  that found something is still `guardrail_flagged`.
- LiteLLM's record of a completed check has `guardrail_action: "NONE"` and
  `violation_categories`: the rule types of what it found. LiteLLM's spend
  logs keep both where they redact `guardrail_response`
  (`store_prompts_in_spend_logs: false`, LiteLLM's default).
- `guardrail_response` keeps each finding's `kind` and the rules that did not
  apply to the text (`coverage.not_applicable`).
- A refusal says why and for which rule: CollieAi's error code as `detail`
  (for example `aho_group_content`) and the rule's ID as `rule_id`, in
  `guardrail_response`, the bypass WARNING and the synthetic check's WARNING.
  They used to say only `capability` and `detail=None`.
- The summary names its guardrail (`guardrail`, the `guardrail_name` of the
  proxy's config), and every other line ends with `(guardrail <name>)`: two
  CollieAi guardrails in one proxy no longer look alike in the log.
- An unset key names the variable to set: `api_key is not set; … is
  COLLIEAI_API_KEY set in the proxy's environment?`. It used to say only that
  `api_key` must be a non-empty string.
- The README shows how to install into the official LiteLLM Docker image,
  which has no `pip`: `python3 -m ensurepip` first.
- Document server-side `stage1a-monitor-all-v2`, which app.collieai.io runs:
  the Starter policy's `prompt-security` dictionary group is admitted, and a
  rule that does not apply to a text is not counted as degraded.

## 0.1.0 — 2026-10-17

First public release, beta.

- A LiteLLM guardrail that sends each eligible chat completion to CollieAi
  before and after the model call, in Monitor: it never blocks or changes a
  call.
- Eligible: non-streaming Chat Completions text with one choice. Anything
  else passes through unchanged and is counted as outside, by reason.
- One request per phase under one absolute deadline; any failure leaves the
  call unchanged and is counted as a bypass, by reason.
- A summary of the counts in the proxy's log every minute, a synthetic check
  at start and every five minutes, and a count of what LiteLLM logged apart
  from the hooks.
- Runs the project's dictionary matching and normalization rules only. A
  policy with another rule type makes each check a bypass `capability`.
- Tested with LiteLLM 1.102.4.
