# Changelog

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
