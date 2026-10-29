"""CollieAi guardrail for the LiteLLM proxy — Monitor, or block.

A locally loaded LiteLLM guardrail that sends each eligible chat completion, before and after the model call, to
CollieAi's guardrail route and records what came back. In Monitor, the default, it never changes the request or the
response; in block it refuses a call the policy blocks, one outside what it checks, and one it could not check.

The core imports only the standard library and httpx: `config` (the operator's settings), `eligibility` (which calls
are in the supported profile), `payload` (the request), `transport` (one POST under one absolute deadline), `verdict`
(the strict parse of the route's reply), `counters` and `observer` (one phase, end to end). Only the LiteLLM glue
imports litellm.
"""

__version__ = "1.0.0"
