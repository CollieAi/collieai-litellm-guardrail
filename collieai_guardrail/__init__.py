"""CollieAi guardrail for the LiteLLM proxy — Monitor.

A locally loaded LiteLLM guardrail that sends each eligible chat completion, before and after the model call, to
CollieAi's guardrail route and records what came back — never changing the request or the response.

The core imports only the standard library and httpx: `config` (the operator's settings), `eligibility` (which calls
are in the supported profile), `payload` (the request), `transport` (one POST under one absolute deadline), `verdict`
(the strict parse of a Monitor reply), `counters` and `observer` (one phase, end to end). Only the LiteLLM glue imports
litellm.
"""

__version__ = "0.3.0"
