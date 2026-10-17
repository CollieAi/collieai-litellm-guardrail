# Security policy

## Reporting a vulnerability

Please report security vulnerabilities privately, not in public issues, pull
requests or discussions:

- email **security@collieai.io**, or
- use GitHub's private vulnerability reporting:
  [Report a vulnerability](https://github.com/CollieAi/collieai-litellm-guardrail/security/advisories/new).

Include the package version (`collieai-litellm-guardrail`), the LiteLLM
version, the steps to reproduce, and the impact you observed. Do not include
real API keys or customer data; synthetic values are enough.

We confirm receipt within **3 business days**, keep you informed while we
investigate and fix, agree the disclosure date with you, and credit you in
the advisory if you wish.

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x (latest release) | Yes |

## Safe harbor

We will not pursue legal action against research done in good faith that
follows this policy:

- test only against your own CollieAi account and projects;
- do not access, change or delete data that is not yours, beyond the minimum
  needed to demonstrate the issue;
- do not degrade the service (no denial-of-service or load testing) and do
  not use social engineering or physical attacks;
- report the issue promptly and give us reasonable time to fix it before any
  public disclosure.

## Scope

This repository contains the guardrail that runs inside your LiteLLM proxy.
Vulnerabilities in the CollieAi service itself can be reported to the same
address. Vulnerabilities in LiteLLM belong to the LiteLLM project.
