# Contributing

Thank you for helping improve the CollieAi guardrail for LiteLLM. Bug
reports, questions and pull requests are welcome.

## How changes land

This repository is published from CollieAi's internal source, where every
change also runs against the CollieAi server's own test suites and a full
LiteLLM proxy. We review pull requests here; an accepted change is then
applied in the internal source and comes back to this repository with the
next export. Your pull request is closed with a link to that commit, and you
are credited as a co-author (`Co-authored-by:`).

Direct pushes to `main` are not possible for anyone: every change arrives
through a reviewed pull request with green CI.

## Before you start

- For a bug, open an issue with the package version, the LiteLLM version, a
  minimal configuration and what you expected.
- For a larger change, open an issue first so we can agree on the approach.
- Security issues go to security@collieai.io, not to issues
  (see [SECURITY.md](SECURITY.md)).

## Setting up and running the tests

The commands below are the ones CI runs ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).
No CollieAi account or network access to the service is needed: the tests use
local fakes. Python 3.11 or newer.

```bash
python -m pip install -e ".[dev]" "litellm[proxy]==1.102.4"
python -m ruff check collieai_guardrail tests
python -m ruff format --check collieai_guardrail tests
python -m mypy collieai_guardrail
python -m pytest
```

A run with skipped tests does not count as passing: the tests of the LiteLLM
glue need LiteLLM installed.

## Behavior changes

The guardrail never changes a call: whatever happens to a check, the request
and the response pass through as they are. A change in behavior is accepted
only with tests, and must keep that promise.

## Commit messages

One logical change per commit, summary in the imperative, with the area in
front:

```
fix(transport): end the phase at its deadline after a dropped connection
docs: clarify network_margin_ms
```

## Code of conduct

Participation in this project is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE) of this repository.
