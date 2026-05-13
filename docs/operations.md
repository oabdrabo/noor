# Operations Runbook

## Local Validation

1. Install dependencies using the package manager already used by the repository.
2. Run formatting or linting commands when they are available.
3. Start the service or UI from the documented entry point.
4. Record any manual verification steps in the related pull request or release note.

## Release Preparation

- Review pending changes on `main`.
- Confirm generated files are not staged accidentally.
- Update `CHANGELOG.md` with the user-facing or operator-facing impact.
- Tag only commits that can be reproduced from a clean checkout.

## Incident Notes

Operational findings should be written down as follow-up tasks, not hidden in local shell history.
