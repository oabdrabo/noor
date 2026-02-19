# Architecture Notes

## Boundaries

- Source code and configuration stay versioned with the service.
- Generated artifacts and local runtime output stay outside committed paths.
- Documentation is updated when workflow or operational expectations change.

## Review Checklist

- Does the change keep startup and configuration paths clear?
- Are failures observable through logs, tests, or documented commands?
- Can a new contributor understand the intended entry point?
- Are environment-specific values kept out of source control?

## Maintenance Notes

The repository favors small changes with narrow scope. Larger changes should be split into setup, implementation, verification, and documentation commits.
