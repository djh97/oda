# Post Hoc Robustness Challenge Coverage

## Purpose

This component-level challenge suite supplements the frozen 400-case model
comparison. It does not regenerate, reopen, or modify the training, validation,
or held-out test datasets. Its purpose is to verify deterministic behavior at
boundary conditions that were intentionally absent from the balanced primary
benchmark and to exercise selected fail-closed application paths.

## Result

All 20 scenarios passed:

| Category | Scenarios | Passed | Failed |
|---|---:|---:|---:|
| Guard boundary conditions | 10 | 10 | 0 |
| Model-output validation | 4 | 4 | 0 |
| Encrypted-storage and gateway failures | 6 | 6 | 0 |
| **Total** | **20** | **20** | **0** |

## Guard Boundary Conditions

| Scenario | Expected and observed behavior |
|---|---|
| Two available candidates | Select the first two candidates in baseline order. |
| Two candidates with one temporary hold | Abstain because fewer than two candidates remain. |
| Two candidates with two temporary holds | Abstain because fewer than two candidates remain. |
| Three candidates with rank one on temporary hold | Select ranks two and three without reordering them. |
| Three candidates with the first two on temporary hold | Abstain because only one candidate remains. |
| Ten candidates with the first eight on temporary hold | Select ranks nine and ten. |
| Twenty candidates with nonadjacent temporary holds | Select the first two remaining candidates in baseline order. |
| One structurally selectable candidate | Abstain before applying note-defined holds. |
| No structurally selectable candidates | Abstain before applying note-defined holds. |
| Rank one marked review required | Retain ranks one and two and preserve the review flag. |

The guard scenarios exercise candidate sets containing 2, 3, 4, 10, and 20
candidate records. They test deterministic guard behavior, not language-model
accuracy at each candidate-set size.

## Model-Output Validation

The strict validation boundary rejected each of the following:

- a missing candidate assessment;
- an assessment for an unexpected candidate;
- a response containing the wrong case identifier; and
- an evidence code incompatible with the reported readiness state.

No rejected output reached guarded selection.

## Encrypted Storage and Gateway Failures

Authenticated decryption rejected modified ciphertext, an incorrect encryption
key, an incorrect record context, and a malformed encryption envelope. Simulated
gateway unavailability was propagated as a storage failure, and an invalid CID
returned by the mocked gateway was rejected.

These are deterministic fault injections. They do not measure uptime or
recovery under a live storage outage and do not simulate theft of an authorized
key.

## Smart-Contract Coverage

The separate 41-test Foundry suite covers contract-level authorization,
reservation conflicts, duplicate approvals, backup promotion and approval
reset, cancellation, and terminal-state rejection. Its detailed matrix is in
`smart-contracts/docs/FOUNDRY_TEST_COVERAGE.md`.

The component challenge suite and Foundry tests do not simulate blockchain
reorganizations, simultaneous transaction races on a live network, sustained
load, or compromise of an authorized account.

## Reproduction

From the `app` directory, run:

```powershell
.\.venv\Scripts\python.exe .\evaluation\run_robustness_challenges.py
```

The command writes the machine-readable result to
`app/pipeline-output/current/robustness/failure_path_challenges.json`. It exits
with a nonzero status if any expected selection, abstention, or rejection is not
observed. No network service is contacted by the challenge runner.
