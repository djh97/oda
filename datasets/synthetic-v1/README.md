# Synthetic Multi-Organ Dataset v1

This directory contains the deterministic data artifacts for protocol
`ODA-SYNTH-MULTIORGAN-1.0`. Every donor, candidate, note, label, and identifier
is synthetic. The records are intended only for technical validation and do
not reproduce an official allocation policy or provide clinical ground truth.

Fields prefixed with `synthetic_` are protocol inputs and must not be
interpreted as values from an official allocation calculator. The synthetic
KDPI and EPTS values support threshold scenarios only. The liver urgency and
status fields are not official MELD or Status assignments, the adult heart
status is a simulation tier, and the lung priority score is a supplied index
rather than an official Composite Allocation Score calculation.

## Files

| File | Purpose |
|---|---|
| `training_cases.jsonl` | 1,600 source cases used to build supervised fine-tuning examples |
| `validation_cases.jsonl` | 320 cases used for fine-tuning validation and service smoke tests |
| `test_cases.jsonl` | Current 400-case pre-lock candidate; its exposed seed is retired and this file must be replaced before evaluation |
| `fine_tuning_training.jsonl` | Training messages in the exact runtime prompt and response schema |
| `fine_tuning_validation.jsonl` | Validation messages in the exact runtime prompt and response schema |
| `generation_manifest.json` | Seeds, counts, paths, byte sizes, and SHA-256 hashes |
| `validation_report.json` | Machine-readable invariant and split-leakage checks |

Each source case has one donor and ten candidates. Six candidates are
structurally compatible by construction, and four exercise organ-type, ABO,
crossmatch, or size exclusion paths. The split is balanced across kidney,
liver, heart, and lung cases.

Within every organ, 20% of cases place the baseline top-ranked candidate on a
temporary hold and 10% place both of the first two candidates on temporary
holds. The latter cases force the reference primary to baseline rank three and
exercise multiple-position displacement. The remaining strata cover a
top-ranked review state, an uncomplicated top-ranked candidate, negated or
resolved language at the top rank, and a hold below the top rank. The same
prespecified proportions are represented in all three splits.

Recipient identifiers are randomly assigned across construction positions,
and the candidate list is independently shuffled within every case. The model
therefore cannot infer structural compatibility or generator position from an
identifier suffix or from payload order. Within each study split, the
validator requires every payload position and every identifier suffix to
contain both selectable and excluded candidates as well as examples of all
three note-readiness states.

## Field Groups

Common structured fields include synthetic identifiers, organ type, blood
group, age, weight, height, crossmatch result, waiting time, and distance.
Kidney cases add six HLA antigens, sensitization, threshold-based longevity and
pediatric priority, and synthetic donor and adult-recipient profile values.
Synthetic EPTS is omitted for pediatric kidney candidates. Liver cases add a
synthetic Status-1 indicator and urgency score. Heart cases add a synthetic
adult status tier. Lung cases add a supplied synthetic priority score. Liver,
heart, and lung cases are adult-only.

Every candidate also contains a generated note plus latent
`reference_note_state` and `reference_evidence_codes` fields. Those reference
fields are retained for scoring only. The runtime payload builder sends the
model the synthetic case ID and organ type followed by each `recipient_id` and
`medical_notes` value. Tests fail if latent labels, structured ranking
attributes, scores, or rank positions enter the model payload.

## Generation and Freeze

Do not run the commands below against the current protocol yet. Test seed
`2026090717` was exposed during an offline source audit and is retired. After
author approval, replace the seed and regenerate the candidate test split
without displaying its records. The training and validation seeds remain
unchanged.

From `Implementation/app`, run:

```powershell
python -m evaluation.generate_synthetic_data
python -m evaluation.validate_synthetic_data
```

Generation is deterministic for the versioned seeds. The command refuses to
overwrite the dataset after a fine-tuning state or locked-test state exists.
Any later dataset correction requires a new protocol version, seed set, and
output directory.

## Interpretation

The labels encode a prespecified synthetic readiness taxonomy with
`eligible`, `review_required`, and `temporary_hold` states. They test whether
the software preserves protocol behavior under controlled language variants.
They do not establish allocation quality, transplant outcomes, diagnostic
accuracy, or clinical safety.
