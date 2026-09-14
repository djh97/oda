# Retained Study Evidence

This directory contains the immutable records used to calculate the reported
software, model, robustness, and workflow results. All cases and note content
in these files are synthetic.

## Contents

- `evidence/evaluation/` contains the 400-case raw outputs, attempt ledgers,
  run configurations, calculated summaries, case-level metrics, confusion
  matrices, paired comparisons, hosted-model cost record, and the post hoc
  output-reliability analysis.
- `evidence/model/` contains the retained training, validation, base-model,
  checkpoint-inheritance, and smoke-test metadata. Model weights and optimizer
  checkpoints are not included.
- `evidence/protocol/` contains the frozen protocol and its amendments.
- `evidence/robustness/` contains the 20-scenario failure-path assessment.
- `evidence/software/` contains the Python test output retained for the paper.
  The Foundry and Slither reports are tracked under `smart-contracts/`.
- `evidence/workflow/` contains selected records from the completed Sepolia
  workflow, including transactions, cost calculations, and the 30-run
  off-chain benchmark.

The raw prediction records are sufficient to recalculate the reported model
metrics without contacting model providers. Reproducing model inference from
scratch additionally requires the pinned base model, the trained LoRA adapter,
appropriate compute, and access to the hosted comparator. The adapter checksum
and training metadata are retained in
`evidence/model/local_lora_training.json`; the adapter weights are excluded
from this source repository.

Some retained JSON fields preserve execution-time paths from the original
workstation. They are provenance metadata only and are not required for the
offline checks below.

## Verification

From the repository root, verify that no evidence file has changed:

```powershell
python reproducibility/verify_evidence.py
```

Validate the synthetic dataset lineage:

```powershell
cd app
python -m evaluation.validate_synthetic_data
```

Recalculate the reported evaluation summaries and output-reliability results
from the raw prediction files:

```powershell
cd ..
python reproducibility/verify_reported_metrics.py
```

The final command reads the retained records without modifying them. It uses
the analysis implementation preserved in this repository and compares the
recalculated numerical results with the retained summaries.
