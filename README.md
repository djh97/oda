# Enhancing Organ Donation and Transplantation Workflows using Blockchain and LLMs

This repository contains the research prototype and reproducibility materials
for a synthetic study of organ donation and transplantation workflows. The
system combines organ-specific deterministic ranking, language-model review of
candidate notes, a deterministic selection guard, encrypted off-chain storage,
and role-gated workflow governance on Ethereum.

The language model does not rank candidates or select recipients. It assigns a
controlled readiness state to each synthetic note. Deterministic code preserves
the structured-data order and skips only candidates classified as temporarily
on hold. The software is a research prototype, not a clinical allocation system
or an implementation of an operational transplant policy.

## Repository contents

```text
.
|-- app/
|   |-- evaluation/       # Data generation, model evaluation, and analysis
|   |-- protocols/        # Frozen synthetic study protocol
|   |-- seed-data/        # Synthetic end-to-end demonstration case
|   |-- src/              # Ranking, guard, storage, blockchain, and web code
|   |-- templates/        # Browser interface
|   `-- tests/            # Python test suite
|-- datasets/
|   `-- synthetic-v1/     # Training, validation, and held-out test data
|-- integration/          # Contract ABI and Sepolia deployment metadata
|-- smart-contracts/
|   |-- docs/             # Test objectives and expected outcomes
|   |-- script/           # Deployment script
|   |-- security/         # Slither reports
|   |-- src/              # Solidity contract
|   |-- test/             # Foundry test suite
|   `-- test-output/      # Retained human-readable test reports
|-- LICENSE
`-- README.md
```

Generated run artifacts, model weights, local environments, credentials, and
manuscript files are intentionally excluded from version control.

## Requirements

- Python 3.11 or newer
- A CUDA-capable environment for local Qwen inference and LoRA training
- Foundry for Solidity compilation, deployment, and testing
- Slither for static analysis
- Access to an Ethereum Sepolia RPC endpoint
- Pinata credentials for the encrypted IPFS demonstration
- An OpenAI API key only when reproducing the hosted comparator

The exact Python package snapshot used in the study is recorded in
`app/requirements-lock.txt`.

## Installation

```powershell
cd app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -r requirements-ml.txt
```

Create the local configuration file from the example and populate only the
services needed for the intended command.

```powershell
Copy-Item .env.example .env
python -m evaluation.generate_encryption_key
```

`app/.env` contains credentials and private keys and must never be committed.
The repository-level `.gitignore` excludes it at every directory depth.

## Synthetic dataset

The `ODA-SYNTH-MULTIORGAN-1.0` protocol defines 1,600 training cases, 320
validation cases, and 400 held-out test cases distributed equally across
kidney, liver, heart, and lung workflows. Every case, profile, note, label, and
identifier is synthetic.

Validate the retained dataset with

```powershell
cd app
python -m evaluation.validate_synthetic_data
```

To recreate it from the frozen protocol, run

```powershell
python -m evaluation.generate_synthetic_data
python -m evaluation.validate_synthetic_data
```

## Model evaluation

The study compares a TF-IDF logistic model, unmodified
`Qwen/Qwen2.5-1.5B-Instruct`, a locally trained LoRA adaptation of Qwen, and
`gpt-4o-mini-2024-07-18`. All text models use the same note-review task and
output schema. The structured ranker and deterministic guard are unchanged
across conditions.

The principal commands are

```powershell
cd app
python -m evaluation.prepare_local_model
python -m evaluation.preflight_local_lora
python -m evaluation.train_local_lora
python -m evaluation.finalize_local_protocol
python -m evaluation.run_classical_baseline --validation-only
python -m evaluation.run_model_evaluation --condition untuned
python -m evaluation.run_model_evaluation --condition fine_tuned
python -m evaluation.run_model_evaluation --condition openai
```

Held-out evaluation is governed by the frozen protocol and append-only attempt
records. Review the command help and the protocol before starting a new run.
Generated outputs are written below `app/pipeline-output/` and are not tracked.

## End-to-end Sepolia workflow

The demonstration deploys a fresh contract, registers synthetic actors and
encrypted profile references, performs deterministic ranking and note review,
records the guarded pair, and exercises the ordered approval and finalization
path.

```powershell
cd app
python -m evaluation.check_environment --mode sepolia
python -m evaluation.paper_full_workflow
python -m evaluation.benchmark_offchain --measured-runs 30 --warmup-runs 3
```

Sepolia test ETH has no monetary value. Any USD figures produced by the cost
script are explicit arithmetic scenarios based on supplied token and gas-price
assumptions, not fees measured on other networks.

## Application

For a local demonstration using a separately seeded contract

```powershell
cd app
python -m evaluation.seed_only
python -m uvicorn src.main:app --reload
```

Open `http://127.0.0.1:8000/` after the server starts.

## Verification

Run the Python suite with

```powershell
cd app
python -m pytest -q
```

In a clean checkout, tests that verify the retained experiment bundle or the
external manuscript are reported as skipped. With those local artifacts in
their original locations, the same command runs the complete study-verification
suite.

Run the Solidity suite with

```powershell
cd smart-contracts
forge test
```

The purpose, setup, and expected result of every Foundry test are documented in
`smart-contracts/docs/FOUNDRY_TEST_COVERAGE.md`. Retained Foundry and Slither
reports are available under `smart-contracts/test-output/` and
`smart-contracts/security/slither/`.

## Privacy and security boundary

The contract stores role bindings, state flags, synthetic identifiers, and
content identifiers. Profile attributes and note text are encrypted with
AES-256-GCM before upload. A production system would additionally require
institutional identity management, managed key custody, privacy review,
independent security assessment, and clinical governance.

## License

This project is licensed under the MIT License. See `LICENSE`.
