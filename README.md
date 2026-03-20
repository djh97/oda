# ODA - Organ Donation Allocation

This repository contains a reproducible prototype for organ donor-recipient matching using:

- Ethereum smart contracts (Foundry) for auditable registration, approvals, and match recording
- IPFS/Pinata for off-chain storage of donor/recipient records and match rationale
- A deterministic baseline ranker plus LLM decision support
- A local FastAPI web UI for running the matching flow and inspecting outputs

Important: this project uses synthetic data only. No real patient data are included.

## Repository layout

```text
ODA/
|-- smart-contracts/              # Foundry project
|   |-- src/                      # Solidity contracts
|   |-- test/                     # Foundry tests
|   |-- script/                   # Deployment scripts
|   |-- security/                 # Slither reports
|   |-- test-output/              # Saved Foundry traces / output
|   `-- standard-input.json       # Verification-oriented standard JSON input
|-- app/
|   |-- src/                      # FastAPI app and runtime modules
|   |-- evaluation/               # Reproducibility / benchmarking scripts
|   |-- templates/                # HTML templates
|   |-- static/                   # Static assets
|   |-- seed-data/                # Synthetic donor / recipient JSON
|   |-- pipeline-output/          # Generated CSV / JSON artifacts
|   |-- requirements.txt          # Python dependencies
|   `-- .env.example              # Environment template
|-- integration/
|   |-- abi/                      # ABI exported for the app
|   `-- addresses/                # Deployed contract addresses per network
`-- datasets/
    `-- llm-finetuning/           # Fine-tuning and held-out evaluation datasets
```

## Prerequisites

- Foundry
- Python 3.11+
- A Sepolia RPC provider
- Sepolia ETH for the required EOAs
- A Pinata JWT
- An OpenAI API key and model ID for decision support

## Smart contracts

From the repository root:

```powershell
cd smart-contracts
forge test -vvv
```

Security outputs are stored under:

```text
smart-contracts/security/slither/
```

## App setup

### 1) Create and activate a virtual environment

```powershell
cd app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Configure environment variables

```powershell
Copy-Item .env.example .env
```

Fill in at least:

- `NETWORK`
- `SEPOLIA_RPC_URL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL_ID`
- `PINATA_JWT`
- `PINATA_GATEWAY`
- all required private keys for regulator, hospital, ethics committee, medical team, donor, recipients, and the authorized LLM signer
- seed CIDs for the donor and recipient records

Never commit `.env`.

## Main workflows

### Canonical full Sepolia workflow

This is the main end-to-end workflow used for the tracked on-chain artifacts.

```powershell
cd app
python -m evaluation.paper_full_workflow
```

This script:

- deploys a fresh contract
- executes governance setup, identity binding, profile registration, ethical approvals, match creation, approvals, and finalization
- writes the transaction manifest and cost tables
- syncs `integration/abi/TransplantManagement.json`
- syncs `integration/addresses/sepolia.json`
- regenerates `smart-contracts/standard-input.json`

### Seed-only workflow

Use this when you want a clean contract state for UI screenshots or manual matching without running the full approval chain.

```powershell
cd app
python -m evaluation.seed_only
```

This seeds:

- trusted roles
- donor / recipient address binding
- donor / recipient profile registration
- donor / recipient ethical approvals

It does not create a match or finalize anything.

### Run the local UI

```powershell
cd app
python -m uvicorn src.main:app --reload
```

Open:

- UI: `http://127.0.0.1:8000/`
- Health: `http://127.0.0.1:8000/health`

### Optional evaluation scripts

These live under `app/evaluation/`:

```powershell
python -m evaluation.benchmark
python -m evaluation.benchmark_chain
python -m evaluation.make_eval_set
python -m evaluation.evaluate_baseline_vs_llm
```

## Generated outputs

Generated artifacts are saved under:

```text
app/pipeline-output/
```

Common files include:

- `tx_manifest.csv` - workflow transaction manifest
- `tx_log.csv` - observed on-chain transaction receipts
- `match_runs.csv` - UI-driven matching runs
- `final_cost_table.csv`
- `final_cost_table.json`
- `final_cost_table_detailed.csv`
- `seed_runs/` - seed-only run logs
- `timing_runs/` - workflow timing runs

## Reproducibility notes

- Synthetic seed records are committed under `app/seed-data/`.
- Selected generated artifacts are intentionally tracked under `app/pipeline-output/` and `smart-contracts/test-output/`.
- The tracked `integration/abi/` and `integration/addresses/` files reflect the latest synchronized deployment state produced by the workflow scripts.

## License

MIT
