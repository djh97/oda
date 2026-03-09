# ODA — Organ Donation Allocation (Blockchain + LLM Decision Support)

This repository contains a reproducible prototype for organ donor–recipient matching using:
- **Ethereum smart contracts** (Foundry) for auditable registration, approvals, and match recording
- **IPFS/Pinata** for off-chain storage of donor/recipient records and match rationale (`matchCID`)
- **Baseline scoring + LLM decision support** to incorporate structured clinical factors **and** unstructured clinical notes
- A **local web UI** (FastAPI + Bootstrap) to run an end-to-end pipeline and capture outputs for reporting

> **Important:** This project uses **synthetic data only** (no real patient data).

---

## Repository layout

```
ODA/
├─ smart-contracts/               # Foundry project (authoritative smart contract source)
│  ├─ src/                        # Solidity contracts
│  ├─ test/                       # Foundry unit tests + outputs
│  ├─ security/                   # Slither (and later Mythril) reports
│  ├─ foundry.toml
│  └─ out/, cache/                # build artifacts (usually gitignored)
├─ app/                           # Python app (local UI + pipeline)
│  ├─ src/                        # pipeline modules + scripts
│  ├─ templates/                  # HTML templates (light theme)
│  ├─ static/                     # CSS/JS assets
│  ├─ seed-data/                  # synthetic donor/recipient JSON (committed)
│  ├─ pipeline-output/            # generated CSV/JSON outputs (committed for reproducibility)
│  └─ .env.example                # environment template (DO NOT commit .env)
├─ integration/
│  ├─ abi/                        # exported ABI JSON used by app
│  └─ addresses/                  # deployed contract addresses per network
└─ docs/                          # (optional) paper artifacts / figures later
```

---

## Prerequisites

- **Foundry** (for smart contracts + tests): https://book.getfoundry.sh/
- **Python 3.11+**
- A Sepolia RPC provider (Infura/Alchemy)
- Sepolia test ETH for required EOAs
- Pinata JWT (optional, for uploading match rationale to IPFS)
- OpenAI API key (optional, for calling the fine-tuned model)

---

## Smart contracts (Foundry)

From repo root:

```bash
cd smart-contracts
forge test -vvv
```

Security (Slither) output is stored under:

```
smart-contracts/security/slither/
```

---

## App (Local UI + Pipeline)

### 1) Create & activate venv

From repo root:

```powershell
cd app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Configure environment

Copy and edit:

```powershell
Copy-Item .env.example .env
```

Fill in:
- `SEPOLIA_RPC_URL`
- private keys for seeding + LLM tx signing (Sepolia)
- `PINATA_JWT` (optional)
- `OPENAI_API_KEY` and `OPENAI_MODEL_ID` (optional)

> **Never commit `.env`.** It contains secrets.

### 3) Seed Sepolia (entities + donor/recipients + ethics approvals)

```powershell
python -m src.seed_sepolia
```

This will:
- register entities (regulator)
- pre-register donor/recipient addresses (regulator)
- register donor/recipients with IPFS CIDs (hospital)
- apply ethical approvals (ethics committee)
- log transaction receipts (observed gas)

### 4) Run the UI locally

```powershell
uvicorn src.main:app --reload
```

Open:

- UI: http://127.0.0.1:8000/
- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

---

## Outputs produced (for reporting)

Generated files are saved under:

```
app/pipeline-output/
```

Common artifacts:
- `match_runs.csv` — end-to-end pipeline runs (baseline + LLM + on-chain match)
- `tx_log.csv` — observed on-chain receipts (tx hash + gas used)
- `benchmarks/` — latency benchmarks (off-chain and chain)
- `final_cost_table.csv/json` — observed-receipt cost table (multi-chain USD based on provided base fees)

---

## Reproducibility notes

- Seed patient JSON is committed under `app/seed-data/`.
- CIDs in `.env` may be stub or real IPFS objects depending on your run; the system is designed to work with either.
- For publication screenshots, the recommended minimal set is:
  1) **Dashboard end-to-end result** (baseline table + LLM decision + on-chain record)
  2) **Etherscan event log** showing `MatchCreated(matchId, donorId, primaryRecipientId, backupRecipientId, matchCID)`

---

## License

MIT
