# Organ Donation Allocation (ODA) — Blockchain + LLM Decision Support

This repository contains an end-to-end prototype for organ donation allocation using:
- **Ethereum smart contracts** (Foundry) for transparent, auditable workflow control
- **Off-chain storage (IPFS/Pinata)** for medical data and match rationales
- **LLM decision support** to incorporate **unstructured clinical notes** (risk flags / contraindications) alongside a transparent baseline scoring shortlist
- A **local FastAPI UI** for running the pipeline and generating reproducible evaluation artifacts

> ⚠️ This project uses **synthetic data only** (see `app/seed-data/`).

---

## Repository Structure

- `smart-contracts/`  
  Foundry project (Solidity contract + tests). Build artifacts (`out/`, `cache/`) are ignored.

- `integration/`  
  Exported ABI and deployed contract addresses used by the app:
  - `integration/abi/TransplantManagement.json`
  - `integration/addresses/sepolia.json`

- `app/`  
  Local FastAPI UI + pipeline scripts + saved evaluation outputs:
  - `app/src/` — pipeline + API
  - `app/templates/` — UI
  - `app/seed-data/` — synthetic donor/recipient JSON
  - `app/pipeline-output/` — CSV/JSON outputs (benchmarks, costs, tx logs)

---

## Smart Contract (Foundry)

### Run tests
From `smart-contracts/`:

```bash
forge test -vvv
