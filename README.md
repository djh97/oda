# Organ Donation Allocation (ODA) — Blockchain + LLM Decision Support

This repository contains an end-to-end prototype for organ donation allocation using:
- **Ethereum smart contracts** (Foundry) for transparent, auditable workflow control
- **Off-chain storage (IPFS/Pinata)** for medical data and match rationales
- **LLM decision support** to incorporate **unstructured clinical notes** (risk flags / contraindications) alongside a transparent baseline scoring shortlist
- A **local FastAPI UI** for running the pipeline and generating reproducible evaluation artifacts

> ⚠️ This project uses **synthetic data only** (see `app/seed-data/`).


