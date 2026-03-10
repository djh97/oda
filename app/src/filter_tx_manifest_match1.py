import csv
from pathlib import Path

INPUT = Path(r"C:\Users\Ahmed\ODA_Foundry_DJH\app\pipeline-output\tx_manifest.csv")
OUTPUT = Path(r"C:\Users\Ahmed\ODA_Foundry_DJH\app\pipeline-output\tx_manifest_match1.csv")

MATCH_ID = "1"

# These categories are always included (setup + onboarding, including all 10 recipients)
ALWAYS_INCLUDE_CATEGORIES = {
    "Deployment",
    "Governance setup",
    "Identity binding",
    "Profile registration",
    "Eligibility approvals",
}

# These functions are match-specific and should be filtered to matchId=1
MATCH_SCOPED_FUNCTIONS = {
    "createMatch",
    "approveMedicalTeam",
    "approveHospital",
    "approveDonor",
    "approveRecipient",
    "approveFinalTransplant",
    "finalizeMatch",
}

def label_has_match_id(label: str, match_id: str) -> bool:
    """
    Your manifest labels typically include: matchId=#
    We'll treat it as a substring check.
    """
    label = (label or "").replace(" ", "")
    return f"matchId={match_id}" in label

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input manifest not found: {INPUT}")

    with INPUT.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError("Input manifest is empty")

    fieldnames = rows[0].keys()
    kept = []
    dropped_match_scoped = 0

    for r in rows:
        category = (r.get("category") or "").strip()
        function = (r.get("function") or "").strip()
        label = (r.get("label") or "").strip()

        # Always keep all setup/onboarding txs
        if category in ALWAYS_INCLUDE_CATEGORIES:
            kept.append(r)
            continue

        # For match workflow txs, keep only matchId=1
        if function in MATCH_SCOPED_FUNCTIONS:
            if label_has_match_id(label, MATCH_ID):
                kept.append(r)
            else:
                dropped_match_scoped += 1
            continue

        # Any other rows: keep by default (safe), but you can tighten if you prefer.
        kept.append(r)

    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept)

    print("Wrote filtered manifest:")
    print(f"  {OUTPUT}")
    print(f"Rows in:  {len(rows)}")
    print(f"Rows out: {len(kept)}")
    print(f"Dropped match-scoped rows (not matchId=1): {dropped_match_scoped}")

if __name__ == "__main__":
    main()