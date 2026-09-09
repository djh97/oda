"""Deploy and seed a demonstration contract for the local interface."""

from evaluation.paper_full_workflow import main as workflow_main


if __name__ == "__main__":
    workflow_main(["--stop-after-seed"])
