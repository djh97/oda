from __future__ import annotations

from copy import deepcopy

from evaluation.amend_protocol_local_lora import (
    MODEL_REVISION,
    amended_manuscript,
    amended_protocol,
    amended_study_protocol,
)
from src.policy import load_protocol


def test_protocol_amendment_preserves_dataset_and_analysis() -> None:
    original = load_protocol()
    original = deepcopy(original)
    original["version"] = "1.0.0"
    original.pop("amendment_history", None)
    original["model_evaluation"]["provider"] = "OpenAI"
    updated = amended_protocol(original)
    assert updated["version"] == "1.1.0"
    assert updated["dataset"] == original["dataset"]
    assert updated["analysis"] == original["analysis"]
    assert updated["model_evaluation"]["base_model_revision"] == MODEL_REVISION
    assert updated["model_evaluation"]["fine_tuning"]["assistant_only_loss"] is True


def test_study_protocol_replacement_is_bounded_by_headings() -> None:
    source = """Protocol identifier: ODA-SYNTH-MULTIORGAN-1.0
provider upload, or external execution
classical or API-based evaluation
## Model Conditions
old model text
## Outcomes
keep this
- OpenAI's GPT-4o mini model documentation and current self-serve fine-tuning
  deprecation schedule for model availability and the access preflight."""
    updated = amended_study_protocol(source)
    assert "version 1.1.0" in updated
    assert "Qwen/Qwen2.5-1.5B-Instruct" in updated
    assert "## Outcomes\nkeep this" in updated


def test_manuscript_replacement_preserves_surrounding_text() -> None:
    source = """before
The two LLM conditions use the untuned and supervised fine-tuned forms of old text.

Figure~\\ref{fig:benchmark_protocol} summarizes after.
The evaluation follows the prespecified protocol ODA-SYNTH-MULTIORGAN-1.0.
The evaluation followed the prespecified ODA-SYNTH-MULTIORGAN-1.0 protocol.
"""
    updated = amended_manuscript(source)
    assert updated.startswith("before\n")
    assert MODEL_REVISION in updated
    assert "Figure~\\ref{fig:benchmark_protocol} summarizes after." in updated
    assert updated.count("version 1.1.0") == 3
