import hashlib
import json
import uuid
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from evaluation import manage_fine_tuning as fine_tuning


@pytest.fixture(autouse=True)
def _approved_test_seed():
    with patch.object(
        fine_tuning,
        "assert_test_seed_not_retired",
        return_value=2026090791,
    ):
        yield


class _Models:
    def __init__(self, model=None, error=None):
        self.model = model
        self.error = error

    def retrieve(self, model_id):
        if self.error:
            raise self.error
        assert model_id == fine_tuning.DEFAULT_BASE_MODEL
        return self.model


class _AccessClient:
    def __init__(self, model=None, error=None):
        self.models = _Models(model=model, error=error)


class _Content:
    def __init__(self, value: bytes):
        self.value = value

    def read(self):
        return self.value


class _Files:
    def retrieve(self, file_id):
        return SimpleNamespace(
            filename="metrics.csv",
            model_dump=lambda mode: {"id": file_id, "filename": "metrics.csv"},
        )

    def content(self, file_id):
        return _Content(f"result for {file_id}\n".encode("ascii"))


class _ProviderObject:
    def __init__(self, **values):
        self._values = values
        for key, value in values.items():
            setattr(self, key, value)

    def model_dump(self, mode):
        assert mode == "json"
        return dict(self._values)


class _Page:
    def __init__(self, data):
        self.data = data
        self.has_more = False

    def iter_pages(self):
        yield self


class _UploadFiles:
    def __init__(self, records):
        self.records = records
        self.create_calls = []

    def create(self, *, file, purpose, extra_headers):
        name = Path(file.name).name
        self.create_calls.append(
            {"filename": name, "purpose": purpose, "extra_headers": extra_headers}
        )
        return _ProviderObject(**self.records[name])


class _Jobs:
    def __init__(self, *, recovered=None, create_error=None):
        self.recovered = recovered
        self.create_error = create_error
        self.list_calls = []
        self.create_calls = []

    def list(self, *, limit, metadata):
        self.list_calls.append({"limit": limit, "metadata": metadata})
        return _Page([] if self.recovered is None else [self.recovered])

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        if self.create_error is not None:
            raise self.create_error
        hyperparameters = kwargs["method"]["supervised"]["hyperparameters"]
        return _ProviderObject(
            id="ftjob-created",
            object="fine_tuning.job",
            status="validating_files",
            fine_tuned_model=None,
            model=kwargs["model"],
            training_file=kwargs["training_file"],
            validation_file=kwargs["validation_file"],
            seed=kwargs["seed"],
            hyperparameters=hyperparameters,
            method={"type": "supervised"},
            metadata=kwargs["metadata"],
            result_files=[],
        )


class _FineTuning:
    def __init__(self, jobs):
        self.jobs = jobs


class _FineTuningClient:
    def __init__(self, files, jobs):
        self.models = _Models(
            model=_ProviderObject(
                id=fine_tuning.DEFAULT_BASE_MODEL,
                owned_by="openai",
                shutdown_date=None,
            )
        )
        self.files = files
        self.fine_tuning = _FineTuning(jobs)


class _DefinitiveDenial(RuntimeError):
    status_code = 403
    request_id = "req-redacted-test"
    code = "training_not_available"


def _creation_paths(tmp_path):
    output_dir = tmp_path / "model"
    train_path = tmp_path / "fine_tuning_training.jsonl"
    validation_path = tmp_path / "fine_tuning_validation.jsonl"
    manifest_path = tmp_path / "generation_manifest.json"
    train_path.write_text('{"training": true}\n', encoding="utf-8")
    validation_path.write_text('{"validation": true}\n', encoding="utf-8")
    manifest_path.write_text("{}\n", encoding="utf-8")
    records = {
        train_path.name: {
            "id": "file-training",
            "object": "file",
            "purpose": "fine-tune",
            "filename": train_path.name,
            "bytes": train_path.stat().st_size,
        },
        validation_path.name: {
            "id": "file-validation",
            "object": "file",
            "purpose": "fine-tune",
            "filename": validation_path.name,
            "bytes": validation_path.stat().st_size,
        },
    }
    return {
        "output_dir": output_dir,
        "state_path": output_dir / "fine_tuning_job.json",
        "access_path": output_dir / "fine_tuning_access_check.json",
        "train_path": train_path,
        "validation_path": validation_path,
        "manifest_path": manifest_path,
        "records": records,
    }


def _recovered_job(state):
    request = state["request_without_file_ids"]
    return _ProviderObject(
        id="ftjob-recovered",
        object="fine_tuning.job",
        status="queued",
        fine_tuned_model=None,
        model=request["model"],
        training_file=state["training_file"]["id"],
        validation_file=state["validation_file"]["id"],
        seed=request["seed"],
        hyperparameters=request["method"]["supervised"]["hyperparameters"],
        method={"type": "supervised"},
        metadata=request["metadata"],
        result_files=[],
    )


def test_access_check_confirms_base_model_without_claiming_job_authorization(tmp_path):
    output_dir = tmp_path / "model"
    access_path = output_dir / "fine_tuning_access_check.json"
    state_path = output_dir / "fine_tuning_job.json"
    model = SimpleNamespace(
        model_dump=lambda mode: {
            "id": fine_tuning.DEFAULT_BASE_MODEL,
            "owned_by": "openai",
            "shutdown_date": None,
            "unrelated_provider_field": "not archived",
        }
    )
    client = _AccessClient(model=model)
    blocked_client = SimpleNamespace(models=MagicMock())
    with patch.object(
        fine_tuning,
        "assert_test_seed_not_retired",
        side_effect=ValueError("retired test seed"),
    ), pytest.raises(ValueError, match="retired test seed"):
        fine_tuning.check_model_access(blocked_client, fine_tuning.DEFAULT_BASE_MODEL)
    blocked_client.models.retrieve.assert_not_called()

    with patch.object(fine_tuning, "OUTPUT_DIR", output_dir), patch.object(
        fine_tuning, "ACCESS_CHECK_PATH", access_path
    ), patch.object(fine_tuning, "STATE_PATH", state_path):
        report = fine_tuning.check_model_access(client, fine_tuning.DEFAULT_BASE_MODEL)

    assert report["model_retrievable"] is True
    assert report["model_id_matches"] is True
    assert report["model_owned_by"] == "openai"
    assert report["fine_tuning_job_authorization_confirmed"] is False
    saved = json.loads(access_path.read_text(encoding="utf-8"))
    assert saved["model_id_returned"] == fine_tuning.DEFAULT_BASE_MODEL
    assert "unrelated_provider_field" not in saved


def test_access_check_fails_closed_when_model_cannot_be_retrieved(tmp_path):
    output_dir = tmp_path / "model"
    access_path = output_dir / "fine_tuning_access_check.json"
    state_path = output_dir / "fine_tuning_job.json"
    with patch.object(fine_tuning, "OUTPUT_DIR", output_dir), patch.object(
        fine_tuning, "ACCESS_CHECK_PATH", access_path
    ), patch.object(fine_tuning, "STATE_PATH", state_path):
        report = fine_tuning.check_model_access(
            _AccessClient(error=PermissionError("model unavailable")),
            fine_tuning.DEFAULT_BASE_MODEL,
        )

    assert report["model_retrievable"] is False
    assert report["model_id_matches"] is False
    assert report["fine_tuning_job_authorization_confirmed"] is False
    assert report["error_details"] == {"type": "PermissionError"}
    assert "model unavailable" not in access_path.read_text(encoding="utf-8")


def test_access_check_does_not_replace_evidence_after_state_creation(tmp_path):
    output_dir = tmp_path / "model"
    access_path = output_dir / "fine_tuning_access_check.json"
    state_path = output_dir / "fine_tuning_job.json"
    output_dir.mkdir(parents=True)
    access_path.write_text('{"preserved": true}\n', encoding="utf-8")
    state_path.write_text('{"job": {"id": "ftjob-existing"}}\n', encoding="utf-8")
    client = SimpleNamespace(models=MagicMock())

    with patch.object(fine_tuning, "OUTPUT_DIR", output_dir), patch.object(
        fine_tuning, "ACCESS_CHECK_PATH", access_path
    ), patch.object(fine_tuning, "STATE_PATH", state_path), pytest.raises(
        RuntimeError, match="state already exists"
    ):
        fine_tuning.check_model_access(client, fine_tuning.DEFAULT_BASE_MODEL)

    client.models.retrieve.assert_not_called()
    assert json.loads(access_path.read_text(encoding="utf-8")) == {"preserved": True}


def test_result_files_are_hashed_and_written_back_to_job_state(tmp_path):
    output_dir = tmp_path / "model"
    state_path = output_dir / "fine_tuning_job.json"
    state = {"job": {"result_files": ["file-result-1"]}}
    client = SimpleNamespace(files=_Files())
    with patch.object(fine_tuning, "OUTPUT_DIR", output_dir), patch.object(
        fine_tuning, "STATE_PATH", state_path
    ):
        updated = fine_tuning.archive_result_files(client, state)

    content = b"result for file-result-1\n"
    archived = updated["archived_result_files"][0]
    assert archived["bytes"] == len(content)
    assert archived["sha256"] == hashlib.sha256(content).hexdigest()
    assert Path(archived["local_path"]).read_bytes() == content
    assert json.loads(state_path.read_text(encoding="utf-8"))["archived_result_files"] == [archived]

    retained_files = SimpleNamespace(
        retrieve=MagicMock(side_effect=AssertionError("provider read was repeated")),
        content=MagicMock(side_effect=AssertionError("provider read was repeated")),
    )
    with patch.object(fine_tuning, "OUTPUT_DIR", output_dir), patch.object(
        fine_tuning, "STATE_PATH", state_path
    ):
        repeated = fine_tuning.archive_result_files(
            SimpleNamespace(files=retained_files),
            updated,
        )
    assert repeated == updated
    retained_files.retrieve.assert_not_called()
    retained_files.content.assert_not_called()

    linked_state = {
        "local_run_id": "a" * 32,
        "base_model_requested": fine_tuning.DEFAULT_BASE_MODEL,
        "training_file": {
            "id": "file-training",
            "purpose": "fine-tune",
            "filename": fine_tuning.TRAIN_PATH.name,
            "bytes": fine_tuning.TRAIN_PATH.stat().st_size,
        },
        "validation_file": {
            "id": "file-validation",
            "purpose": "fine-tune",
            "filename": fine_tuning.VALIDATION_PATH.name,
            "bytes": fine_tuning.VALIDATION_PATH.stat().st_size,
        },
    }
    provider_job = {
        "id": "ftjob-test",
        "object": "fine_tuning.job",
        "model": fine_tuning.DEFAULT_BASE_MODEL,
        "training_file": "file-training",
        "validation_file": "file-validation",
        "seed": fine_tuning.MODEL_SETTINGS["seed"],
        "hyperparameters": {"n_epochs": fine_tuning.DEFAULT_EPOCHS},
        "method": {"type": "supervised"},
        "metadata": {
            "protocol": fine_tuning.ACTIVE_PROTOCOL["protocol_id"],
            "purpose": "synthetic-note-readiness",
            "local_run_id": "a" * 32,
        },
    }
    assert fine_tuning.validate_provider_job(linked_state, provider_job)["all_passed"] is True
    provider_job["training_file"] = "file-other"
    report = fine_tuning.validate_provider_job(linked_state, provider_job)
    assert report["all_passed"] is False
    assert report["checks"]["training_file_matches"] is False


def test_development_validation_does_not_parse_held_out_cases(tmp_path):
    train_cases = tmp_path / "training_cases.jsonl"
    validation_cases = tmp_path / "validation_cases.jsonl"
    fine_tuning_train = tmp_path / "fine_tuning_training.jsonl"
    fine_tuning_validation = tmp_path / "fine_tuning_validation.jsonl"
    manifest_path = tmp_path / "generation_manifest.json"
    for path in (train_cases, validation_cases, fine_tuning_train, fine_tuning_validation):
        path.write_text("{}\n", encoding="utf-8")
    artifacts = {
        path.name: {"sha256": fine_tuning._sha256(path), "bytes": path.stat().st_size}
        for path in (train_cases, validation_cases, fine_tuning_train, fine_tuning_validation)
    }
    manifest_path.write_text(
        json.dumps(
            {
                "protocol_id": fine_tuning.ACTIVE_PROTOCOL["protocol_id"],
                "protocol_version": fine_tuning.ACTIVE_PROTOCOL["version"],
                "source_inputs": {
                    "protocol_json": {
                        "sha256": fine_tuning._sha256(fine_tuning.DEFAULT_PROTOCOL_PATH)
                    },
                    "system_prompt_sha256": hashlib.sha256(
                        fine_tuning.SYSTEM_PROMPT.encode("utf-8")
                    ).hexdigest(),
                },
                "artifacts": artifacts,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loaded_paths = []
    real_loader = fine_tuning._load_jsonl

    def tracking_loader(path):
        loaded_paths.append(path)
        return real_loader(path)

    with (
        patch.object(fine_tuning, "TRAIN_CASES_PATH", train_cases),
        patch.object(fine_tuning, "VALIDATION_CASES_PATH", validation_cases),
        patch.object(fine_tuning, "TRAIN_PATH", fine_tuning_train),
        patch.object(fine_tuning, "VALIDATION_PATH", fine_tuning_validation),
        patch.object(fine_tuning, "GENERATION_MANIFEST_PATH", manifest_path),
        patch.object(fine_tuning, "_load_jsonl", side_effect=tracking_loader),
        patch.object(
            fine_tuning,
            "validate_cases",
            return_value={"valid": True, "errors": []},
        ),
    ):
        report = fine_tuning.validate_development_data()

    assert report["valid"] is True
    assert loaded_paths == [train_cases, validation_cases]


def test_create_job_preserves_idempotency_and_local_run_metadata(tmp_path):
    paths = _creation_paths(tmp_path)
    files = _UploadFiles(paths["records"])
    jobs = _Jobs()
    client = _FineTuningClient(files, jobs)
    development_report = {"valid": True, "errors": [], "scope": "development-only"}
    file_validation = {
        "training": {"row_count": 1, "sha256": "train"},
        "validation": {"row_count": 1, "sha256": "validation"},
    }
    fixed_uuid = uuid.UUID("11111111-1111-1111-1111-111111111111")
    with (
        patch.object(fine_tuning, "OUTPUT_DIR", paths["output_dir"]),
        patch.object(fine_tuning, "STATE_PATH", paths["state_path"]),
        patch.object(fine_tuning, "ACCESS_CHECK_PATH", paths["access_path"]),
        patch.object(fine_tuning, "TRAIN_PATH", paths["train_path"]),
        patch.object(fine_tuning, "VALIDATION_PATH", paths["validation_path"]),
        patch.object(fine_tuning, "GENERATION_MANIFEST_PATH", paths["manifest_path"]),
        patch.object(fine_tuning, "validate_development_data", return_value=development_report),
        patch.object(fine_tuning, "validate_fine_tuning_files", return_value=file_validation),
        patch.object(fine_tuning.uuid, "uuid4", return_value=fixed_uuid),
    ):
        state = fine_tuning.create_job(
            client,
            fine_tuning.DEFAULT_BASE_MODEL,
            fine_tuning.DEFAULT_EPOCHS,
        )

    assert state["status"] == "job_created"
    assert state["local_run_id"] == fixed_uuid.hex
    assert len(files.create_calls) == 2
    for call in files.create_calls:
        assert call["extra_headers"]["Idempotency-Key"] == call["extra_headers"][
            "X-Client-Request-Id"
        ]
    assert jobs.list_calls == [{"limit": 100, "metadata": {"local_run_id": fixed_uuid.hex}}]
    assert len(jobs.create_calls) == 1
    creation_call = jobs.create_calls[0]
    assert creation_call["metadata"]["local_run_id"] == fixed_uuid.hex
    assert creation_call["extra_headers"]["Idempotency-Key"] == state["request_ids"][
        "job_creation"
    ]
    assert state["job_creation_attempts"][-1]["outcome"] == "provider_response_received"


def test_uncertain_job_creation_recovers_without_second_paid_job(tmp_path):
    paths = _creation_paths(tmp_path)
    development_report = {"valid": True, "errors": [], "scope": "development-only"}
    file_validation = {
        "training": {"row_count": 1, "sha256": "train"},
        "validation": {"row_count": 1, "sha256": "validation"},
    }
    first_files = _UploadFiles(paths["records"])
    first_jobs = _Jobs(create_error=TimeoutError("sensitive provider transport detail"))
    first_client = _FineTuningClient(first_files, first_jobs)
    fixed_uuid = uuid.UUID("22222222-2222-2222-2222-222222222222")
    common_patches = [
        patch.object(fine_tuning, "OUTPUT_DIR", paths["output_dir"]),
        patch.object(fine_tuning, "STATE_PATH", paths["state_path"]),
        patch.object(fine_tuning, "ACCESS_CHECK_PATH", paths["access_path"]),
        patch.object(fine_tuning, "TRAIN_PATH", paths["train_path"]),
        patch.object(fine_tuning, "VALIDATION_PATH", paths["validation_path"]),
        patch.object(fine_tuning, "GENERATION_MANIFEST_PATH", paths["manifest_path"]),
        patch.object(fine_tuning, "validate_development_data", return_value=development_report),
        patch.object(fine_tuning, "validate_fine_tuning_files", return_value=file_validation),
        patch.object(fine_tuning.uuid, "uuid4", return_value=fixed_uuid),
    ]
    with ExitStack() as stack:
        for context in common_patches:
            stack.enter_context(context)
        with pytest.raises(RuntimeError, match="uncertain outcome"):
            fine_tuning.create_job(
                first_client,
                fine_tuning.DEFAULT_BASE_MODEL,
                fine_tuning.DEFAULT_EPOCHS,
            )

    persisted = json.loads(paths["state_path"].read_text(encoding="utf-8"))
    assert persisted["status"] == "job_creation_uncertain"
    assert persisted["job_creation_attempts"][-1]["error_details"] == {
        "type": "TimeoutError"
    }
    assert "sensitive provider transport detail" not in paths["state_path"].read_text(
        encoding="utf-8"
    )

    malformed = {**persisted, "job_creation_attempts": {}}
    with (
        patch.object(fine_tuning, "ACCESS_CHECK_PATH", paths["access_path"]),
        patch.object(fine_tuning, "TRAIN_PATH", paths["train_path"]),
        patch.object(fine_tuning, "VALIDATION_PATH", paths["validation_path"]),
        patch.object(fine_tuning, "GENERATION_MANIFEST_PATH", paths["manifest_path"]),
        pytest.raises(RuntimeError, match="job_creation_attempts"),
    ):
        fine_tuning._validate_resume_state(
            malformed,
            malformed["request_without_file_ids"],
            development_report,
            file_validation,
        )

    second_files = _UploadFiles(paths["records"])
    second_jobs = _Jobs(recovered=_recovered_job(persisted))
    second_client = _FineTuningClient(second_files, second_jobs)
    with (
        patch.object(fine_tuning, "OUTPUT_DIR", paths["output_dir"]),
        patch.object(fine_tuning, "STATE_PATH", paths["state_path"]),
        patch.object(fine_tuning, "ACCESS_CHECK_PATH", paths["access_path"]),
        patch.object(fine_tuning, "TRAIN_PATH", paths["train_path"]),
        patch.object(fine_tuning, "VALIDATION_PATH", paths["validation_path"]),
        patch.object(fine_tuning, "GENERATION_MANIFEST_PATH", paths["manifest_path"]),
        patch.object(fine_tuning, "validate_development_data", return_value=development_report),
        patch.object(fine_tuning, "validate_fine_tuning_files", return_value=file_validation),
    ):
        recovered = fine_tuning.create_job(
            second_client,
            fine_tuning.DEFAULT_BASE_MODEL,
            fine_tuning.DEFAULT_EPOCHS,
        )

    assert recovered["status"] == "job_recovered"
    assert recovered["job"]["id"] == "ftjob-recovered"
    assert recovered["job_creation_attempts"][-1]["outcome"] == "provider_job_recovered"
    assert second_files.create_calls == []
    assert second_jobs.create_calls == []


def test_definitive_job_rejection_is_terminal_and_not_replayed(tmp_path):
    paths = _creation_paths(tmp_path)
    files = _UploadFiles(paths["records"])
    jobs = _Jobs(create_error=_DefinitiveDenial("sensitive provider detail"))
    client = _FineTuningClient(files, jobs)
    development_report = {"valid": True, "errors": []}
    file_validation = {
        "training": {"row_count": 1, "sha256": "train"},
        "validation": {"row_count": 1, "sha256": "validation"},
    }
    patches = (
        patch.object(fine_tuning, "OUTPUT_DIR", paths["output_dir"]),
        patch.object(fine_tuning, "STATE_PATH", paths["state_path"]),
        patch.object(fine_tuning, "ACCESS_CHECK_PATH", paths["access_path"]),
        patch.object(fine_tuning, "TRAIN_PATH", paths["train_path"]),
        patch.object(fine_tuning, "VALIDATION_PATH", paths["validation_path"]),
        patch.object(fine_tuning, "GENERATION_MANIFEST_PATH", paths["manifest_path"]),
        patch.object(fine_tuning, "validate_development_data", return_value=development_report),
        patch.object(fine_tuning, "validate_fine_tuning_files", return_value=file_validation),
    )
    with ExitStack() as stack:
        for context in patches:
            stack.enter_context(context)
        with pytest.raises(RuntimeError, match="definitively rejected"):
            fine_tuning.create_job(
                client,
                fine_tuning.DEFAULT_BASE_MODEL,
                fine_tuning.DEFAULT_EPOCHS,
            )

    state = json.loads(paths["state_path"].read_text(encoding="utf-8"))
    assert state["status"] == "job_creation_rejected"
    assert state["job_creation_attempts"][-1]["outcome"] == "provider_rejected"
    assert state["job_creation_attempts"][-1]["error_details"]["provider_code"] == (
        "training_not_available"
    )
    assert "sensitive provider detail" not in paths["state_path"].read_text(encoding="utf-8")

    with ExitStack() as stack:
        for context in patches:
            stack.enter_context(context)
        with pytest.raises(RuntimeError, match="refusing another creation request"):
            fine_tuning.create_job(
                client,
                fine_tuning.DEFAULT_BASE_MODEL,
                fine_tuning.DEFAULT_EPOCHS,
            )
    assert len(jobs.create_calls) == 1


def test_wait_rejects_nonpositive_poll_interval():
    with pytest.raises(ValueError, match="positive"):
        fine_tuning.wait_for_job(SimpleNamespace(), 0)

    jobs = MagicMock()
    client = SimpleNamespace(fine_tuning=SimpleNamespace(jobs=jobs))
    with (
        patch.object(fine_tuning, "_load_state", return_value={"local_run_id": "invalid"}),
        patch.object(
            fine_tuning,
            "validate_development_data",
            return_value={"valid": True, "errors": []},
        ),
        patch.object(
            fine_tuning,
            "validate_fine_tuning_files",
            return_value={"training": {}, "validation": {}},
        ),
        pytest.raises(RuntimeError, match="stale or malformed"),
    ):
        fine_tuning.refresh_job(client)
    jobs.retrieve.assert_not_called()
