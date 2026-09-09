from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

from evaluation.check_pinata_access import check_access


def test_pinata_access_record_is_redacted(tmp_path: Path) -> None:
    response = Mock(status_code=200)
    response.json.return_value = {"message": "Authentication succeeded"}
    output = tmp_path / "pinata.json"
    with patch("evaluation.check_pinata_access.requests.get", return_value=response) as get:
        record = check_access("secret-jwt", output_path=output)

    assert record["success"] is True
    assert record["http_status"] == 200
    text = output.read_text(encoding="utf-8")
    assert "secret-jwt" not in text
    assert "Authentication succeeded" not in text
    assert json.loads(text) == record
    assert get.call_args.kwargs["headers"]["Authorization"] == "Bearer secret-jwt"
