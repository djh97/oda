from types import SimpleNamespace

import pytest

from src.chain_reader import ChainDataError, get_all_recipients, get_donor


class _Call:
    def __init__(self, value=None, error: Exception | None = None):
        self.value = value
        self.error = error

    def call(self):
        if self.error is not None:
            raise self.error
        return self.value


class _Functions:
    def __init__(self, *, donor=None, recipients=None, count=None, error=None):
        self.donor = donor
        self.recipient_values = recipients or {}
        self.count = len(self.recipient_values) if count is None else count
        self.error = error

    def donors(self, _donor_id):
        return _Call(self.donor, self.error)

    def donorHasOpenMatch(self, _donor_id):
        return _Call(False)

    def recipientCounter(self):
        return _Call(self.count)

    def recipients(self, recipient_id):
        return _Call(self.recipient_values[recipient_id])


def _context(functions):
    return SimpleNamespace(contract=SimpleNamespace(functions=functions))


def test_chain_reader_decodes_exact_contract_shapes() -> None:
    address = "0x" + "1" * 40
    functions = _Functions(
        donor=(1, address, "cid-donor", True, True, False),
        recipients={
            1: (1, address, "cid-recipient", True, True, False, False),
        },
    )
    donor = get_donor(_context(functions), 1)
    recipients = get_all_recipients(_context(functions))

    assert donor["donorId"] == 1
    assert donor["profileCID"] == "cid-donor"
    assert recipients[0]["recipientId"] == 1
    assert recipients[0]["profileCID"] == "cid-recipient"


@pytest.mark.parametrize(
    ("donor", "message"),
    [
        ((1, "0x" + "1" * 40, "cid", True, True), "expected 6"),
        ((2, "0x" + "1" * 40, "cid", True, True, False), "requested ID"),
        ((1, "0x" + "1" * 40, "cid", "true", True, False), "not a Boolean"),
    ],
)
def test_chain_reader_rejects_malformed_donor_records(donor, message) -> None:
    with pytest.raises(ChainDataError, match=message):
        get_donor(_context(_Functions(donor=donor)), 1)


def test_chain_reader_rejects_sparse_recipient_records() -> None:
    address = "0x" + "1" * 40
    functions = _Functions(
        recipients={1: (0, address, "", False, False, False, False)},
    )
    with pytest.raises(ChainDataError, match="requested ID"):
        get_all_recipients(_context(functions))


def test_chain_reader_does_not_expose_provider_error_text() -> None:
    functions = _Functions(error=RuntimeError("sensitive RPC endpoint detail"))
    with pytest.raises(ChainDataError) as raised:
        get_donor(_context(functions), 1)
    assert "sensitive RPC endpoint detail" not in str(raised.value)
