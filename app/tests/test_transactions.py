from types import SimpleNamespace

import pytest
from hexbytes import HexBytes
from web3 import Web3

from src.transactions import TransactionError, send_signed_transaction


SENDER = Web3.to_checksum_address("0x" + "1" * 40)
OTHER = Web3.to_checksum_address("0x" + "2" * 40)
TX_HASH = HexBytes("0x" + "a" * 64)


class _Account:
    def from_key(self, private_key):
        assert private_key == "test-private-key"
        return SimpleNamespace(address=SENDER)

    def sign_transaction(self, transaction, *, private_key):
        assert private_key == "test-private-key"
        return SimpleNamespace(raw_transaction=b"signed", hash=TX_HASH)


class _Eth:
    def __init__(self, receipt):
        self.account = _Account()
        self.chain_id = 11155111
        self.gas_price = 25_000_000_000
        self.receipt = receipt

    def get_transaction_count(self, sender, state):
        assert sender == SENDER
        assert state == "pending"
        return 7

    def get_block(self, block):
        assert block == "latest"
        return {"baseFeePerGas": 10_000_000_000}

    def send_raw_transaction(self, raw_transaction):
        assert raw_transaction == b"signed"
        return TX_HASH

    def wait_for_transaction_receipt(self, tx_hash, *, timeout, poll_latency):
        assert tx_hash == TX_HASH
        assert timeout == 600.0
        assert poll_latency == 2.0
        return self.receipt


class _W3:
    def __init__(self, receipt):
        self.eth = _Eth(receipt)

    @staticmethod
    def to_wei(value, unit):
        return Web3.to_wei(value, unit)


class _Function:
    def __init__(self, *, estimate=50_000, mutate=None, error=None):
        self.estimate = estimate
        self.mutate = mutate
        self.error = error

    def estimate_gas(self, transaction):
        if self.error is not None:
            raise self.error
        assert transaction == {"from": SENDER}
        return self.estimate

    def build_transaction(self, fields):
        transaction = dict(fields)
        if self.mutate is not None:
            transaction.update(self.mutate)
        return transaction


def _receipt(**updates):
    values = {
        "status": 1,
        "transactionHash": TX_HASH,
        "from": SENDER,
        "gasUsed": 50_000,
        "effectiveGasPrice": 12_000_000_000,
        "blockNumber": 123,
        "contractAddress": None,
    }
    values.update(updates)
    return values


def test_signed_transaction_validates_and_returns_receipt_measurements():
    receipt = send_signed_transaction(
        _W3(_receipt()),
        _Function(),
        "test-private-key",
        role="test role",
    )

    assert receipt.tx_hash == Web3.to_hex(TX_HASH).lower()
    assert receipt.sender == SENDER
    assert receipt.status == 1
    assert receipt.gas_used == 50_000
    assert receipt.effective_gas_price_wei == 12_000_000_000
    assert receipt.block_number == 123
    assert receipt.contract_address is None
    assert receipt.confirmation_seconds >= 0


def test_signed_transaction_journals_irreversible_boundaries_without_raw_bytes():
    events = []

    def record(event_type, details):
        events.append((event_type, dict(details)))

    send_signed_transaction(
        _W3(_receipt()),
        _Function(),
        "test-private-key",
        role="test role",
        event_recorder=record,
        event_context={
            "transaction_id": "tx-001",
            "workflow_stage": "Deployment",
            "contract_function": "constructor",
            "argument_sha256": "c" * 64,
        },
    )

    assert [event_type for event_type, _ in events] == [
        "transaction_prepared",
        "transaction_signed",
        "transaction_submitted",
        "transaction_confirmed",
    ]
    serialized = repr(events)
    assert "test-private-key" not in serialized
    assert "b'signed'" not in serialized
    assert events[1][1]["signed_transaction_hash"] == Web3.to_hex(TX_HASH).lower()
    assert len(events[1][1]["raw_signed_transaction_sha256"]) == 64


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"transactionHash": HexBytes("0x" + "b" * 64)}, "hash differs"),
        ({"from": OTHER}, "sender differs"),
        ({"gasUsed": 70_001}, "gas use is outside"),
        ({"effectiveGasPrice": 0}, "effective gas price"),
        ({"blockNumber": -1}, "block number"),
    ],
)
def test_receipt_inconsistencies_fail_closed(updates, message):
    with pytest.raises(TransactionError, match=message):
        send_signed_transaction(
            _W3(_receipt(**updates)),
            _Function(),
            "test-private-key",
            role="test role",
        )


def test_built_transaction_cannot_change_critical_fields():
    with pytest.raises(TransactionError, match="critical numeric field"):
        send_signed_transaction(
            _W3(_receipt()),
            _Function(mutate={"nonce": 8}),
            "test-private-key",
            role="test role",
        )


def test_rpc_error_details_are_not_copied_into_failure_message():
    with pytest.raises(TransactionError) as captured:
        send_signed_transaction(
            _W3(_receipt()),
            _Function(error=RuntimeError("https://credential.example/secret")),
            "test-private-key",
            role="test role",
        )

    assert "RuntimeError" in str(captured.value)
    assert "credential.example" not in str(captured.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout_seconds": 0},
        {"poll_latency_seconds": 0},
        {"priority_fee_gwei": -1},
        {"gas_multiplier": 0.9},
        {"gas_padding": -1},
        {"timeout_seconds": float("nan")},
    ],
)
def test_invalid_transaction_parameters_are_rejected(kwargs):
    with pytest.raises(TransactionError):
        send_signed_transaction(
            _W3(_receipt()),
            _Function(),
            "test-private-key",
            role="test role",
            **kwargs,
        )
