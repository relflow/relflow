"""Class declarations retain branch configuration and model factory contracts."""

from typing import assert_type

import relflow as rf


class Transaction(rf.Branch):
    amount = rf.Number
    merchant: rf.TreeFieldInput = rf.Category


class NormalizedTransaction(Transaction):
    merchant: rf.TreeFieldInput = rf.Category(cased=False)


class FraudDetector(rf.Model):
    account = rf.Hash
    transactions = Transaction(length=64, overflow="tail")
    fraudulent = rf.Boolean(mask=True)


def declared_api(schema: rf.Schema) -> None:
    branch = Transaction(length=32, overflow="random")
    assert_type(branch, Transaction)
    assert_type(branch.length, int | None)
    assert_type(NormalizedTransaction(length=8), NormalizedTransaction)
    assert_type(FraudDetector(d_model=64, n_layers=2, n_heads=4), FraudDetector)
    assert_type(FraudDetector.xs(), FraudDetector)
    assert_type(FraudDetector.sm(d_model=64), FraudDetector)
    assert_type(FraudDetector(schema), FraudDetector)
    assert_type(FraudDetector.load("fraud.rf"), FraudDetector)
