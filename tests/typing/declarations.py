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


class UnboundedHistory(rf.Branch):
    transactions = Transaction


class UnboundedFraud(rf.Model):
    history = UnboundedHistory
    fraudulent = rf.Boolean(mask=True)


def declared_api(schema: rf.Schema) -> None:
    branch = Transaction(length=32, overflow="random")
    assert_type(branch, Transaction)
    assert_type(branch.length, int | None)
    assert_type(NormalizedTransaction(length=8), NormalizedTransaction)
    assert_type(FraudDetector(d_model=64, n_layers=2, n_heads=4), FraudDetector)
    assert_type(FraudDetector(preset="small"), FraudDetector)
    assert_type(FraudDetector(preset="small", d_model=64), FraudDetector)
    assert_type(FraudDetector(schema), FraudDetector)
    assert_type(FraudDetector.load("fraud.rf"), FraudDetector)
    assert_type(UnboundedFraud(preset="small"), UnboundedFraud)
    assert_type(rf.Model(preset="small", history=Transaction, fraudulent=rf.Boolean(mask=True)), rf.Model)
    rf.Model(d_model=16, n_layers=1, n_heads=2, fields={"history": UnboundedHistory}, target=rf.Boolean(mask=True))
    rf.Branch(length=2, fields={"history": Transaction})
