"""Retrieve an argmax Number payload through bounded Attention outputs.

Use ``Attention(n_outputs=candidates)`` when a tested fixed-width route is
preferable to pass-through. The output slots are anonymous joint summaries,
not one declared slot per candidate. Rotating payloads relative to scores must
destroy the learned behavior. See ``__init__.py`` for the complete guide,
current status, remaining work, and promotion criteria.

Example
-------
```yaml
input:
  configuration:
    root_reduction: Attention(n_outputs=3)
    items_reduction: Attention()
  items:
    - {score: -0.4, payload: 2.0}
    - {score: 1.7, payload: -3.0}
    - {score: 0.2, payload: 5.0}
expected_output:
  answer: -3.0  # payload paired with the highest score
control:
  change: rotate payloads while retaining their marginal distribution
  items:
    - {score: -0.4, payload: 5.0}
    - {score: 1.7, payload: 2.0}
    - {score: 0.2, payload: -3.0}
  retained_answer: -3.0
expected_control_output:
  behavior: the retained answer is no longer recoverable from the argmax pair
```

Run with::

    uv run pytest -n 0 proofs/relational/argmax_retrieval/test_attention_number_payload.py -q
"""

import numpy as np
import pytest

from proofs.relational.argmax_retrieval.support import fit, predictions, records, rmse

pytestmark = pytest.mark.proof


def test_multi_output_attention_retrieves_number_payload_at_argmax_coordinate() -> None:
    seed = 23
    train = records(rows=4096, seed=seed + 1)
    validate = records(rows=1024, seed=seed + 2)
    test = records(rows=2048, seed=seed + 3)
    broken = records(rows=2048, seed=seed + 3, break_pairs=True)

    model = fit("summarized", train, validate, seed=seed)
    actual = test["answer"].combine_chunks().to_numpy(zero_copy_only=False)
    predicted = predictions(model, test)
    broken_predicted = predictions(model, broken)
    train_mean = train["answer"].combine_chunks().to_numpy(zero_copy_only=False).mean()
    baseline_rmse = rmse(actual, np.full_like(actual, train_mean))
    normalized = rmse(actual, predicted) / baseline_rmse
    broken_normalized = rmse(actual, broken_predicted) / baseline_rmse

    assert normalized <= 0.35, (
        "candidate-count Attention did not preserve enough retrieval evidence: "
        f"nrmse={normalized:.4f}, rmse={rmse(actual, predicted):.4f}, baseline={baseline_rmse:.4f}"
    )
    assert broken_normalized >= 0.90 and broken_normalized >= normalized + 0.35, (
        "breaking score/payload alignment did not remove multi-output retrieval skill: "
        f"paired={normalized:.4f}, broken={broken_normalized:.4f}, expected broken >= 0.90"
    )
