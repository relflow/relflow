# %% [markdown]
# ---
# title: Can text meaning transfer to unfamiliar phrases?
# categories:
# - Representation
# description: Read a customer request together with an ordinary or inverted routing
#   policy and predict the hidden review flag.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Held-out Text macro F1
#   metric:
#   - arms
#   - joint
#   - macro_f1
#   format: number
# - label: Whole-string Category macro F1
#   metric:
#   - arms
#   - category
#   - macro_f1
#   format: number
# evidence-group: Representation
# evidence-requires:
# - relflow[text]
# - 'EVIDENCE_TEXT_MODEL: pinned local Hugging Face encoder/tokenizer snapshot, available
#   offline'
# ---
#
# An unfamiliar string is not necessarily an unfamiliar meaning. This research experiment asks whether a
# real frozen language encoder helps relflow recognize held-out paraphrases and combine their meaning with a
# second structured input.
#
# ## Example
#
# ```yaml
# request: give my money back
# policy: ordinary
# review: true
# ```
#
# Read a customer request together with an ordinary or inverted routing policy and predict the hidden review
# flag. The example is illustrative; its answer is hidden during scoring.
#
# ```{typst}
# //| label: fig-evidence-p092
# //| fig-cap: "Visible fields supply evidence; prediction targets are hidden."
# //| fig-alt: "Read a customer request together with an ordinary or inverted routing policy and predict the hidden review flag."
# #tree(node("record",kind:"root",children:(node("request",type:"Text",body:[Real frozen encoder]),node("policy",type:"Enum"),node("review",kind:"target",type:"Boolean"))))
# ```
#
# ## Comparison
#
# Refund and cancellation requests receive review under ordinary policy; payment and profile-update requests
# do not. Inverted policy reverses that rule. Balanced policies make either text alone or policy alone
# insufficient. Compare joint Text, whole-string Category, text-only, policy-only, shuffled text, and a
# small head over the same real frozen vectors.
#
# ## Result
#
# {{< evidence P092 status >}}
#
# No result is claimed before a full run is recorded. The checks below are provisional hypotheses, not
# established capabilities. Lower error and proper loss are better; larger accuracy is better. Controls and
# each evaluation condition remain separate.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P092 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P092 script >}}
#
# ### Process, splits, and interpretation
#
# Corpus version 1 contains four intents with four training,two validation,and three held-out paraphrase
# families per intent. None of the phrases overlaps across splits. Training samples 1,024 balanced
# intent-policy combinations; validation 256 and test 512 use independent streams. Duplicated phrases are
# deliberate and are not independent language populations; the held-out unit is the phrase family. Each
# model uses 500 updates without checkpoint selection. The frozen-vector reference uses mean pooling and a
# two-layer 32-unit head; it is a strong separate baseline, not claimed parameter-matched. Configure
# EVIDENCE_TEXT_MODEL with an immutable local Hugging Face snapshot; its config,tokenizer,and weight digest
# is recorded. All encoder loading is offline and uses genuine weights. The model is frozen; only relflow
# projections/context/decoders or the explicit reference head learn. Thresholds are provisional, synthetic
# routing semantics are not business recommendations, and this small English corpus does not establish broad
# language quality.
#
# ### Complete experiment code
# %%
"""P092: use a real frozen encoder on held-out paraphrase families and visible policy."""

import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import relflow as rf
import torch
from reporting import experiment, report

BUDGET = 500
CORPUS_VERSION = 1
INTENTS = ("refund", "cancel", "pay", "update")
POLICIES = ("ordinary", "inverted")
CORPUS = {
    "train": (
        ("refund my payment", "give my money back", "reverse the purchase charge", "send the amount back to me"),
        ("cancel my subscription", "end the membership", "stop recurring payments", "close the service plan"),
        ("pay my bill", "settle the invoice", "send payment for this balance", "complete this purchase"),
        (
            "update my address",
            "change my contact details",
            "edit the phone number",
            "replace the stored mailing information",
        ),
    ),
    "validate": (
        ("return what I paid", "reimburse this purchase"),
        ("terminate my enrollment", "do not renew this plan"),
        ("clear the amount I owe", "make the scheduled payment"),
        ("correct my profile details", "amend the delivery address"),
    ),
    "test": (
        (
            "credit me for this returned order",
            "the merchant owes my payment back",
            "I want compensation for a returned purchase",
        ),
        ("this subscription should finish", "I no longer want the service", "please discontinue the membership"),
        ("transfer funds to cover my dues", "take money to finish the order", "discharge my outstanding charge"),
        (
            "my contact information needs revision",
            "use my new residence details",
            "the number on this account is outdated",
        ),
    ),
}


def records(rows: int, seed: int, split: str):
    rng = np.random.default_rng(seed)
    combinations = np.resize(np.arange(8), rows)
    rng.shuffle(combinations)
    for combination in combinations:
        intent = int(combination // 2)
        policy = int(combination % 2)
        phrases = CORPUS[split][intent]
        text = phrases[int(rng.integers(len(phrases)))]
        yield {"request": text, "policy": POLICIES[policy], "review": bool((intent < 2) ^ bool(policy))}


def resource() -> tuple[Path, dict]:
    configured = os.environ.get("EVIDENCE_TEXT_MODEL")
    if not configured:
        raise RuntimeError(
            "P092 needs EVIDENCE_TEXT_MODEL pointing to a pinned local Hugging Face encoder snapshot; provision the real model and relflow[text] before running."
        )
    path = Path(configured).expanduser().resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError(f"P092 encoder snapshot must be a local model directory with config.json, got {path}")
    weights = sorted([*path.glob("*.safetensors"), *path.glob("pytorch_model*.bin")])
    if not weights:
        raise ValueError(f"P092 encoder snapshot {path} has no local model weights")
    digest = hashlib.sha256()
    for file in sorted([path / "config.json", *weights, *path.glob("*token*.json"), *path.glob("vocab.*")]):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return path, {"directory": str(path), "snapshot_sha256": digest.hexdigest(), "snapshot_name": path.name}


def build(arm: str, path: Path) -> rf.Model:
    fields = {"review": rf.Boolean(mask=True)}
    if arm != "policy_only":
        fields["request"] = (
            rf.Category(p_unavailable=0.0)
            if arm == "category"
            else rf.Text(model=str(path), encoder_pooling="mean", max_length=64, cache=True)
        )
    if arm != "text_only":
        fields["policy"] = rf.Enum(values=POLICIES)
    return rf.Model(d_model=16, n_layers=1, n_heads=4, dropout=0.0, fields=fields)


def encoder_state(path: Path) -> dict:
    """Audit the actual extension-owned frozen encoder, including its buffers."""
    from relflow.tensorfields.extensions.text import CachedModel

    encoder = CachedModel.get_model(str(path)).model
    digest = hashlib.sha256()

    def update(value):
        """Hash nested extra state with explicit types, without pickle metadata."""
        digest.update(type(value).__name__.encode())
        if isinstance(value, torch.Tensor):
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                update(key)
                update(value[key])
        elif isinstance(value, (tuple, list)):
            digest.update(str(len(value)).encode())
            for item in value:
                update(item)
        elif value is None or isinstance(value, (str, bool, int, float)):
            digest.update(json.dumps(value, sort_keys=True).encode())
        else:
            raise TypeError(f"P092 cannot fingerprint encoder extra-state value of type {type(value).__name__}")

    for name, value in encoder.state_dict().items():
        digest.update(name.encode())
        update(value)
    return {
        "state_sha256": digest.hexdigest(),
        "parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad),
    }


def probability(model: rf.Model, rows: list[dict], arm: str) -> np.ndarray:
    fields = ("policy",) if arm == "policy_only" else ("request",) if arm == "text_only" else ("request", "policy")
    output = pl.DataFrame(model.predict(pa.Table.from_pylist([{key: row[key] for key in fields} for row in rows])))[
        "predictions"
    ].to_list()
    return np.asarray([row["review"]["content"]["probability"] for row in output])


def measure(probability: np.ndarray, rows: list[dict]) -> dict:
    target = np.asarray([row["review"] for row in rows])
    guess = probability >= 0.5
    f1 = []
    for value in (False, True):
        tp = np.sum((guess == value) & (target == value))
        fp = np.sum((guess == value) & (target != value))
        fn = np.sum((guess != value) & (target == value))
        f1.append(float(2 * tp / max(2 * tp + fp + fn, 1)))
    clipped = probability.clip(1e-7, 1 - 1e-7)
    return {
        "accuracy": float(np.mean(guess == target)),
        "macro_f1": float(np.mean(f1)),
        "nll": float(
            np.mean(-target.astype(float) * np.log(clipped) - (1 - target.astype(float)) * np.log1p(-clipped))
        ),
    }


def vector_reference(path: Path, train: list[dict], test: list[dict], seed: int, budget: int) -> np.ndarray:
    """Use the same real frozen encoder with a small explicit two-layer head."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    encoder = AutoModel.from_pretrained(str(path), local_files_only=True).eval().requires_grad_(False)
    unique = sorted({row["request"] for row in train + test})
    token = tokenizer(unique, padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        hidden = encoder(**token).last_hidden_state
        mask = token["attention_mask"].unsqueeze(-1)
        vectors = torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1).clamp_min(1), dim=-1)
    lookup = {text: i for i, text in enumerate(unique)}

    def inputs(rows):
        indices = torch.tensor([lookup[row["request"]] for row in rows])
        policy = torch.tensor([[float(row["policy"] == "inverted")] for row in rows])
        return torch.cat([vectors[indices], policy], dim=1)

    torch.manual_seed(seed)
    head = torch.nn.Sequential(torch.nn.Linear(vectors.shape[1] + 1, 32), torch.nn.GELU(), torch.nn.Linear(32, 1))
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.003, weight_decay=0.01)
    x = inputs(train)
    target = torch.tensor([row["review"] for row in train], dtype=torch.float32)
    rng = np.random.default_rng(seed)
    for _ in range(budget):
        indices = rng.choice(len(train), min(64, len(train)), replace=False)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(x[indices]).flatten(), target[indices])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return head(inputs(test)).flatten().sigmoid().numpy()


def fit(model: rf.Model, train: list[dict], validation: list[dict], seed: int, budget: int, accelerator: str) -> int:
    """Fit only the supplied training rows with a fixed optimizer/update budget."""
    data = rf.ArrowDataModule(
        train=pa.Table.from_pylist(train),
        validate=pa.Table.from_pylist(validation),
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    model.fit(
        data,
        batch_size=64,
        learning_rate=0.002,
        weight_decay=0.01,
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_steps=budget,
        max_epochs=-1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
        num_sanity_val_steps=0,
        limit_val_batches=1,
    )
    model.eval()
    return int(model.trainer.global_step)


@experiment("P092", seed=9201)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    path, origin = resource()
    budget = BUDGET if steps is None else min(steps, BUDGET)
    train = list(records(1024, seed + 1, "train"))
    validation = list(records(256, seed + 2, "validate"))
    test = list(records(512, seed + 3, "test"))
    metrics = {
        "corpus_version": CORPUS_VERSION,
        "encoder": origin,
        "transformers_version": version("transformers"),
        "arms": {},
    }
    checks = {}
    texts = {split: {phrase for intent in CORPUS[split] for phrase in intent} for split in CORPUS}
    checks["Training validation and test phrase families are disjoint"] = not (
        texts["train"] & texts["validate"] or texts["train"] & texts["test"] or texts["validate"] & texts["test"]
    )
    for arm in ("joint", "category", "text_only", "policy_only", "shuffled"):
        lit.seed_everything(seed, workers=True)
        model = build(arm, path)
        before_encoder = encoder_state(path) if arm != "category" and arm != "policy_only" else None
        fitting = train
        if arm == "shuffled":
            order = np.random.default_rng(seed + 4).permutation(len(train))
            fitting = [dict(row, request=train[order[i]]["request"]) for i, row in enumerate(train)]
        used = fit(model, fitting, validation, seed, budget, accelerator)
        scored = measure(probability(model, test, arm), test)
        scored["updates"] = used
        scored["training_template_accuracy"] = measure(
            probability(model, list(records(256, seed + 5, "train")), arm), list(records(256, seed + 5, "train"))
        )["accuracy"]
        metrics["arms"][arm] = scored
        checks[f"{arm}: finite held-out scores"] = bool(np.isfinite(list(scored.values())).all())
        if before_encoder is not None:
            after_encoder = encoder_state(path)
            scored["frozen_encoder"] = after_encoder
            checks[f"{arm}: real encoder has zero trainable parameters"] = after_encoder["trainable_parameters"] == 0
            checks[f"{arm}: training and evaluation preserve frozen encoder state"] = before_encoder == after_encoder
    metrics["arms"]["frozen_vector_head"] = measure(vector_reference(path, train, test, seed, budget), test)
    checks["Joint Text model reaches 0.75 held-out macro F1"] = metrics["arms"]["joint"]["macro_f1"] >= 0.75
    for control in ("category", "text_only", "policy_only", "shuffled"):
        checks[f"Joint Text exceeds {control} by 0.10 macro F1"] = (
            metrics["arms"]["joint"]["macro_f1"] >= metrics["arms"][control]["macro_f1"] + 0.10
        )
        checks[f"{control} remains below 0.65 accuracy"] = metrics["arms"][control]["accuracy"] <= 0.65
    return metrics, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
