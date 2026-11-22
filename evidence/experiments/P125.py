# %% [markdown]
# ---
# title: Does a pitch classifier trained on WAV transfer to unseen Vorbis recordings?
# description: Train on real PCM files and score matched WAV, Vorbis and missing inputs on new tones.
# evidence-group: Representation
# evidence-readout:
# - {label: WAV accuracy, metric: [conditions, wav, accuracy], format: percent}
# - {label: Vorbis accuracy, metric: [conditions, vorbis, accuracy], format: percent}
# - {label: Missing clip accuracy, metric: [conditions, missing, accuracy], format: percent}
# evidence-review:
#   question: Can one Audio classifier preserve a pitch decision across a lossy format change?
#   modeling: Classify balanced low and high tones using a compact frozen encoder distilled from Whisper-base
#     and a learned relflow classifier. The downstream classifier trains and validates on WAV recordings.
#   comparison:
#     kind: Controlled comparison
#     metric: Accuracy
#     conditions:
#     - {label: Original WAV, path: [conditions, wav, accuracy], format: percent}
#     - {label: Vorbis transcode, path: [conditions, vorbis, accuracy], format: percent}
#     - {label: Missing clip, path: [conditions, missing, accuracy], format: percent}
#     definition: Fraction of held-out tones assigned their correct pitch band.
#     direction: Higher accuracy is better. A constant prediction scores 50 percent.
#   note: The test holds the waveform fixed across formats. The two pitch bands are well separated.
# evidence-plot:
#   mode: summary
#   focus: Does pitch accuracy survive a change in encoding?
#   comparison: Compare predictions for the same held-out tones in WAV, Vorbis and missing-clip form.
#   limit: Results cover two synthetic frequency bands and a distilled convolutional encoder.
# ---
#
# ## Example
#
# ```yaml
# clip: 9d9836042eac.wav
# pitch: high
# ```
#
# A half-second recording with an 1800 Hz fundamental belongs to the high band.
# The model receives its audio vector and predicts the hidden pitch label.
#
# ## Comparison
#
# Each independent split varies frequency, phase, loudness and background noise.
# Each held-out WAV has a Vorbis copy encoded from its exact PCM samples.
# Removing the file reference leaves the same rows and labels with no acoustic clue.
#
# ## Result
#
# Accuracy measures recovery of the hidden pitch band. Paired agreement measures how
# often the predicted band stays the same after transcoding, including shared errors.
# Missing-input accuracy gives the balanced-class reference level.
#
# ## Method
#
# Train on 64 tones for 80 optimizer steps, validate on 16 new tones and score
# 32 further tones in each condition. The offline checkpoint was distilled from Whisper-base on 512 independent,
# unlabeled harmonic mixtures. Its two convolutional layers learn to match the
# same teacher features for PCM and Vorbis versions of each unlabeled mixture.
# Each downstream split introduces new recording identities.
#
# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["lightning", "numpy", "polars", "pyyaml>=6", "relflow[audio]>=0.5.3", "soundfile", "torch", "transformers>=5.9"]
#
# [tool.relflow.evidence]
# capability = "representation"
# types = ["Audio"]
# conditions = ["unseen-inputs", "missing-inputs"]
# ///

"""P125 measures cross-codec generalization on independently synthesized pitch bands."""

import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import relflow as rf
import soundfile as sf
import torch
from policies import CONTROL
from reporting import experiment, report
from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperModel

BUDGET = 80
FIXTURE = Path(__file__).with_suffix(".npz")
SHA256 = "35ea838b38a91a5c76c67f1a4a2311e3e9c8ac3b61d0c99f4ac2f203c9f0d26a"


def checkpoint(directory):
    """Restore the frozen offline student and verify every serialized encoder parameter."""
    if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != SHA256:
        raise ValueError("P125 distilled encoder fixture differs from its pinned SHA-256")
    WhisperFeatureExtractor(chunk_length=1, feature_size=80).save_pretrained(directory)
    with np.load(FIXTURE, allow_pickle=False) as weights:
        metadata = json.loads(str(weights["metadata"]))
        model = WhisperModel(WhisperConfig(**metadata["config"]))
        model.encoder.load_state_dict(
            {key: torch.from_numpy(weights[key]) for key in weights.files if key != "metadata"}
        )
    model.save_pretrained(directory)
    return directory


def tones(directory, rng, count, transcode=False):
    rate = 16000
    time = np.arange(rate // 2) / rate
    for high in rng.permutation(np.arange(count) % 2):
        frequency = rng.uniform(1400, 2200) if high else rng.uniform(160, 320)
        wave = rng.uniform(0.2, 0.7) * np.sin(2 * np.pi * frequency * time + rng.uniform(0, 2 * np.pi))
        wave += rng.normal(0, 0.006, len(wave))
        path = directory / (rng.bytes(12).hex() + ".wav")
        sf.write(path, wave, rate, subtype="PCM_16")
        if transcode:
            pcm, _ = sf.read(path, dtype="float32")
            sf.write(path.with_suffix(".ogg"), pcm, rate, format="OGG", subtype="VORBIS")
        yield {"clip": path.name, "pitch": "high" if high else "low"}


@experiment("P125", seed=12501)
def run(seed, steps, accelerator):
    lit.seed_everything(seed, workers=True)
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        encoder = checkpoint(directory / "whisper")

        class Tone(rf.Model):
            clip = rf.Audio(base=temporary, model=str(encoder), store=str(directory / "cache"), immutable=True)
            pitch = rf.Enum("low", "high", mask=True)

        model = Tone(preset=CONTROL.name)
        train_rng, val_rng, test_rng = np.random.default_rng(seed).spawn(3)
        train = pl.DataFrame(tones(directory, train_rng, 64))
        validate = pl.DataFrame(tones(directory, val_rng, 16))
        test = pl.DataFrame(tones(directory, test_rng, 32, transcode=True))
        module = rf.PolarsDataModule(
            train=train, validate=validate, seed=seed, num_workers=0, persistent_workers=False, pin_memory=False
        )
        store = model.nodes["/clip"].embedder.store
        initial_vectors = store.stats()["results"]
        try:
            model.fit(
                module,
                batch_size=16,
                max_steps=min(steps or BUDGET, BUDGET),
                max_epochs=-1,
                learning_rate=0.001,
                early_stopping=False,
                rollback=False,
                accelerator=accelerator,
                devices=1,
                strategy="auto",
                precision="32-true",
                deterministic=True,
                logger=False,
                enable_checkpointing=False,
                enable_model_summary=False,
                enable_progress_bar=False,
                num_sanity_val_steps=0,
            )
            conditions = {
                "wav": test.select("clip"),
                "vorbis": test.select(pl.col("clip").str.replace(r"\.wav$", ".ogg")),
                "missing": test.with_columns(clip=pl.lit(None, dtype=pl.String)).select("clip"),
            }
            predicted = {}
            new_vectors = {}
            for name, inputs in conditions.items():
                before = store.stats()["results"]
                output = model.predict(inputs, batch_size=16)["predictions"]
                new_vectors[name] = store.stats()["results"] - before
                predicted[name] = output.struct.field("pitch").struct.field("content").struct.field("value").to_list()
        finally:
            store.owner.close()
    expected = test["pitch"].to_numpy()
    accuracy = {name: float(np.mean(np.array(values) == expected)) for name, values in predicted.items()}
    agreement = float(np.mean(np.array(predicted["wav"]) == predicted["vorbis"]))
    return (
        dict(
            conditions={name: {"accuracy": score} for name, score in accuracy.items()},
            agreement=agreement,
            cache=dict(initial=initial_vectors, new_vectors=new_vectors),
            observations=dict(truth=expected, predictions=predicted),
            steps=model.trainer.global_step,
        ),
        {
            "cold encodes for each WAV and Vorbis file": initial_vectors == 0
            and new_vectors == {"wav": len(test), "vorbis": len(test), "missing": 0},
            "full training budget": model.trainer.global_step == BUDGET,
            "WAV accuracy at least 90 percent": accuracy["wav"] >= 0.9,
            "Vorbis accuracy at least 90 percent": accuracy["vorbis"] >= 0.9,
            "paired agreement at least 90 percent": agreement >= 0.9,
            "missing clips at balanced chance": accuracy["missing"] == 0.5,
        },
    )


if __name__ == "__main__":
    report(run)
