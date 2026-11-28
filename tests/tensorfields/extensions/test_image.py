import os
import threading
from collections import Counter
from types import SimpleNamespace

import numpy as np
import polars as pl
import pyarrow as pa
import pytest
import torch
from PIL import Image
from pyarrow import fs

import relflow as rf
from relflow.tensorfields.extensions import image
from relflow.tensorfields.shared.frozen import Backbone


class Processor:
    def __init__(self):
        self.scale = 1.0

    def to_json_string(self):
        return str(self.scale)

    def __call__(self, *, images, return_tensors):
        assert images.mode == "RGB" and return_tensors == "np"
        rgb = np.asarray(images.resize((8, 8)), dtype=np.float32).transpose(2, 0, 1)
        return {"pixel_values": (rgb * self.scale / 255)[None]}


class Vision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=3)
        self.batches = []
        self.weight = torch.nn.Parameter(torch.ones(3), requires_grad=False)

    def forward(self, *, pixel_values):
        self.batches.append(len(pixel_values))
        content = pixel_values.mean(dim=(-2, -1)) * self.weight
        return SimpleNamespace(last_hidden_state=content.unsqueeze(1))


@pytest.fixture
def vision(monkeypatch, tmp_path):
    processor = Processor()
    model = Vision()
    monkeypatch.setattr(Backbone, "_models", {"tiny": Backbone("tiny", model, 3)})
    from transformers.models.auto.image_processing_auto import AutoImageProcessor

    monkeypatch.setattr(AutoImageProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    monkeypatch.setenv("RELFLOW_CACHE_DIR", str(tmp_path / "cache"))
    Image.new("RGB", (9, 7), "red").save(tmp_path / "red.png")
    Image.new("RGB", (7, 9), "blue").save(tmp_path / "blüe.png")
    return model


def build(base, **options):
    return rf.Model(d_model=8, n_heads=2, n_layers=1, photo=rf.Image(model="tiny", base=str(base), **options))


def vectors(model, paths):
    field = model.encode(pl.DataFrame({"photo": paths}, schema={"photo": pl.String}))["/photo"]
    return model.nodes["/photo"].embedder.encode(field.content, field.state)


@pytest.mark.parametrize("resident", [0, 1])
def test_concurrent_reads_compact_ids_and_warm_parity(vision, tmp_path, monkeypatch, resident):
    model = build(tmp_path, immutable=True, read_workers=2, encoder_batch_size=2, resident=resident)
    paths = ["blüe.png", None, "red.png", "blüe.png"]
    data = pl.DataFrame({"photo": paths})
    assert model.encode(data)["/photo"].content["ids"].numel() == len(paths)
    reads = Counter()
    barrier = threading.Barrier(2)
    original = image.open_image

    def concurrent(path):
        reads[path] += 1
        barrier.wait(timeout=10)
        return original(path)

    monkeypatch.setattr(image, "open_image", concurrent)
    first = vectors(model, paths)
    assert sorted(reads.values()) == [1, 1] and vision.batches == [2]
    torch.testing.assert_close(
        first[:, 0], torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    )
    monkeypatch.setattr(image, "open_image", lambda *args: pytest.fail("warm hit opened an image"))
    assert torch.equal(vectors(model, paths), first)
    assert model.nodes["/photo"].embedder.warm(pa.array(paths)) == 0
    assert model.nodes["/photo"].embedder.warm("red.png") == 0
    assert vision.batches == [2]
    monkeypatch.setattr(image, "open_image", original)
    direct = build(tmp_path, cache=False, encoder_batch_size=1)
    torch.testing.assert_close(vectors(direct, paths), first)
    assert vision.batches == [2, 1, 1]
    with pytest.raises(ValueError, match="cache=True"):
        direct.nodes["/photo"].embedder.warm(paths)


def test_mutable_reference_changes_and_failed_reads_can_retry(vision, tmp_path):
    model = build(tmp_path)
    path = tmp_path / "red.png"
    assert image.location(path.name, tmp_path.as_uri()) == str(path)
    root = tmp_path.parents[-1]
    assert image.location("photo.jpg", root.as_uri()) == str(root / "photo.jpg")
    first = vectors(model, [path.as_uri()])
    Image.new("RGB", (9, 7), "green").save(tmp_path / "new.png")
    os.replace(tmp_path / "new.png", path)
    assert not torch.equal(first, vectors(model, [str(path)]))
    file = tmp_path / "broken.png"
    file.write_bytes(b"broken")
    with pytest.raises(OSError) as error:
        vectors(model, [str(file)])
    assert any("/photo" in note for note in error.value.__notes__)
    assert model.nodes["/photo"].embedder.store.stats()["claims"] == 0
    Image.new("L", (8, 8), 80).save(file)
    assert vectors(model, [str(file)]).count_nonzero() == 3
    with pytest.raises(FileNotFoundError, match="missing.png"):
        vectors(model, ["missing.png"])
    with pytest.raises(ValueError, match="nonempty image path"):
        vectors(model, [""])


def test_snapshot_cannot_publish_replaced_file_under_old_id(vision, tmp_path):
    model = build(tmp_path)
    field = model.encode(pl.DataFrame({"photo": ["red.png"]}))["/photo"]
    os.replace(tmp_path / "blüe.png", tmp_path / "red.png")
    encoder = model.nodes["/photo"].embedder
    with pytest.raises(ValueError, match="changed after tensorization"):
        encoder.encode(field.content, field.state)
    assert encoder.store.stats()["results"] == 0


@pytest.mark.parametrize("immutable", [False, True])
def test_remote_snapshot_reads_body_once_and_skips_decode_on_hits(vision, tmp_path, monkeypatch, immutable):
    storage = fs.SubTreeFileSystem(str(tmp_path), fs.LocalFileSystem())
    monkeypatch.setattr(image, "remote", lambda authority: (storage, ""))
    opens = []
    decodes = []
    open_image, pixels = image.open_image, image.pixels

    def opened(path):
        opens.append(path)
        return open_image(path)

    def decoded(*args, **kwargs):
        decodes.append(1)
        return pixels(*args, **kwargs)

    monkeypatch.setattr(image, "open_image", opened)
    monkeypatch.setattr(image, "pixels", decoded)
    model = build("s3://fixtures", immutable=immutable)
    first = vectors(model, ["red.png", "red.png"])
    assert len(opens) == len(decodes) == 1
    second = vectors(model, ["red.png"])
    assert torch.equal(first[0], second[0])
    assert len(opens) == (1 if immutable else 2)
    assert len(decodes) == 1 and vision.batches == [1]
    if not immutable:
        os.replace(tmp_path / "blüe.png", tmp_path / "red.png")
        assert not torch.equal(first[0], vectors(model, ["red.png"])[0])
        assert len(opens) == 3 and len(decodes) == 2


def test_nested_fields_share_pristine_vectors_and_preserve_holes(vision, tmp_path):
    def photo(**kw):
        return rf.Image(model="tiny", base=str(tmp_path), **kw)

    model = rf.Model(
        d_model=8, n_layers=1, n_heads=2, album=rf.Branch(photo=photo(embed=True)), answer=photo(predict=True)
    )
    data = pl.DataFrame(
        {
            "album": [[{"photo": "red.png"}, None, {"photo": "blüe.png"}], [], None],
            "answer": ["red.png", "blüe.png", None],
        }
    )
    batch = model.encode(data, strata="train")
    hidden = batch["/answer"]
    assert not hidden.content["ids"].any()
    left = model.nodes["/album/photo"].embedder
    assert left.warm(["red.png", "blüe.png"]) == 2
    target = model.nodes["/answer"].embedder
    encoded = target.encode(hidden.targets["content"], hidden.targets["state"])
    assert encoded[:, 0].count_nonzero() == 2
    reports = model.predict(data)
    assert reports.columns == ["representations"]
    assert [row["parents"] for row in reports["representations"][0]] == [[0], [2]]
    assert left.linear is not target.linear and vision.batches == [2]
    copy = rf.Schema.model_validate_json(model.schema.model_dump_json())
    assert copy.requests["/album/photo"].model == "tiny"


def test_transform_changes_invalidate_cache(vision, tmp_path):
    model = build(tmp_path, immutable=True)
    initial = vectors(model, ["red.png"])
    encoder = model.nodes["/photo"].embedder
    encoder.processor.scale = 0.5
    torch.testing.assert_close(vectors(model, ["red.png"]), 0.5 * initial)
    with torch.no_grad():
        vision.weight.mul_(0.5)
    torch.testing.assert_close(vectors(model, ["red.png"]), 0.25 * initial)
    assert vision.batches == [1, 1, 1]


def test_oversize_descriptor_batch_splits_and_preserves_order(vision, tmp_path, monkeypatch):
    model = build(tmp_path, resident=0)
    store = model.nodes["/photo"].embedder.store
    fetch = store.client.fetch

    def limited(ids, variant):
        if len(ids) > 1:
            raise ValueError("artifact lookup exceeds 32 MiB; reduce batch size or split large artifacts")
        return fetch(ids, variant)

    monkeypatch.setattr(store.client, "fetch", limited)
    result = vectors(model, ["blüe.png", "red.png", "blüe.png"])
    torch.testing.assert_close(result[:, 0], torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    assert vision.batches == [1, 1] and store.stats()["claims"] == 0


def test_chunked_dictionaries_and_empty_collections(vision, tmp_path):
    model = build(tmp_path, embed=True)
    chunks = [pa.array(["red.png", None]).dictionary_encode(), pa.array(["blüe.png", "red.png"]).dictionary_encode()]
    data = pa.table({"photo": pa.chunked_array(chunks)})
    field = model.encode(data)["/photo"]
    result = model.nodes["/photo"].embedder.encode(field.content, field.state)
    torch.testing.assert_close(
        result[:, 0], torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    )
    assert model.predict(data.slice(0, 0)).height == 0


@pytest.mark.parametrize("kind, size, orientation", [("RGB", (65, 38), 1), ("L", (23, 57), 6), ("RGBA", (7, 9), 2)])
def test_native_crop_matches_checkpoint_pixels(tmp_path, kind, size, orientation):
    from transformers import BitImageProcessorPil

    processor = BitImageProcessorPil(size={"shortest_edge": 24}, crop_size={"height": 19, "width": 20})
    crop = image.Crop.from_processor(processor)
    assert crop is not None
    path = tmp_path / ("photo.png" if kind == "RGBA" else "photo.jpg")
    raw = np.random.default_rng(787).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    raster = Image.fromarray(raw).convert(kind)
    exif = Image.Exif()
    exif[274] = orientation
    raster.save(path, exif=exif)
    row = image.snapshot(str(path), True)[0]
    reference = image.pixels(row, processor, {})
    np.testing.assert_array_equal(image.pixels(row, processor, {}, crop=crop), reference)
    assert reference.shape == (3, 19, 20)
    processor.crop_size = {"height": 64, "width": 64}
    assert image.Crop.from_processor(processor) is None


def test_decoder_reduction_has_its_own_cache_identity(vision, tmp_path):
    path = tmp_path / "texture.jpg"
    Image.fromarray(np.random.default_rng(54).integers(0, 256, (175, 193, 3), dtype=np.uint8)).save(path)
    full = build(tmp_path)
    reduced = build(tmp_path, jpeg_short_edge=16)
    direct = build(tmp_path, jpeg_short_edge=16, cache=False)
    before = vectors(full, [str(path)])
    after = vectors(reduced, [str(path)])
    assert not torch.equal(before, after)
    torch.testing.assert_close(vectors(direct, [str(path)]), after)
    assert torch.equal(vectors(reduced, [str(path)]), after)
    assert reduced.nodes["/photo"].embedder.store.stats()["results"] == 2
    assert vision.batches == [1, 1, 1]


@pytest.mark.integration
def test_offline_dino_training_prediction_and_checkpoint_with_loader_worker(tmp_path):
    from transformers import BitImageProcessorPil, Dinov2Config, Dinov2Model

    assets = tmp_path / "encoder"
    config = Dinov2Config(
        image_size=16, patch_size=4, hidden_size=12, num_hidden_layers=1, num_attention_heads=3, mlp_ratio=2
    )
    Dinov2Model(config).save_pretrained(assets)
    BitImageProcessorPil(size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}).save_pretrained(assets)
    paths = []
    for color in ("red", "blue"):
        path = tmp_path / f"{color}.png"
        Image.new("RGB", (17, 21), color).save(path)
        paths.append(str(path))

    def photo(**options):
        return rf.Image(model=str(assets), store=str(tmp_path / "store"), **options)

    model = rf.Model(d_model=8, n_layers=1, n_heads=2, photo=photo(embed=True), target=photo(predict=True))
    data = pl.DataFrame({"photo": paths, "target": list(reversed(paths))})
    embedder = model.nodes["/photo"].embedder
    before = embedder.linear[0].weight.detach().clone()
    dm = rf.PolarsDataModule(train=data, validate=data, num_workers=1, persistent_workers=False, pin_memory=False)
    model.fit(
        dm,
        batch_size=2,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        logger=False,
        rollback=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    assert not torch.equal(before, embedder.linear[0].weight)
    assert all(p.grad is None for p in Backbone.get_model(str(assets)).model.parameters())
    assert embedder.store.stats()["results"] == 2
    predicted = model.predict(data)
    assert predicted.columns == ["representations"]
    checkpoint = tmp_path / "image.ckpt"
    model.save(checkpoint)
    restored = rf.Model.load(checkpoint)
    assert restored.predict(data).equals(predicted)
