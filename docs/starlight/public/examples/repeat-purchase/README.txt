Repeat purchase: a complete relflow workflow

Requirements: uv and Python 3.12 or newer. From the directory containing
repeat_purchase.py, run:

uv run --no-project --python 3.12 \
  --with "relflow @ git+https://github.com/relflow/relflow.git@dc64b85d75a85b3cef8ceada1869ee512f2fbe65" \
  repeat_purchase.py

This pins the relflow 0.5.0 candidate used for this example. No text encoder
or other model weights are needed. To check only the execution path, append
--smoke. Use --output DIRECTORY to select the artifact directory.

The script creates 512 fictional training customers and 128 separate
validation customers, trains for six CPU epochs with one thread, saves and
reloads the model, then scores three new customer records without labels.
Each order history is ordered from oldest to newest; all inputs describe
information available at the decision cutoff. The target describes whether
the customer makes another purchase during the following 30 days.

The fictional label rule uses the age of the latest known purchase. It samples
a return with probability 0.85 when that purchase is at most 14 days old,
and 0.15 otherwise. The data demonstrates the API and file workflow. It is
not a benchmark or evidence of real customer behavior or application quality.

repeat-purchase-run/ contains the generated Parquet data, the trained model,
predictions, checkpoints, and recorded-run.json. The receipt records the seed,
settings, package versions, source and data hashes, and actual predictions.
The --smoke receipt is marked separately and is not a full training result.
