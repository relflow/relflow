"""Refresh Read the Docs latest after the package and its Evidence are published."""

from __future__ import annotations

import argparse
import json
import os
import time
from urllib.request import Request, urlopen

from evidence.assets import PUBLICATION_LIMIT, download

RTD = "https://app.readthedocs.org"
API = f"{RTD}/api/v3/projects/relflow"
SELECTION = "https://www.relflow.ai/latest/evidence/selected.json"


def api(path: str, token: str, *, method: str = "GET") -> dict:
    request = Request(f"{API}/{path}", method=method)
    request.add_unredirected_header("Authorization", f"Token {token}")
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def selection(query: str = "") -> tuple[str, str] | None:
    content = download(f"{SELECTION}{query}", missing=True, limit=PUBLICATION_LIMIT)
    if content is None:
        return None
    publication = json.loads(content)
    return publication["version"], publication["revision"]


def refresh(version: str, revision: str) -> None:
    expected = (version, revision)
    if selection() == expected:
        print(f"Latest already serves relflow {version} Evidence from {revision}.")
        return
    token = os.environ.get("READTHEDOCS_TOKEN")
    if not token:
        raise ValueError("Set the READTHEDOCS_TOKEN Actions secret to refresh latest documentation after release.")

    build = api("versions/latest/builds/", token, method="POST")["build"]["id"]
    if type(build) is not int or build < 1:
        raise ValueError("Read the Docs did not return a valid build ID")
    address = f"{RTD}/projects/relflow/builds/{build}/"
    print(f"Triggered Read the Docs latest build {address}", flush=True)
    for _ in range(40):
        status = api(f"builds/{build}/", token)
        if status["state"]["code"] == "finished":
            if status["success"] is not True:
                raise RuntimeError(f"Read the Docs latest build failed or was cancelled. See {address}")
            break
        time.sleep(15)
    else:
        raise TimeoutError(f"Read the Docs latest build did not finish within 10 minutes. See {address}")

    # The hosted asset may still be propagating through the CDN after a build completes.
    for attempt in range(6):
        actual = selection(f"?build={build}&attempt={attempt}")
        if actual == expected:
            print(f"Latest serves relflow {version} Evidence from {revision}.")
            return
        if attempt < 5:
            time.sleep(10)
    raise ValueError(f"Read the Docs latest serves {actual!r} Evidence. Expected {expected!r}. See {address}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("revision")
    arguments = parser.parse_args()
    refresh(arguments.version, arguments.revision)


if __name__ == "__main__":
    main()
