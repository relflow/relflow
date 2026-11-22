"""Restore the archived 0.5.2 docs and refresh the published 0.5.3 docs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

RTD = "https://app.readthedocs.org"
API = f"{RTD}/api/v3/projects/relflow"
SITE = "https://www.relflow.ai"
BRANCH = "ops/restore-release-docs-v0.5.2"
PREVIEW = BRANCH.replace("/", "-")
SOURCE_TAG = "v0.5.2-source-tag"
V052 = "v0.5.2"
V053 = "v0.5.3"
SOURCE_052 = "0b700f2df4edda5a3353b9a83dd8cf73fbcc1394"
SOURCE_053 = "0f70f86e99d123ea0de029cd412effea427fa8af"
DIGEST_052 = "8ad6c9157e1a42dacdd0d4a62fc281cb183248b97fa73a4984569497f830e76b"
DIGEST_053 = "8453333bdb540838a9cab1e0dda5490d4e2f9031344eec82c9ca54361c687304"
PUBLICATION_LIMIT = 32 * 1024 * 1024


def api(path, *, method="GET", payload=None, missing=False):
    """Send credentials only to the project API."""
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(f"{API}/{path}", data=body, headers={"Content-Type": "application/json"}, method=method)
    request.add_unredirected_header("Authorization", "Token " + os.environ["READTHEDOCS_TOKEN"])
    try:
        with urlopen(request, timeout=30) as response:
            data = response.read()
    except HTTPError as error:
        if missing and error.code == 404:
            return None
        raise
    return json.loads(data) if data else None


def wait(check, description, *, seconds=720):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(10)
    raise TimeoutError(description)


def version(slug):
    return api(f"versions/{slug}/", missing=True)


def update(current_slug, **changes):
    api(f"versions/{current_slug}/", method="PATCH", payload=changes)


def newest(slug):
    builds = api(f"versions/{slug}/builds/?limit=1")["results"]
    return builds[0]["id"] if builds else 0


def trigger(slug):
    result = api(f"versions/{slug}/builds/", method="POST")
    return result["build"]["id"]


def selection(slug, digest, build):
    """Wait for the CDN to serve the exact frozen publication."""
    address = f"{SITE}/{slug}/evidence/selected.json"

    def matches():
        url = f"{address}?build={build}&check={time.time_ns()}"
        try:
            request = Request(url, headers={"User-Agent": "relflow-evidence-verification"})
            with urlopen(request, timeout=30) as response:
                content = response.read(PUBLICATION_LIMIT + 1)
        except HTTPError as error:
            if error.code in {404, 502, 503, 504}:
                return False
            raise
        return len(content) <= PUBLICATION_LIMIT and hashlib.sha256(content).hexdigest() == digest

    wait(matches, f"{address} did not serve the expected Evidence digest {digest}", seconds=120)
    print(f"Verified {address} SHA-256 {digest}", flush=True)


def verify(slug, build, source, digest):
    address = f"{RTD}/projects/relflow/builds/{build}/"
    print(f"Waiting for {slug} build {address}", flush=True)

    def finished():
        status = api(f"builds/{build}/")
        if status["state"]["code"] != "finished":
            return False
        if status["success"] is not True or status["commit"] != source:
            raise RuntimeError(f"{slug} build did not succeed from {source}. See {address}")
        return True

    wait(finished, f"Timed out waiting for {address}")
    selection(slug, digest, build)


def refresh():
    if version(V053)["identifier"] != SOURCE_053:
        raise ValueError(f"{V053} does not identify the verified release {SOURCE_053}")
    main = subprocess.check_output(
        ["gh", "api", "--method", "GET", "repos/relflow/relflow/git/ref/heads/main", "--jq", ".object.sha"],
        text=True,
    ).strip()
    slugs = [V053]
    if main == SOURCE_053:
        slugs.append("latest")
    else:
        print(f"Main has advanced to {main}. Refreshing only the version tag.", flush=True)
    build_for = newest if os.environ.get("VERIFY_REFRESH_ONLY") == "true" else trigger
    builds = {slug: build_for(slug) for slug in slugs}
    for slug, build in builds.items():
        verify(slug, build, SOURCE_053, DIGEST_053)


def restore():
    """Verify the hidden recovery build before assigning the archived release URL."""
    source = os.environ["GITHUB_SHA"]
    current = version(V052)
    if current and current["type"] == "branch" and current["identifier"] == BRANCH:
        verify(V052, trigger(V052), source, DIGEST_052)
        return

    if not version(PREVIEW):
        api("sync-versions/", method="POST")
    branch = wait(lambda: version(PREVIEW), f"Read the Docs did not discover {BRANCH}", seconds=120)
    if branch["type"] != "branch" or branch["identifier"] != BRANCH:
        raise ValueError(f"Read the Docs {PREVIEW} does not identify {BRANCH}")
    if branch["active"]:
        update(PREVIEW, hidden=True)
        preview_build = trigger(PREVIEW)
    else:
        previous = newest(PREVIEW)
        update(PREVIEW, active=True, hidden=True)
        preview_build = wait(
            lambda: build if (build := newest(PREVIEW)) > previous else None,
            f"Read the Docs did not start the recovery build for {PREVIEW}",
            seconds=120,
        )
    verify(PREVIEW, preview_build, source, DIGEST_052)

    original = version(V052)
    archived = version(SOURCE_TAG)
    tag = original or archived
    if tag is None or tag["type"] != "tag" or tag["identifier"] != SOURCE_052 or tag["built"]:
        raise ValueError(f"{V052} must still be the unpublished version of {SOURCE_052} before swapping its URL")
    if original and archived:
        raise ValueError(f"The archive slug {SOURCE_TAG} is already in use")
    original_settings = (
        {"active": tag["active"], "hidden": tag["hidden"]}
        if original
        else {
            "active": True,
            "hidden": False,
        }
    )
    previous = newest(PREVIEW)

    try:
        if original:
            update(V052, slug=SOURCE_TAG, active=False, hidden=True)
        update(PREVIEW, slug=V052, active=True, hidden=False)
        public_build = wait(
            lambda: build if (build := newest(V052)) > previous else None,
            f"Read the Docs did not start a new build after moving the {V052} URL",
            seconds=120,
        )
        verify(V052, public_build, source, DIGEST_052)
    except Exception:
        public = version(V052)
        if public and public["type"] == "branch" and public["identifier"] == BRANCH:
            update(V052, slug=PREVIEW, active=True, hidden=True)
        if not version(V052) and version(SOURCE_TAG):
            update(SOURCE_TAG, slug=V052, **original_settings)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("refresh", "restore"))
    arguments = parser.parse_args()
    if arguments.operation == "refresh":
        refresh()
    else:
        restore()


if __name__ == "__main__":
    main()
