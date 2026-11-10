# Group research artifacts

The Group research reports, experiment scripts, selected plots, and input
checksums remain in this directory. Full captures, raw measurements, logs,
and input snapshots from the 2026-11-09 investigations are preserved on the
lab instead of being checked into the repository.

The archive is `~/relflow-research/archives/relflow-pr87-research-b7458e4.tar.gz`
on `lab.tail815717.ts.net`. Its SHA-256 is
`ae9a16c52dfa22a9949caf4c30c6cb92bba25fb7aad73bb03a50ba3a57188e8b`.
It contains all 2,396 research files that were present at commit `b7458e4`,
including the complete original captures and their source and provenance.

To inspect or reproduce a run, retrieve and extract the archive into a separate
directory, then follow the relevant report's instructions:

```sh
ssh lab.tail815717.ts.net 'cat ~/relflow-research/archives/relflow-pr87-research-b7458e4.tar.gz' > /tmp/relflow-pr87-research.tar.gz
echo 'ae9a16c52dfa22a9949caf4c30c6cb92bba25fb7aad73bb03a50ba3a57188e8b  /tmp/relflow-pr87-research.tar.gz' | shasum -a 256 -c
mkdir -p /tmp/relflow-pr87-research
tar -xzf /tmp/relflow-pr87-research.tar.gz -C /tmp/relflow-pr87-research
```

The archive is research provenance, not a substitute for the hosted Evidence
CI report or a new measurement. Paths such as `output/` and `attempts/` in the
individual reports refer to the extracted archive.
