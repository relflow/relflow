"""Execute one evidence task inside a network-isolated, disposable container."""

import json
import resource
import sys
from pathlib import Path

from evidence_ci import execute, write_json

if __name__ == "__main__":
    task = json.loads(Path(sys.argv[1]).read_text())
    record = execute(task, root=Path("/suite"), timeout=1700)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    record["cpu_seconds"] = round(usage.ru_utime + usage.ru_stime, 3)
    record["peak_rss_bytes"] = usage.ru_maxrss * 1024
    write_json(Path("/results/record.json"), record)
