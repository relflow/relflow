"""Small S3-compatible server for testing real object-store reads locally."""

from __future__ import annotations

import hashlib
import threading
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit


class S3Fixture:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = dict(objects)
        self.modified = {key: datetime(2026, 10, 2, tzinfo=UTC) for key in objects}
        self.requests = []
        self.deny_reads = False
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def respond(self, include_body):
                parsed = urlsplit(self.path)
                key = unquote(parsed.path).removeprefix("/diagnostics/")
                query = parse_qs(parsed.query, keep_blank_values=True)
                fixture.requests.append((self.command, key, query))
                if "list-type" in query:
                    prefix = query.get("prefix", [""])[0]
                    delimiter = query.get("delimiter", [""])[0]
                    root = ET.Element("ListBucketResult", xmlns="http://s3.amazonaws.com/doc/2006-03-01/")
                    ET.SubElement(root, "Name").text = "diagnostics"
                    ET.SubElement(root, "Prefix").text = prefix
                    ET.SubElement(root, "IsTruncated").text = "false"
                    directories = set()
                    for name, body in sorted(fixture.objects.items()):
                        if not name.startswith(prefix):
                            continue
                        tail = name[len(prefix) :]
                        if delimiter and delimiter in tail:
                            directories.add(prefix + tail.split(delimiter, 1)[0] + delimiter)
                            continue
                        item = ET.SubElement(root, "Contents")
                        ET.SubElement(item, "Key").text = name
                        ET.SubElement(item, "LastModified").text = fixture.modified[name].isoformat()
                        ET.SubElement(item, "ETag").text = '"' + hashlib.md5(body).hexdigest() + '"'
                        ET.SubElement(item, "Size").text = str(len(body))
                        ET.SubElement(item, "StorageClass").text = "STANDARD"
                    for directory in sorted(directories):
                        ET.SubElement(ET.SubElement(root, "CommonPrefixes"), "Prefix").text = directory
                    ET.SubElement(root, "KeyCount").text = str(len(root.findall("Contents")) + len(directories))
                    ET.SubElement(root, "MaxKeys").text = query.get("max-keys", ["1000"])[0]
                    body = ET.tostring(root)
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if key not in fixture.objects or (include_body and fixture.deny_reads):
                    self.send_response(403 if fixture.deny_reads else 404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                payload = fixture.objects[key]
                start, stop = 0, len(payload) - 1
                requested_range = self.headers.get("Range") if include_body else None
                if requested_range:
                    first, last = requested_range.removeprefix("bytes=").split("-", 1)
                    start = int(first)
                    stop = min(int(last), stop) if last else stop
                body = payload[start : stop + 1]
                self.send_response(206 if requested_range else 200)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("ETag", '"' + hashlib.md5(payload).hexdigest() + '"')
                self.send_header("Last-Modified", format_datetime(fixture.modified[key], usegmt=True))
                if requested_range:
                    self.send_header("Content-Range", f"bytes {start}-{stop}/{len(payload)}")
                self.send_header("Connection", "close")
                self.end_headers()
                if include_body:
                    self.wfile.write(body)

            def do_HEAD(self):
                self.respond(False)

            def do_GET(self):
                self.respond(True)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def options(self):
        return {
            "endpoint_override": f"127.0.0.1:{self.server.server_port}",
            "scheme": "http",
            "region": "us-east-1",
            "access_key": "fixture",
            "secret_key": "fixture",
            "connect_timeout": 3.0,
            "request_timeout": 3.0,
        }

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        assert not self.thread.is_alive()
