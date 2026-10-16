"""Review tracked recordings in the browser and keep the ones worth training on.

    mimicvla review outputs/tracking --replays outputs/datasets/my_set_replays

Shows each clip's ``annotated.mp4`` (from ``mimicvla track``) next to its
Panda replay (``<replays>/<clip>/replay.mp4``, or ``<clip>/replay/replay.mp4``
from ``mimicvla replay``). Accept or reject each one; the accepted clips are
written to ``<tracking>/accepted_clips.txt``, ready for
``mimicvla build-dataset --clips``. Decisions are kept in
``<tracking>/review.json``, so a review can be resumed.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

PAGE = """<!doctype html>
<html lang="en"><meta charset="utf-8"><title>MimicVLA review</title>
<style>
  body { margin: 0; font: 15px system-ui, sans-serif; background: #111827; color: #f3f4f6; }
  header { padding: 12px 20px; border-bottom: 1px solid #374151; display: flex; gap: 16px; align-items: center; }
  main { display: grid; grid-template-columns: 260px 1fr; height: calc(100vh - 50px); }
  aside { overflow: auto; border-right: 1px solid #374151; }
  aside button { display: block; width: 100%; text-align: left; padding: 8px 12px; background: none;
                 color: inherit; border: 0; border-bottom: 1px solid #1f2937; cursor: pointer; }
  aside button.active { background: #1e3a5f; }
  .accept { color: #4ade80; } .reject { color: #f87171; }
  article { padding: 16px 20px; overflow: auto; }
  .videos { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  video { width: 100%; max-height: 70vh; background: #000; }
  .actions button { font: inherit; padding: 8px 16px; margin-right: 8px; border-radius: 6px; border: 0; cursor: pointer; }
  #yes { background: #15803d; color: white; } #no { background: #b91c1c; color: white; }
</style>
<header><strong>MimicVLA review</strong><span id="progress"></span>
<small>A: accept · R: reject · J/K: next/previous</small></header>
<main><aside id="list"></aside><article>
  <h2 id="title"></h2>
  <p>Accept a clip if the plug-tip circle follows the real plug and the robot replay follows the same motion into the port.</p>
  <div class="videos"><div><h3>Tracked recording</h3><video id="source" controls muted></video></div>
  <div><h3>Panda replay</h3><video id="replay" controls muted></video><p id="noreplay" hidden>No replay found for this clip.</p></div></div>
  <p class="actions"><button id="yes">Accept (A)</button><button id="no">Reject (R)</button></p>
</article></main>
<script>
let clips = [], decisions = {}, current = 0;
const $ = id => document.getElementById(id);
function render() {
  $('list').replaceChildren(...clips.map((c, i) => {
    const b = document.createElement('button'); b.textContent = c.name;
    b.className = (i === current ? 'active ' : '') + (decisions[c.name] || '');
    b.onclick = () => show(i); return b; }));
  const n = Object.keys(decisions).length, a = Object.values(decisions).filter(d => d === 'accept').length;
  $('progress').textContent = `${n}/${clips.length} reviewed, ${a} accepted`;
}
function show(i) {
  current = Math.max(0, Math.min(clips.length - 1, i)); const c = clips[current];
  $('title').textContent = c.name + (decisions[c.name] ? ` (${decisions[c.name]})` : '');
  $('source').src = `/video/source/${encodeURIComponent(c.name)}`;
  $('replay').hidden = !c.replay; $('noreplay').hidden = c.replay;
  if (c.replay) $('replay').src = `/video/replay/${encodeURIComponent(c.name)}`;
  render();
}
async function decide(decision) {
  const name = clips[current].name;
  await fetch('/decide', {method: 'POST', body: JSON.stringify({clip: name, decision})});
  decisions[name] = decision; show(current + 1);
}
$('yes').onclick = () => decide('accept'); $('no').onclick = () => decide('reject');
document.onkeydown = e => ({a: () => decide('accept'), r: () => decide('reject'),
  j: () => show(current + 1), k: () => show(current - 1)})[e.key]?.();
fetch('/clips').then(r => r.json()).then(d => { clips = d.clips; decisions = d.decisions;
  const first = clips.findIndex(c => !decisions[c.name]); show(first < 0 ? 0 : first); });
</script></html>
"""


def find_clips(tracking: Path, replays: Path | None) -> list[dict]:
    clips = []
    for video in sorted(tracking.glob("*/annotated.mp4")):
        name = video.parent.name
        candidates = ([replays / name / "replay.mp4"] if replays else []) + [
            video.parent / "replay" / "replay.mp4"]
        replay = next((path for path in candidates if path.is_file()), None)
        clips.append({"name": name, "source": video, "replay": replay})
    if not clips:
        raise SystemExit(f"No */annotated.mp4 under {tracking}; run `mimicvla track` first")
    return clips


def save(tracking: Path, clips: list[dict], decisions: dict[str, str]) -> None:
    (tracking / "review.json").write_text(json.dumps(decisions, indent=2) + "\n")
    accepted = [clip["name"] for clip in clips if decisions.get(clip["name"]) == "accept"]
    (tracking / "accepted_clips.txt").write_text("".join(f"{name}\n" for name in accepted))


def handler(tracking: Path, clips: list[dict]):
    by_name = {clip["name"]: clip for clip in clips}
    review = tracking / "review.json"
    decisions = json.loads(review.read_text()) if review.exists() else {}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, data: bytes, content_type: str):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_video(self, path: Path):
            # Byte ranges let the browser seek.
            size = path.stat().st_size
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
            start = int(match.group(1)) if match else 0
            end = min(int(match.group(2)), size - 1) if match and match.group(2) else size - 1
            self.send_response(206 if match else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if match:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as stream:
                stream.seek(start)
                self.wfile.write(stream.read(end - start + 1))

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self.send(PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/clips":
                listing = [{"name": c["name"], "replay": c["replay"] is not None} for c in clips]
                self.send(json.dumps({"clips": listing, "decisions": decisions}).encode(),
                          "application/json")
            elif path.startswith("/video/"):
                _, _, kind, name = path.split("/", 3)
                clip = by_name.get(unquote(name))
                file = clip and clip.get(kind)
                if kind in ("source", "replay") and file:
                    self.send_video(file)
                else:
                    self.send_error(404)
            else:
                self.send_error(404)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if payload.get("clip") not in by_name or payload.get("decision") not in ("accept", "reject"):
                self.send_error(400)
                return
            with lock:
                decisions[payload["clip"]] = payload["decision"]
                save(tracking, clips, decisions)
            self.send(b"{}", "application/json")

    return Handler


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("tracking", type=Path, help="folder of `mimicvla track` outputs")
    parser.add_argument("--replays", type=Path,
                        help="replay folder written by build-dataset (<output>_replays)")
    parser.add_argument("--port", type=int, default=8765)


def run(args: argparse.Namespace) -> None:
    clips = find_clips(args.tracking, args.replays)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(args.tracking, clips))
    print(f"Reviewing {len(clips)} clips at http://127.0.0.1:{args.port}/ "
          f"(accepted list: {args.tracking / 'accepted_clips.txt'}); Ctrl+C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
