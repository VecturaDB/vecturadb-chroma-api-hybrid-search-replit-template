"""A small question-answering page over the handbook of a fictional bike-share service, with VecturaDB inside the app.

Every question is answered three ways: by meaning (dense embeddings), by words (BM25), and by both (hybrid search).
The page also answers JSON at /api/search?q=<question>.
"""
import html
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, quote, urlparse

from search import open_handbook, search

EXAMPLES = ["the bike will not come out of the dock", "E17", "what happens if I keep the bike too long"]
COLUMNS = {
    "dense": ("By meaning", "dense embeddings of a small local model"),
    "bm25": ("By words", "the BM25 index, a sparse vector of the words of each passage"),
    "hybrid": ("Hybrid", "both rankings fused with RRF"),
}

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hybrid search with VecturaDB (Chroma API)</title>
<style>
  body {{ font: 16px/1.5 system-ui, sans-serif; margin: 0 auto; max-width: 72rem; padding: 1rem; color: #1d2433; background: #fff; }}
  form {{ display: flex; gap: .5rem; }} input {{ flex: 1; padding: .5rem; font: inherit; }} button {{ font: inherit; padding: .5rem 1rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(18rem, 1fr)); gap: 1rem; margin-top: 1rem; }}
  .col {{ border: 1px solid #d7dbe3; border-radius: .5rem; padding: .75rem; }} .col h2 {{ margin: 0; font-size: 1.1rem; }}
  .hint, .title, small {{ color: #5b6475; }} .hit {{ margin-top: .75rem; }} .title {{ font-size: .85rem; }}
</style></head><body>
<h1>Ask the Spokeford Bikes handbook</h1>
<p class="hint">Twenty passages of a fictional handbook in <code>docs/</code>, searched with VecturaDB inside this app.
In each column a lower score ranks first. Try:
{examples}</p>
<form><input name="q" value="{question}" placeholder="Ask a question" autofocus><button>Search</button></form>
<div class="grid">{columns}</div>
</body></html>"""


def render(question, results):
    examples = ", ".join(f'<a href="/?q={quote(e)}">{html.escape(e)}</a>' for e in EXAMPLES)
    columns = ""
    for kind, (name, how) in COLUMNS.items():
        hits = "".join(
            f'<div class="hit"><div class="title">{html.escape(row["metadata"]["title"])} '
            f'<small>score {row["score"]:.4f}</small></div>{html.escape(row["document"])}</div>'
            for row in results.get(kind, [])
        )
        columns += f'<div class="col"><h2>{name}</h2><div class="hint">{how}</div>{hits}</div>'
    return PAGE.format(examples=examples, question=html.escape(question), columns=columns)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        question = parse_qs(url.query).get("q", [""])[0].strip()
        results = search(handbook, question) if question else {}
        if url.path == "/api/search":
            self.reply(json.dumps(results).encode(), "application/json")
        elif url.path == "/":
            self.reply(render(question, results).encode(), "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def reply(self, body, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    started = time.monotonic()
    handbook = open_handbook()
    print(f"{handbook.count()} passages ready in {time.monotonic() - started:.1f} s", flush=True)
    # HOST is 127.0.0.1 by default, so the page is reachable from this machine only; .replit sets 0.0.0.0 for Replit.
    host, port = os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", "5000"))
    print(f"Serving on http://{host}:{port}", flush=True)
    HTTPServer((host, port), Handler).serve_forever()
