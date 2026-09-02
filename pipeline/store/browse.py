"""A local, offline HTML browser for the vector store.

Qdrant's local file mode has no dashboard of its own; that is a feature of
running Qdrant as a server, which this project deliberately does not do.
The corpus is Housing Bank internal-use material, so nothing here is
published anywhere hosted: this writes one self-contained HTML file to
disk, no CDN, no network calls once generated, meant to be opened directly
from the filesystem in a browser.

What this module does not do: it does not query by similarity. It is for
looking at what got stored, source, page, section, actor, the text itself,
not for testing retrieval quality. That is bakeoff.py and the spot checks
in LEARNING/embedding.md.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from qdrant_client import QdrantClient

from ..chunking.chunk import CHUNK_TYPE_COLORS
from ..config import CONTEXT_VARIANTS, DATA_DIR, QDRANT_PATH
from .qdrant import PAYLOAD_FIELDS, collection_name

BROWSER_OUTPUT = DATA_DIR / "qdrant_browser.html"

_COLUMNS = ("chunk_id", "source", "page", "chunk_type", "section_path", "actor", "char_count")


def _fetch_collection(client: QdrantClient, variant: str) -> list[dict]:
    """Every point in one collection, payload only, no vectors: this is for
    reading, and a 1024-float vector printed on screen helps nobody.
    """
    points: list[dict] = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection_name(variant),
            limit=256, offset=offset, with_payload=True, with_vectors=False,
        )
        points.extend(p.payload for p in batch)
        if offset is None:
            break
    return points


def _row_json(payload: dict) -> dict:
    row = {field: payload.get(field) for field in PAYLOAD_FIELDS}
    row["text"] = payload.get("text", "")
    return row


def render_html(data: dict[str, list[dict]]) -> str:
    payload_json = json.dumps(data, ensure_ascii=False)
    variant_options = "".join(
        f'<option value="{html.escape(v)}">{html.escape(v)}</option>' for v in data
    )
    columns_json = json.dumps(_COLUMNS)
    type_colors_json = json.dumps(CHUNK_TYPE_COLORS)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GBG vector store browser</title>
<style>
  /* Housing Bank's own two colors, read off hbtf.com's live stylesheet
     rather than guessed: #005295 is the dominant navy across its headings,
     nav and active states (400+ occurrences), #c8b18b the gold accent on
     its buttons and borders. No logo or wordmark is reproduced here, only
     the two colors, on a page that never leaves this machine. */
  :root {{
    color-scheme: light dark;
    --navy: #005295;
    --navy-deep: #003a6b;
    --gold: #c8b18b;
    --gold-deep: #a9823f;
    --paper: #f7f7f8;
    --surface: #fff;
    --ink: #262626;
    --muted: #6b6b6b;
    --border: #e2e6ea;
    --hover: #eaf1f8;
  }}
  body {{ font-family: system-ui, sans-serif; margin: 0; background: var(--paper); color: var(--ink); }}
  header {{
    padding: 12px 16px; color: #fff; display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
    background: linear-gradient(135deg, var(--navy) 0%, var(--navy-deep) 100%);
    border-bottom: 3px solid var(--gold);
  }}
  header h1 {{ font-size: 15px; margin: 0; font-weight: 600; letter-spacing: .02em; }}
  header h1 .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--gold); margin-right: 6px; }}
  header select, header input {{
    padding: 6px 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,.35);
    font-size: 13px; background: rgba(255,255,255,.12); color: #fff;
  }}
  header select option {{ color: var(--ink); }}
  header input {{ flex: 1; min-width: 200px; }}
  header input::placeholder {{ color: rgba(255,255,255,.7); }}
  header select:focus, header input:focus {{ outline: 2px solid var(--gold); outline-offset: 1px; }}
  #count {{ font-size: 12px; color: var(--gold); font-weight: 600; white-space: nowrap; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--surface); }}
  th, td {{ padding: 7px 10px; border-bottom: 1px solid var(--border); font-size: 12px; text-align: left; vertical-align: top; }}
  th {{
    position: sticky; top: 0; background: var(--navy); color: #fff; cursor: pointer;
    user-select: none; font-weight: 600; border-bottom: 2px solid var(--gold);
  }}
  th:hover {{ background: var(--navy-deep); }}
  th.sorted {{ color: var(--gold); }}
  tr:hover {{ background: var(--hover); cursor: pointer; }}
  tr:nth-child(even) {{ background: rgba(0,82,149,.03); }}
  td.rtl, .detail-text {{ direction: rtl; unicode-bidi: plaintext; text-align: right; font-size: 14px; }}
  .chip {{
    display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 11px;
    font-weight: 600; color: #fff; white-space: nowrap;
  }}
  #detail {{ display: none; position: fixed; inset: 0; background: rgba(0,40,74,.45); }}
  #detail .panel {{
    position: absolute; right: 0; top: 0; bottom: 0; width: min(560px, 92vw); background: var(--surface);
    padding: 18px; overflow-y: auto; box-shadow: -4px 0 20px rgba(0,40,74,.35);
    border-left: 6px solid var(--gold);
  }}
  #detail .panel dt {{ font-size: 11px; color: var(--navy); font-weight: 700; text-transform: uppercase; letter-spacing: .04em; margin-top: 12px; }}
  #detail .panel dd {{ margin: 2px 0 0 0; font-size: 13px; }}
  #detail .close {{ float: right; cursor: pointer; font-size: 20px; border: none; background: none; color: var(--navy); }}
  #detail .close:hover {{ color: var(--gold-deep); }}
  #detail .detail-text {{ white-space: pre-wrap; margin-top: 4px; padding: 10px; background: var(--paper); border-radius: 6px; border-right: 3px solid var(--gold); }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --paper: #0f1720; --surface: #16212c; --ink: #e8e8e8; --muted: #9aa5ad;
      --border: #223142; --hover: #1c2e40; --navy-deep: #06263f;
    }}
    header {{ background: linear-gradient(135deg, #003a6b 0%, #041e33 100%); }}
    #detail {{ background: rgba(0,10,20,.6); }}
  }}
</style>
</head>
<body>
<header>
  <h1><span class="dot"></span>GBG vector store</h1>
  <select id="variant">{variant_options}</select>
  <input id="filter" type="text" placeholder="filter (chunk id, source, section, actor, text)">
  <span id="count"></span>
</header>
<table id="tbl">
  <thead><tr id="head"></tr></thead>
  <tbody id="body"></tbody>
</table>
<div id="detail"><div class="panel">
  <button class="close" onclick="document.getElementById('detail').style.display='none'">&times;</button>
  <dl id="detailBody"></dl>
</div></div>
<script>
const DATA = {payload_json};
const COLUMNS = {columns_json};
// From chunk.CHUNK_TYPE_COLORS, so this file and embedding/visualize.py
// never carry two definitions of the same mapping that could drift apart.
const TYPE_COLORS = {type_colors_json};
const variantSel = document.getElementById('variant');
const filterBox = document.getElementById('filter');
const head = document.getElementById('head');
const body = document.getElementById('body');
const countEl = document.getElementById('count');

head.innerHTML = COLUMNS.map(c => `<th data-col="${{c}}">${{c}}</th>`).join('');

let sortCol = null, sortDir = 1;

function currentRows() {{
  return DATA[variantSel.value] || [];
}}

function escapeHtml(s) {{
  return s.replace(/[&<>"']/g, m => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[m]));
}}

function chipHtml(chunkType) {{
  const color = TYPE_COLORS[chunkType] || '#7d7d7d';
  return `<span class="chip" style="background:${{color}}">${{escapeHtml(chunkType)}}</span>`;
}}

function render() {{
  const q = filterBox.value.trim().toLowerCase();
  let rows = currentRows();
  if (q) {{
    rows = rows.filter(r => COLUMNS.concat(['text']).some(c => String(r[c] ?? '').toLowerCase().includes(q)));
  }}
  if (sortCol) {{
    rows = rows.slice().sort((a, b) => {{
      const av = a[sortCol] ?? '', bv = b[sortCol] ?? '';
      return av > bv ? sortDir : av < bv ? -sortDir : 0;
    }});
  }}
  countEl.textContent = rows.length + ' / ' + currentRows().length + ' chunks';
  head.querySelectorAll('th').forEach(th => th.classList.toggle('sorted', th.dataset.col === sortCol));
  body.innerHTML = rows.map((r, i) => {{
    const cells = COLUMNS.map(c => {{
      const v = r[c] ?? '';
      if (c === 'chunk_type') return `<td>${{chipHtml(String(v))}}</td>`;
      const rtl = /[\\u0600-\\u06FF]/.test(String(v));
      return `<td class="${{rtl ? 'rtl' : ''}}">${{escapeHtml(String(v))}}</td>`;
    }}).join('');
    return `<tr data-idx="${{i}}">${{cells}}</tr>`;
  }}).join('');
  body.querySelectorAll('tr').forEach((tr, i) => {{
    tr.onclick = () => showDetail(rows[i]);
  }});
}}

function showDetail(row) {{
  const dl = document.getElementById('detailBody');
  const allFields = COLUMNS.concat(['doc_version', 'issue_date', 'review_date', 'unit', 'table_id',
    'extraction_quality', 'context_prefix', 'llm_source']);
  let out = '';
  for (const f of allFields) {{
    if (row[f] === undefined || row[f] === null || row[f] === '') continue;
    const value = f === 'chunk_type' ? chipHtml(String(row[f])) : escapeHtml(String(row[f]));
    out += `<dt>${{f}}</dt><dd>${{value}}</dd>`;
  }}
  out += `<dt>text</dt><dd class="detail-text">${{escapeHtml(row.text || '')}}</dd>`;
  dl.innerHTML = out;
  document.getElementById('detail').style.display = 'block';
}}

document.getElementById('detail').addEventListener('click', (e) => {{
  if (e.target.id === 'detail') e.target.style.display = 'none';
}});
head.addEventListener('click', (e) => {{
  const col = e.target.dataset.col;
  if (!col) return;
  if (sortCol === col) sortDir *= -1; else {{ sortCol = col; sortDir = 1; }}
  render();
}});
variantSel.onchange = render;
filterBox.oninput = render;
render();
</script>
</body>
</html>
"""


def run(qdrant_path: Path = QDRANT_PATH, output_path: Path = BROWSER_OUTPUT) -> bool:
    if not qdrant_path.exists():
        print(f"{qdrant_path} not found. Run `python cli.py store` first.")
        return False

    client = QdrantClient(path=str(qdrant_path))
    try:
        data = {}
        for variant in CONTEXT_VARIANTS:
            name = collection_name(variant)
            if not client.collection_exists(name):
                print(f"  {name} does not exist yet, skipping")
                continue
            points = _fetch_collection(client, variant)
            data[variant] = [_row_json(p) for p in points]
            print(f"  {variant}: {len(points)} chunks")
    finally:
        client.close()

    if not data:
        print("No collections found. Run `python cli.py store` first.")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(data), encoding="utf-8")
    print(f"\nwritten to {output_path}")
    print(f"open it directly: file:///{output_path.resolve().as_posix()}")
    return True


if __name__ == "__main__":
    run()
