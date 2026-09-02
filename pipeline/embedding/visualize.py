"""A 2D look at the embedding space, for reading rather than for a gate.

Position: reads whatever the winning model already embedded, the same
cached vectors embedder.py wrote during the bake-off and the store build.
This computes nothing a downstream stage depends on; it exists the way
store/browse.py exists, so a person can look at what got built.

PCA, not t-SNE or UMAP. advanced-rag-plan.md's cuts named UMAP explicitly
to fit a five-day budget, but the real reason to prefer PCA here is
honesty about what a 2D picture of a 1024-dimension space can claim. PCA's
two axes are the two directions of greatest linear variance in the real
space, so a distance on this page is a true, if lossy, shadow of the same
cosine geometry retrieval actually searches, and the page states exactly
how much variance those two axes capture rather than leaving that
implicit. A manifold method optimises purely for a convincing 2D layout
and can draw a tight, confident-looking cluster between points that are
not particularly close in the space retrieval searches, which is a worse
failure mode for a page meant to be looked at and trusted than variance
loss honestly disclosed.

sklearn is not a dependency of this project, so PCA is written out as a
plain SVD, following sparse.py's own rule: fifteen real lines of formula
beats a library pulled in for one function.

What this module does not do: it decides nothing. No downstream stage
reads its output, and it never calls a model; every vector it plots was
already computed and cached by embedder.py for a different reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..chunking.chunk import CHUNK_TYPE_COLORS, Chunk, load_chunks
from ..config import CONTEXT_OUTPUTS, PROCESSED_DIR
from ..embedding import embedder
from ..embedding.bakeoff import load_winning_model

PROJECTION_OUTPUT = PROCESSED_DIR / "04_embedding_projection.html"

_HOVER_FIELDS = (
    "chunk_id", "source", "page", "chunk_type", "section_path", "actor",
)


def pca_2d(vectors: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Project rows of ``vectors`` onto their top two principal components.

    Centers on the mean, takes the SVD, and projects onto the first two
    right-singular vectors, which is exactly what PCA means: the two
    directions along which the points spread out the most. Returns the 2D
    coordinates and the fraction of total variance each axis explains, the
    number that has to sit next to the plot rather than be assumed, since
    two axes out of BGE-M3's 1024 dimensions can honestly capture very
    little of the real geometry.
    """
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    # full_matrices=False: only the first min(n, d) singular vectors exist
    # or matter here, and n (357 chunks) is far smaller than d (1024).
    u, s, _vt = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, :2] * s[:2]
    total_variance = float((s ** 2).sum())
    explained = tuple(
        float((s[i] ** 2) / total_variance) if total_variance > 0 else 0.0
        for i in (0, 1)
    )
    return coords, explained


def _project_variant(chunks: list[Chunk], model_key: str) -> dict[str, Any]:
    vectors = embedder.embed_passages([c.text for c in chunks], model_key)
    coords, explained = pca_2d(vectors)
    points = []
    for chunk, (x, y) in zip(chunks, coords):
        point = {field: chunk.metadata.get(field) for field in _HOVER_FIELDS}
        point["x"] = round(float(x), 4)
        point["y"] = round(float(y), 4)
        point["char_count"] = chunk.metadata.get("char_count")
        points.append(point)
    return {"points": points, "explained": explained}


# --- rendering ----------------------------------------------------------------

def render_html(data: dict[str, dict[str, Any]], model_key: str) -> str:
    payload_json = json.dumps(data, ensure_ascii=False)
    type_colors_json = json.dumps(CHUNK_TYPE_COLORS)
    model_json = json.dumps(model_key)
    variant_options = "".join(
        f'<option value="{variant}">{variant}</option>' for variant in data
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GBG embedding projection</title>
<style>
  /* Same two Housing Bank colors as store/browse.py, read off hbtf.com's
     own stylesheet: #005295 navy, #c8b18b gold. */
  :root {{
    color-scheme: light dark;
    --navy: #005295; --navy-deep: #003a6b; --gold: #c8b18b; --gold-deep: #a9823f;
    --paper: #f7f7f8; --surface: #fff; --ink: #262626; --muted: #6b6b6b;
    --border: #e2e6ea;
  }}
  body {{ font-family: system-ui, sans-serif; margin: 0; background: var(--paper); color: var(--ink); }}
  header {{
    padding: 12px 16px; color: #fff; display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
    background: linear-gradient(135deg, var(--navy) 0%, var(--navy-deep) 100%);
    border-bottom: 3px solid var(--gold);
  }}
  header h1 {{ font-size: 15px; margin: 0; font-weight: 600; letter-spacing: .02em; }}
  header h1 .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--gold); margin-right: 6px; }}
  header select {{
    padding: 6px 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,.35);
    font-size: 13px; background: rgba(255,255,255,.12); color: #fff;
  }}
  header select option {{ color: var(--ink); }}
  header label {{ font-size: 12px; display: flex; align-items: center; gap: 5px; }}
  #variance {{ font-size: 12px; color: var(--gold); font-weight: 600; margin-left: auto; }}
  #stage {{ position: relative; width: 100%; height: calc(100vh - 52px); background: var(--surface); overflow: hidden; }}
  #plot {{ display: block; cursor: crosshair; }}
  #legend {{
    position: absolute; top: 12px; left: 12px; background: var(--surface);
    border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px;
    font-size: 11px; box-shadow: 0 2px 8px rgba(0,40,74,.12);
  }}
  #legend .row {{ display: flex; align-items: center; gap: 6px; padding: 2px 0; cursor: pointer; }}
  #legend .swatch {{ width: 10px; height: 10px; border-radius: 50%; flex: none; }}
  #legend .row.off {{ opacity: .35; }}
  #tooltip {{
    position: absolute; display: none; max-width: 340px; background: var(--navy-deep); color: #fff;
    padding: 8px 10px; border-radius: 8px; font-size: 12px; pointer-events: none;
    border-left: 4px solid var(--gold); box-shadow: 0 4px 14px rgba(0,40,74,.3);
  }}
  #tooltip .section {{ direction: rtl; unicode-bidi: plaintext; text-align: right; margin-top: 4px; color: var(--gold); }}
  #tooltip .actor {{ direction: rtl; unicode-bidi: plaintext; text-align: right; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --paper: #0f1720; --surface: #16212c; --ink: #e8e8e8; --border: #223142; --navy-deep: #06263f; }}
    header {{ background: linear-gradient(135deg, #003a6b 0%, #041e33 100%); }}
  }}
</style>
</head>
<body>
<header>
  <h1><span class="dot"></span>GBG embedding projection</h1>
  <select id="variant">{variant_options}</select>
  <label><input type="checkbox" id="bySource"> color by source document instead of chunk type</label>
  <span id="variance"></span>
</header>
<div id="stage">
  <canvas id="plot"></canvas>
  <div id="legend"></div>
  <div id="tooltip"></div>
</div>
<script>
const DATA = {payload_json};
const TYPE_COLORS = {type_colors_json};
const MODEL = {model_json};
const SOURCE_COLORS = {{}};
const SOURCE_PALETTE = ['#005295', '#c8b18b', '#3c7a5c', '#6b4c9a'];

const variantSel = document.getElementById('variant');
const bySourceBox = document.getElementById('bySource');
const varianceEl = document.getElementById('variance');
const canvas = document.getElementById('plot');
const ctx = canvas.getContext('2d');
const legend = document.getElementById('legend');
const tooltip = document.getElementById('tooltip');
const stage = document.getElementById('stage');

const hidden = new Set();
let hovered = null;
let transform = null;

function resize() {{
  const rect = stage.getBoundingClientRect();
  canvas.width = rect.width * devicePixelRatio;
  canvas.height = rect.height * devicePixelRatio;
  canvas.style.width = rect.width + 'px';
  canvas.style.height = rect.height + 'px';
  draw();
}}

function colorKeyFor(point) {{
  return bySourceBox.checked ? point.source : point.chunk_type;
}}

function colorFor(key) {{
  if (bySourceBox.checked) {{
    if (!(key in SOURCE_COLORS)) {{
      SOURCE_COLORS[key] = SOURCE_PALETTE[Object.keys(SOURCE_COLORS).length % SOURCE_PALETTE.length];
    }}
    return SOURCE_COLORS[key];
  }}
  return TYPE_COLORS[key] || '#7d7d7d';
}}

function currentPoints() {{
  return (DATA[variantSel.value] || {{points: []}}).points;
}}

function fitTransform(points) {{
  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const w = canvas.width, h = canvas.height;
  const pad = 60 * devicePixelRatio;
  const sx = (w - 2 * pad) / ((xMax - xMin) || 1);
  const sy = (h - 2 * pad) / ((yMax - yMin) || 1);
  const s = Math.min(sx, sy);
  return (p) => [
    pad + (p.x - xMin) * s,
    h - pad - (p.y - yMin) * s,
  ];
}}

function draw() {{
  const points = currentPoints();
  transform = fitTransform(points);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const r = 4.5 * devicePixelRatio;
  for (const p of points) {{
    const key = colorKeyFor(p);
    if (hidden.has(key)) continue;
    const c = transform(p);
    ctx.beginPath();
    ctx.arc(c[0], c[1], p === hovered ? r * 1.6 : r, 0, 2 * Math.PI);
    ctx.fillStyle = colorFor(key);
    ctx.globalAlpha = p === hovered ? 1 : 0.72;
    ctx.fill();
    if (p === hovered) {{
      ctx.globalAlpha = 1;
      ctx.lineWidth = 2 * devicePixelRatio;
      ctx.strokeStyle = '#c8b18b';
      ctx.stroke();
    }}
  }}
  ctx.globalAlpha = 1;
  drawLegend(points);
}}

function drawLegend(points) {{
  const keys = [...new Set(points.map(colorKeyFor))].sort();
  legend.innerHTML = keys.map(k => {{
    const offClass = hidden.has(k) ? 'off' : '';
    return '<div class="row ' + offClass + '" data-key="' + escapeHtml(k) + '">' +
      '<span class="swatch" style="background:' + colorFor(k) + '"></span>' +
      '<span>' + escapeHtml(k) + '</span></div>';
  }}).join('');
  legend.querySelectorAll('.row').forEach(row => {{
    row.onclick = () => {{
      const k = row.dataset.key;
      if (hidden.has(k)) hidden.delete(k); else hidden.add(k);
      draw();
    }};
  }});
}}

function escapeHtml(s) {{
  return String(s).replace(/[&<>"']/g, m => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[m]));
}}

function nearestPoint(mx, my) {{
  const points = currentPoints();
  let best = null, bestDist = 14 * devicePixelRatio;
  for (const p of points) {{
    const c = transform(p);
    const d = Math.hypot(c[0] - mx, c[1] - my);
    if (d < bestDist) {{ bestDist = d; best = p; }}
  }}
  return best;
}}

canvas.addEventListener('mousemove', (e) => {{
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) * devicePixelRatio;
  const my = (e.clientY - rect.top) * devicePixelRatio;
  const p = nearestPoint(mx, my);
  if (p !== hovered) {{ hovered = p; draw(); }}
  if (p) {{
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX - rect.left + 14) + 'px';
    tooltip.style.top = (e.clientY - rect.top + 10) + 'px';
    const actorLine = p.actor ? ('<div class="actor">' + escapeHtml(p.actor) + '</div>') : '';
    tooltip.innerHTML =
      '<div><strong>' + escapeHtml(p.chunk_id) + '</strong></div>' +
      '<div>' + escapeHtml(p.chunk_type) + ' &middot; page ' + escapeHtml(p.page) +
      ' &middot; ' + escapeHtml(p.char_count) + ' chars</div>' +
      '<div class="section">' + escapeHtml(p.section_path || '') + '</div>' + actorLine;
  }} else {{
    tooltip.style.display = 'none';
  }}
}});
canvas.addEventListener('mouseleave', () => {{ tooltip.style.display = 'none'; hovered = null; draw(); }});

function updateVariance() {{
  const entry = DATA[variantSel.value] || {{explained: [0, 0]}};
  const e1 = entry.explained[0], e2 = entry.explained[1];
  varianceEl.textContent =
    MODEL + ', PC1 ' + (e1 * 100).toFixed(1) + '% + PC2 ' + (e2 * 100).toFixed(1) + '% of variance';
}}

variantSel.onchange = () => {{ hidden.clear(); updateVariance(); draw(); }};
bySourceBox.onchange = () => {{ hidden.clear(); draw(); }};
window.onresize = resize;
updateVariance();
resize();
</script>
</body>
</html>
"""


def run(
    variant_paths: dict[str, Path] = CONTEXT_OUTPUTS,
    output_path: Path = PROJECTION_OUTPUT,
) -> bool:
    try:
        model_key = load_winning_model()
    except FileNotFoundError as error:
        print(error)
        return False

    missing = [p for p in variant_paths.values() if not p.exists()]
    if missing:
        print(f"Missing {[str(p) for p in missing]}. Run `python cli.py context` first.")
        return False

    print(f"projecting {model_key}'s dense vectors to 2D with PCA")
    data = {}
    for variant, path in variant_paths.items():
        chunks = load_chunks(path)
        result = _project_variant(chunks, model_key)
        e1, e2 = result["explained"]
        print(f"  {variant}: {len(chunks)} points, "
              f"PC1 {e1 * 100:.1f}% + PC2 {e2 * 100:.1f}% of variance")
        data[variant] = result

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(data, model_key), encoding="utf-8")
    embedder.release(model_key)
    print(f"\nwritten to {output_path}")
    print(f"open it directly: file:///{output_path.resolve().as_posix()}")
    return True


if __name__ == "__main__":
    run()
