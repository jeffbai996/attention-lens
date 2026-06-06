"""
attention-lens — a toy for watching GPT-2 small's attention heads light up live.

Type a sentence, watch every head's attention pattern render as a grid.
Click a head to blow it up: see which tokens each token attends to.

This is the *playground* version of mech interp — it wires up TransformerLens
(load model, run_with_cache, pull cache["pattern", layer]) so that when you move
to the real IOI work, the model's already running and the cache is already flowing.

Run:  python app.py   ->  open http://localhost:5000
"""

import json
from flask import Flask, request, jsonify, Response
import torch
from transformer_lens import HookedTransformer

app = Flask(__name__)

# Cap input length. Attention is O(seq^2) per head across all 144 heads, and the
# response serializes every weight to JSON — so an unbounded prompt is a memory /
# payload blow-up. GPT-2's context is 1024 tokens anyway; we cap well under that.
MAX_CHARS = 2000

# --- model load (once, at startup) ---
print("loading gpt2-small via TransformerLens ... (first run downloads ~500MB)")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
model.eval()
N_LAYERS = model.cfg.n_layers      # 12 for gpt2-small
N_HEADS = model.cfg.n_heads        # 12 for gpt2-small
print(f"loaded. device={DEVICE}  layers={N_LAYERS}  heads={N_HEADS}")


def get_attention(text: str):
    """Run the model, return tokens + every head's attention pattern.

    cache["pattern", layer] has shape [batch, head, query_pos, key_pos].
    We strip batch and hand back a [layer][head][query][key] nested list,
    plus the string tokens so the frontend can label the axes.
    """
    tokens = model.to_tokens(text)                      # [1, seq]
    str_tokens = model.to_str_tokens(text)              # list[str], includes BOS
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, return_type=None)

    # patterns[layer] -> tensor [head, query, key]
    patterns = []
    for layer in range(N_LAYERS):
        p = cache["pattern", layer][0]                  # drop batch -> [head, q, k]
        patterns.append(p.cpu().float().tolist())

    # also pull the model's actual next-token prediction for the fun factor
    with torch.no_grad():
        logits = model(tokens, return_type="logits")
    next_id = int(logits[0, -1].argmax())
    next_tok = model.to_string([next_id])

    return {
        "tokens": str_tokens,
        "patterns": patterns,                           # [layer][head][q][k]
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "prediction": next_tok,
    }


@app.route("/api/attn", methods=["POST"])
def api_attn():
    # silent=True so a missing/!json Content-Type returns our 400 instead of raising
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "").strip()
    if not text:
        return jsonify({"error": "empty"}), 400
    if len(text) > MAX_CHARS:
        return jsonify({"error": f"text too long (max {MAX_CHARS} chars)"}), 400
    try:
        return jsonify(get_attention(text))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>attention-lens · gpt2-small</title>
<style>
  /* ---- aesthetic: calm, paper-light "research instrument". A warm near-white
        canvas, slate ink, and a single blue data ramp for attention weight.
        Inter for everything (a refined grotesque, in the Anthropic-Sans family);
        a mono ONLY for the model's raw tokens, since those literally ARE data. ---- */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  :root{
    --bg:#f6f7f9; --panel:#ffffff; --panel-2:#fbfcfd;
    --line:#e3e7ec; --line-2:#eef1f4;
    --ink:#1c2530; --ink-soft:#48566a; --dim:#75839a; --dimmer:#9aa6b8;
    --blue:#2f6df0; --blue-deep:#1b3f9e; --blue-soft:#d6e2fb;
    --accent:#c2603f;            /* warm clay accent for the selected head */
    --accent-soft:#f4ddd2;
    --grid-empty:#eef2f7;
    --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{
    background:var(--bg);
    color:var(--ink); font-family:'Inter',system-ui,sans-serif; font-weight:400;
    -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility;
    min-height:100vh; line-height:1.5;
  }
  .wrap{max-width:1080px;margin:0 auto;padding:40px 26px 96px}
  a{color:var(--blue);text-decoration:none}
  a:hover{text-decoration:underline}

  /* ---- header ---- */
  header{display:flex;align-items:baseline;gap:13px;flex-wrap:wrap;margin-bottom:8px}
  h1{font-size:25px;font-weight:700;letter-spacing:-.025em;margin:0;
    display:flex;align-items:center;gap:10px;color:var(--ink)}
  h1 .dot{width:9px;height:9px;border-radius:50%;background:var(--blue);
    box-shadow:0 0 0 4px var(--blue-soft)}
  .sub{color:var(--dim);font-size:13.5px}
  .meta{margin-left:auto;color:var(--dimmer);font-family:var(--mono);
    font-size:11px;text-transform:uppercase;letter-spacing:.1em}

  /* ---- intro / explainer ---- */
  .lede{font-size:15.5px;color:var(--ink-soft);max-width:74ch;margin:14px 0 4px}
  .lede b{color:var(--ink);font-weight:600}
  details.explain{margin:18px 0 28px;border:1px solid var(--line);border-radius:12px;
    background:var(--panel);overflow:hidden}
  details.explain>summary{cursor:pointer;padding:14px 18px;font-weight:600;font-size:14px;
    color:var(--ink);list-style:none;display:flex;align-items:center;gap:9px;user-select:none}
  details.explain>summary::-webkit-details-marker{display:none}
  details.explain>summary .chev{color:var(--blue);transition:transform .2s;font-size:12px}
  details.explain[open]>summary .chev{transform:rotate(90deg)}
  details.explain .body{padding:2px 18px 18px;font-size:14px;color:var(--ink-soft);
    border-top:1px solid var(--line-2)}
  details.explain .body p{margin:13px 0}
  details.explain .body strong{color:var(--ink);font-weight:600}
  .kbd{font-family:var(--mono);font-size:12px;background:var(--panel-2);
    border:1px solid var(--line);border-radius:5px;padding:1px 6px;color:var(--ink-soft)}

  /* ---- console ---- */
  .console{display:flex;gap:10px;margin-bottom:9px}
  input[type=text]{flex:1;background:var(--panel);border:1px solid var(--line);
    color:var(--ink);font-family:var(--mono);font-size:14.5px;
    padding:13px 15px;border-radius:10px;outline:none;transition:border-color .15s,box-shadow .15s}
  input[type=text]:focus{border-color:var(--blue);box-shadow:0 0 0 3px var(--blue-soft)}
  button{background:var(--blue);color:#fff;border:0;font-family:'Inter';
    font-weight:600;font-size:14px;padding:0 26px;border-radius:10px;cursor:pointer;
    transition:background .15s,transform .08s}
  button:hover{background:var(--blue-deep)}
  button:active{transform:translateY(1px)}
  button:disabled{opacity:.5;cursor:wait}
  .examples{font-size:12.5px;color:var(--dim);margin:0 2px 22px}
  .examples b{color:var(--ink-soft);font-weight:500}
  .ex{font-family:var(--mono);font-size:12px;color:var(--blue);cursor:pointer;
    border-bottom:1px dashed var(--blue-soft)}
  .ex:hover{border-bottom-style:solid}

  .predline{font-family:var(--mono);font-size:13.5px;color:var(--ink-soft);
    margin:0 2px 6px;min-height:20px}
  .predline b{color:var(--accent);font-weight:600;background:var(--accent-soft);
    padding:1px 7px;border-radius:5px}

  /* ---- "how to read" callout above the grid ---- */
  .readme{display:flex;gap:18px;flex-wrap:wrap;background:var(--panel);
    border:1px solid var(--line);border-radius:12px;padding:14px 18px;margin:0 0 18px;
    font-size:13px;color:var(--ink-soft)}
  .readme .col{flex:1;min-width:190px}
  .readme .col h4{margin:0 0 4px;font-size:12px;text-transform:uppercase;
    letter-spacing:.07em;color:var(--dim);font-weight:600}
  .readme .swatch{display:inline-block;width:46px;height:9px;border-radius:3px;
    vertical-align:middle;margin:0 6px;
    background:linear-gradient(90deg,var(--grid-empty),var(--blue))}

  /* ---- grid of all heads ---- */
  .gridwrap{background:var(--panel);border:1px solid var(--line);border-radius:14px;
    padding:18px 20px 20px}
  .gridtitle{font-size:13px;color:var(--dim);margin:0 0 14px;font-weight:500}
  .gridtitle b{color:var(--ink)}
  .layers{display:flex;flex-direction:column;gap:3px}
  .layer-row{display:flex;align-items:center;gap:10px}
  .layer-label{width:30px;flex:none;font-family:var(--mono);font-size:10px;
    color:var(--dim);text-align:right;letter-spacing:.04em}
  .heads{display:grid;grid-template-columns:repeat(12,1fr);gap:3px;flex:1}
  .head{aspect-ratio:1;background:var(--grid-empty);border-radius:3px;cursor:pointer;
    position:relative;overflow:hidden;border:1.5px solid transparent;
    transition:border-color .12s,transform .1s,box-shadow .12s}
  .head:hover{border-color:var(--blue);transform:scale(1.1);z-index:2;
    box-shadow:0 3px 10px -2px rgba(47,109,240,.35)}
  .head canvas{width:100%;height:100%;display:block;image-rendering:pixelated}
  .head.sel{border-color:var(--accent);box-shadow:0 0 0 2px var(--accent-soft)}
  .colhdr{display:grid;grid-template-columns:repeat(12,1fr);gap:3px;flex:1;margin-bottom:4px}
  .colhdr span{font-family:var(--mono);font-size:9px;color:var(--dimmer);
    text-align:center;letter-spacing:.03em}
  .colhdr-row{display:flex;align-items:center;gap:10px;margin-bottom:5px}
  .axhint{font-size:11.5px;color:var(--dim);margin:12px 2px 0;display:flex;
    justify-content:space-between;font-family:var(--mono)}

  /* ---- detail panel ---- */
  .detail{margin-top:22px;background:var(--panel);border:1px solid var(--line);
    border-radius:14px;padding:24px;min-height:120px}
  .detail .empty{color:var(--dim);font-size:13.5px;text-align:center;padding:30px 0}
  .detail h2{margin:0 0 4px;font-size:17px;font-weight:600;color:var(--ink)}
  .detail h2 span{color:var(--accent);font-family:var(--mono)}
  .detail .hint{color:var(--ink-soft);font-size:13.5px;margin:0 0 18px;max-width:78ch}
  .detail .hint b{color:var(--ink);font-weight:600}
  .bigmap{display:grid;gap:2px;overflow-x:auto;padding-bottom:6px}
  .bigmap .cell{aspect-ratio:1;border-radius:2px;min-width:14px;border:1px solid rgba(0,0,0,.02)}
  .bigmap .axhead{font-family:var(--mono);font-size:10px;color:var(--ink-soft);
    white-space:nowrap;display:flex;align-items:center;justify-content:flex-end;
    padding-right:7px;min-width:0}
  .bigmap .axtop{font-family:var(--mono);font-size:10px;color:var(--ink-soft);
    writing-mode:vertical-rl;transform:rotate(180deg);justify-self:center;
    max-height:78px;overflow:hidden}
  .legend{display:flex;align-items:center;gap:8px;margin-top:16px;color:var(--dim);
    font-family:var(--mono);font-size:10px;letter-spacing:.04em}
  .legend .bar{height:9px;width:130px;border-radius:5px;
    background:linear-gradient(90deg,var(--grid-empty),var(--blue))}
  .err{color:#c0392b;font-family:var(--mono);font-size:13px;background:#fdecea;
    border:1px solid #f5c6cb;border-radius:8px;padding:12px 14px}

  footer{margin-top:46px;padding-top:18px;border-top:1px solid var(--line);
    font-size:12.5px;color:var(--dim)}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1><span class="dot"></span>attention-lens</h1>
    <span class="sub">gpt2-small · 12 layers × 12 heads · live attention</span>
    <span class="meta" id="device">·</span>
  </header>

  <p class="lede">A transformer "reads" by letting each token <b>look back</b> at
  earlier tokens and pull in information from them. That looking-back is
  <b>attention</b>. This tool runs your sentence through GPT-2 small and shows you
  <b>all 144 attention heads at once</b> — every head is a tiny map of who-looked-at-whom.
  Click any head to blow it up.</p>

  <details class="explain">
    <summary><span class="chev">▶</span>New to this? A 30-second primer on attention heads</summary>
    <div class="body">
      <p><strong>What a "head" is.</strong> GPT-2 small has 12 layers, and each layer
      has 12 independent <strong>attention heads</strong> — 144 in total. Each head
      learns its own rule for which earlier tokens a given token should pay attention
      to. Some are boring plumbing; a few do real reasoning.</p>
      <p><strong>How to read a single head's map.</strong> It's a grid. Each
      <strong>row is a query</strong> token (the one doing the looking) and each
      <strong>column is a key</strong> token (a candidate to look at). A bright cell at
      (row&nbsp;<span class="kbd">i</span>, col&nbsp;<span class="kbd">j</span>) means
      "token&nbsp;i paid a lot of attention to token&nbsp;j." Rows sum to 1 — each token
      spreads a fixed budget of attention across the tokens before it.</p>
      <p><strong>Why it's a triangle.</strong> A token can only look <em>backwards</em>
      (the model predicts left-to-right and mustn't peek at the future), so the entire
      upper-right triangle is always blank. All the signal lives on or below the diagonal.</p>
      <p><strong>Heads you can spot by eye:</strong></p>
      <p>• <strong>Attention-sink heads</strong> — a bright <em>first column</em>: nearly
      everything dumps attention onto the first token. Common, low-level, basically idle.<br>
      • <strong>Previous-token heads</strong> — a bright line <em>just below the diagonal</em>:
      each token attends to the one right before it. The model's short-term memory.<br>
      • <strong>Induction heads</strong> (often layers 5–6) — the reasoning primitive. On
      repeated text ("the cat sat. the cat ___") they look back to <em>what followed the
      last time</em> this token appeared. Try a sentence with a repeated word.<br>
      • <strong>Name-mover heads</strong> (later layers) — on the John/Mary sentence, the
      final token attends strongly back to " Mary". You're literally seeing (part of) the
      circuit that produces the answer.</p>
    </div>
  </details>

  <div class="console">
    <input id="txt" type="text" spellcheck="false"
      value="When John and Mary went to the store, John gave a drink to"
      placeholder="type a sentence...">
    <button id="run">run</button>
  </div>
  <div class="examples"><b>try:</b>
    <span class="ex" data-ex="When John and Mary went to the store, John gave a drink to">the IOI sentence</span> ·
    <span class="ex" data-ex="The cat sat on the mat. The cat sat on the">a repeat (hunt induction heads)</span> ·
    <span class="ex" data-ex="Paris is the capital of France. London is the capital of">a fact</span>
  </div>
  <div class="predline" id="pred"></div>

  <div class="readme" id="readme" style="display:none">
    <div class="col">
      <h4>Reading the grid below</h4>
      Each small square is one head. Inside it, brightness = attention weight
      <span class="swatch"></span> (pale = none, blue = strong). Rows run top→bottom by
      layer (<b>L0</b> input-side → <b>L11</b> output-side); columns are heads <b>H0–H11</b>.
    </div>
    <div class="col">
      <h4>What to do</h4>
      <b>Hover</b> to enlarge a head, <b>click</b> to open its full query×key heatmap below.
      Look for the patterns from the primer: a hot first column (sink), a bright
      sub-diagonal (previous-token), or late-layer heads pointing at the answer token.
    </div>
  </div>

  <div class="gridwrap" id="gridwrap" style="display:none">
    <p class="gridtitle">All <b>144 heads</b> — <span id="gridtok"></span></p>
    <div class="colhdr-row" id="colhdrRow" style="display:none">
      <span class="layer-label"></span>
      <div class="colhdr" id="colhdr"></div>
    </div>
    <div class="layers" id="layers"></div>
    <div class="axhint"><span>← layer 0 (top) reads the input</span><span>layer 11 (bottom) writes the prediction →</span></div>
  </div>

  <div class="detail" id="detail">
    <div class="empty">Run a sentence, then click any head above to inspect its full attention map.</div>
  </div>

  <footer>
    GPT-2 small via <a href="https://github.com/TransformerLensOrg/TransformerLens" target="_blank" rel="noopener">TransformerLens</a>.
    Attention weights are read straight from the model's cache — nothing is faked or post-processed.
    A playground for mechanistic interpretability.
  </footer>
</div>

<script>
const $ = s => document.querySelector(s);
let DATA = null;          // last response
let SEL = null;           // [layer, head]

// blue colormap: weight 0..1 -> rgb (pale grey-blue -> deep blue)
function lerp(a,b,t){return Math.round(a+(b-a)*t);}
function colorize(w){
  // pale (#eef2f7) -> blue (#2f6df0), eased so faint weights stay readable
  const t = Math.pow(w, 0.75);
  return `rgb(${lerp(238,47,t)},${lerp(242,109,t)},${lerp(247,240,t)})`;
}

function drawThumb(canvas, mat){
  const n = mat.length;
  canvas.width = n; canvas.height = n;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(n, n);
  for(let q=0;q<n;q++) for(let k=0;k<n;k++){
    const t = Math.pow(mat[q][k], 0.75);
    const i = (q*n+k)*4;
    img.data[i]   = lerp(238,47,t);
    img.data[i+1] = lerp(242,109,t);
    img.data[i+2] = lerp(247,240,t);
    img.data[i+3] = 255;
  }
  ctx.putImageData(img,0,0);
}

function renderGrid(){
  const L = DATA.n_layers, H = DATA.n_heads;
  const colhdr = $('#colhdr'); colhdr.innerHTML='';
  for(let h=0;h<H;h++){ const s=document.createElement('span'); s.textContent='H'+h; colhdr.appendChild(s); }
  $('#colhdrRow').style.display='flex';

  const layers = $('#layers'); layers.innerHTML='';
  for(let l=0;l<L;l++){
    const row = document.createElement('div'); row.className='layer-row';
    const lab = document.createElement('div'); lab.className='layer-label'; lab.textContent='L'+l;
    const heads = document.createElement('div'); heads.className='heads';
    for(let h=0;h<H;h++){
      const cell = document.createElement('div'); cell.className='head';
      cell.dataset.l=l; cell.dataset.h=h;
      cell.title = `layer ${l}, head ${h} — click to inspect`;
      const cv = document.createElement('canvas');
      drawThumb(cv, DATA.patterns[l][h]);
      cell.appendChild(cv);
      cell.onclick = ()=>selectHead(l,h);
      heads.appendChild(cell);
    }
    row.appendChild(lab); row.appendChild(heads);
    layers.appendChild(row);
  }
}

function selectHead(l,h){
  SEL=[l,h];
  document.querySelectorAll('.head.sel').forEach(e=>e.classList.remove('sel'));
  const cell=document.querySelector(`.head[data-l="${l}"][data-h="${h}"]`);
  if(cell) cell.classList.add('sel');
  renderDetail();
  $('#detail').scrollIntoView({behavior:'smooth',block:'nearest'});
}

function renderDetail(){
  const d=$('#detail'); if(!SEL||!DATA){return;}
  const [l,h]=SEL;
  const mat=DATA.patterns[l][h];
  const toks=DATA.tokens.map(t=>t.replace(/ /g,'·'));   // show spaces as ·
  const n=toks.length;

  let html=`<h2>layer <span>${l}</span> · head <span>${h}</span></h2>`;
  html+=`<p class="hint"><b>Each row is a query token</b> (the token doing the looking); `+
        `<b>each column is a key token</b> (the token being looked at). A brighter cell means the `+
        `row's token paid more attention to the column's token. The blank upper-right triangle is `+
        `the causal mask — tokens can only attend backwards, never to the future. Spaces are shown as ·.</p>`;

  const cols = `minmax(78px,auto) repeat(${n}, minmax(14px,1fr))`;
  html+=`<div class="bigmap" style="grid-template-columns:${cols}">`;
  html+=`<div></div>`;                                   // corner
  for(let k=0;k<n;k++) html+=`<div class="axtop">${esc(toks[k])}</div>`;
  for(let q=0;q<n;q++){
    html+=`<div class="axhead">${esc(toks[q])}</div>`;
    for(let k=0;k<n;k++){
      const w=mat[q][k];
      html+=`<div class="cell" style="background:${colorize(w)}" title="${esc(toks[q])} → ${esc(toks[k])}: ${w.toFixed(3)}"></div>`;
    }
  }
  html+=`</div>`;
  html+=`<div class="legend"><span>0.0</span><span class="bar"></span><span>1.0 — attention weight</span></div>`;
  d.innerHTML=html;
}

function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

async function run(){
  const text=$('#txt').value.trim(); if(!text)return;
  const btn=$('#run'); btn.disabled=true; btn.textContent='running…';
  $('#pred').textContent='';
  try{
    const r=await fetch('/api/attn',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text})});
    const j=await r.json();
    if(j.error){$('#detail').innerHTML=`<div class="err">error: ${esc(String(j.error))}</div>`;return;}
    DATA=j; SEL=null;
    $('#device').textContent='model: gpt2-small';
    $('#pred').innerHTML=`the model's next-token prediction → <b>${esc(j.prediction.replace(/ /g,'·'))}</b>`;
    $('#readme').style.display='flex';
    $('#gridwrap').style.display='block';
    $('#gridtok').textContent=`${DATA.tokens.length} tokens · click any to inspect`;
    renderGrid();
    $('#detail').innerHTML=`<div class="empty">Click any head above to open its full attention map. (Tip: the last few layers, on the John/Mary sentence, are where the answer-moving heads live.)</div>`;
    selectHead(DATA.n_layers-1, 0);   // auto-open one interesting head
  }catch(e){
    $('#detail').innerHTML=`<div class="err">request failed: ${esc(String(e))}</div>`;
  }finally{btn.disabled=false; btn.textContent='run';}
}

$('#run').onclick=run;
$('#txt').addEventListener('keydown',e=>{if(e.key==='Enter')run();});
document.querySelectorAll('.ex').forEach(el=>el.onclick=()=>{$('#txt').value=el.dataset.ex; run();});
run();   // run the default sentence on load
</script>
</body>
</html>"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
