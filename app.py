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
  /* ---- aesthetic: instrument-panel / oscilloscope. dark, phosphor-green data on
        near-black, one warm amber accent. monospace ONLY for the model's raw tokens
        (they ARE code/data); everything else is a refined grotesque. ---- */
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600&display=swap');

  :root{
    --bg:#0a0c0d; --panel:#101415; --line:#1c2426;
    --ink:#e8efe9; --dim:#5e6f66; --dimmer:#3a4742;
    --phosphor:#46f2a0; --phosphor-soft:#1d6b48; --amber:#ffb454;
    --grid-cell:#0e1413;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{
    background:
      radial-gradient(ellipse 120% 80% at 50% -10%, #0f1614 0%, var(--bg) 60%),
      var(--bg);
    color:var(--ink); font-family:'Sora',sans-serif; font-weight:300;
    -webkit-font-smoothing:antialiased; min-height:100vh;
  }
  .wrap{max-width:1180px;margin:0 auto;padding:42px 28px 80px}
  header{display:flex;align-items:baseline;gap:16px;flex-wrap:wrap;
    border-bottom:1px solid var(--line);padding-bottom:18px;margin-bottom:26px}
  h1{font-size:26px;font-weight:600;letter-spacing:-.02em;margin:0;
    display:flex;align-items:center;gap:11px}
  h1 .dot{width:9px;height:9px;border-radius:50%;background:var(--phosphor);
    box-shadow:0 0 10px var(--phosphor);animation:pulse 2.4s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
  .sub{color:var(--dim);font-size:13px;letter-spacing:.01em}
  .meta{margin-left:auto;color:var(--dimmer);font-family:'Space Mono',monospace;
    font-size:11px;text-transform:uppercase;letter-spacing:.12em}

  .console{display:flex;gap:12px;margin-bottom:14px}
  input[type=text]{flex:1;background:var(--panel);border:1px solid var(--line);
    color:var(--ink);font-family:'Space Mono',monospace;font-size:15px;
    padding:14px 16px;border-radius:9px;outline:none;transition:border-color .2s,box-shadow .2s}
  input[type=text]:focus{border-color:var(--phosphor-soft);
    box-shadow:0 0 0 3px rgba(70,242,160,.07)}
  button{background:var(--phosphor);color:#04140c;border:0;font-family:'Sora';
    font-weight:600;font-size:14px;padding:0 26px;border-radius:9px;cursor:pointer;
    transition:transform .08s,box-shadow .2s;box-shadow:0 0 0 0 rgba(70,242,160,.5)}
  button:hover{box-shadow:0 0 22px -2px rgba(70,242,160,.55)}
  button:active{transform:translateY(1px)}
  button:disabled{opacity:.4;cursor:wait;box-shadow:none}

  .predline{font-family:'Space Mono',monospace;font-size:13px;color:var(--dim);
    margin:0 2px 26px;min-height:18px}
  .predline b{color:var(--amber);font-weight:700}

  /* grid of all heads */
  .layers{display:flex;flex-direction:column;gap:3px}
  .layer-row{display:flex;align-items:center;gap:10px}
  .layer-label{width:54px;flex:none;font-family:'Space Mono',monospace;font-size:10px;
    color:var(--dimmer);text-align:right;letter-spacing:.08em}
  .heads{display:grid;grid-template-columns:repeat(12,1fr);gap:3px;flex:1}
  .head{aspect-ratio:1;background:var(--grid-cell);border-radius:3px;cursor:pointer;
    position:relative;overflow:hidden;border:1px solid transparent;
    transition:border-color .15s,transform .1s}
  .head:hover{border-color:var(--phosphor-soft);transform:scale(1.08);z-index:2}
  .head canvas{width:100%;height:100%;display:block;image-rendering:pixelated}
  .head.sel{border-color:var(--amber);box-shadow:0 0 14px -2px var(--amber)}
  .colhdr{display:grid;grid-template-columns:repeat(12,1fr);gap:3px;flex:1;
    margin-bottom:3px}
  .colhdr span{font-family:'Space Mono',monospace;font-size:9px;color:var(--dimmer);
    text-align:center;letter-spacing:.05em}
  .colhdr-row{display:flex;align-items:center;gap:10px;margin-bottom:5px}

  /* detail panel */
  .detail{margin-top:34px;background:var(--panel);border:1px solid var(--line);
    border-radius:13px;padding:24px;min-height:120px}
  .detail .empty{color:var(--dimmer);font-family:'Space Mono',monospace;font-size:13px;
    text-align:center;padding:34px 0}
  .detail h2{margin:0 0 4px;font-size:16px;font-weight:600}
  .detail h2 span{color:var(--amber);font-family:'Space Mono',monospace}
  .detail .hint{color:var(--dim);font-size:12.5px;margin:0 0 20px}
  .bigmap{display:grid;gap:2px;overflow-x:auto;padding-bottom:6px}
  .bigmap .cell{aspect-ratio:1;border-radius:2px;min-width:13px}
  .bigmap .axhead{font-family:'Space Mono',monospace;font-size:10px;color:var(--dim);
    white-space:nowrap;display:flex;align-items:center;justify-content:flex-end;
    padding-right:6px;min-width:0}
  .bigmap .axtop{font-family:'Space Mono',monospace;font-size:10px;color:var(--dim);
    writing-mode:vertical-rl;transform:rotate(180deg);justify-self:center;
    max-height:74px;overflow:hidden}
  .legend{display:flex;align-items:center;gap:8px;margin-top:16px;color:var(--dimmer);
    font-family:'Space Mono',monospace;font-size:10px;letter-spacing:.06em}
  .legend .bar{height:8px;width:120px;border-radius:4px;
    background:linear-gradient(90deg,var(--grid-cell),var(--phosphor))}
  .err{color:#ff6b6b;font-family:'Space Mono',monospace;font-size:13px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1><span class="dot"></span>attention-lens</h1>
    <span class="sub">gpt2-small · 12 layers × 12 heads · live attention patterns</span>
    <span class="meta" id="device">·</span>
  </header>

  <div class="console">
    <input id="txt" type="text" spellcheck="false"
      value="When John and Mary went to the store, John gave a drink to"
      placeholder="type a sentence...">
    <button id="run">run</button>
  </div>
  <div class="predline" id="pred"></div>

  <div class="colhdr-row" id="colhdrRow" style="display:none">
    <span class="layer-label"></span>
    <div class="colhdr" id="colhdr"></div>
  </div>
  <div class="layers" id="layers"></div>

  <div class="detail" id="detail">
    <div class="empty">run a sentence, then click any head above to inspect it</div>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
let DATA = null;          // last response
let SEL = null;           // [layer, head]

// phosphor colormap: weight 0..1 -> rgb (black -> green)
function colorize(w){
  const g = Math.round(40 + 200*w);
  const r = Math.round(10 + 30*w);
  const b = Math.round(15 + 70*w);
  return `rgb(${r},${g},${b})`;
}

function drawThumb(canvas, mat){
  const n = mat.length;
  canvas.width = n; canvas.height = n;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(n, n);
  for(let q=0;q<n;q++) for(let k=0;k<n;k++){
    const w = mat[q][k];
    const i = (q*n+k)*4;
    img.data[i]   = 10 + 30*w;
    img.data[i+1] = 40 + 200*w;
    img.data[i+2] = 15 + 70*w;
    img.data[i+3] = 255;
  }
  ctx.putImageData(img,0,0);
}

function renderGrid(){
  const L = DATA.n_layers, H = DATA.n_heads;
  // column header (head indices)
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
}

function renderDetail(){
  const d=$('#detail'); if(!SEL||!DATA){return;}
  const [l,h]=SEL;
  const mat=DATA.patterns[l][h];
  const toks=DATA.tokens.map(t=>t.replace(/ /g,'·'));   // show spaces as ·
  const n=toks.length;

  let html=`<h2>layer <span>${l}</span> · head <span>${h}</span></h2>`;
  html+=`<p class="hint">each row = a query token (the token doing the looking). `+
        `brighter cell = more attention paid to that key token (the token being looked at). `+
        `the lower-left triangle is all there is — tokens can only attend backwards.</p>`;

  // grid: top-left corner blank, top row = key tokens (vertical), then rows
  const cols = `minmax(74px,auto) repeat(${n}, minmax(13px,1fr))`;
  html+=`<div class="bigmap" style="grid-template-columns:${cols}">`;
  html+=`<div></div>`;                                   // corner
  for(let k=0;k<n;k++) html+=`<div class="axtop">${esc(toks[k])}</div>`;
  for(let q=0;q<n;q++){
    html+=`<div class="axhead">${esc(toks[q])}</div>`;
    for(let k=0;k<n;k++){
      const w=mat[q][k];
      html+=`<div class="cell" style="background:${colorize(w)}" title="${toks[q]} → ${toks[k]}: ${w.toFixed(3)}"></div>`;
    }
  }
  html+=`</div>`;
  html+=`<div class="legend"><span>0.0</span><span class="bar"></span><span>1.0 attention weight</span></div>`;
  d.innerHTML=html;
}

function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

async function run(){
  const text=$('#txt').value.trim(); if(!text)return;
  const btn=$('#run'); btn.disabled=true; btn.textContent='...';
  $('#pred').textContent='';
  try{
    const r=await fetch('/api/attn',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text})});
    const j=await r.json();
    if(j.error){$('#detail').innerHTML=`<div class="err">error: ${j.error}</div>`;return;}
    DATA=j; SEL=null;
    $('#device').textContent='model: gpt2-small';
    $('#pred').innerHTML=`next-token prediction → <b>${esc(j.prediction.replace(/ /g,'·'))}</b>`;
    renderGrid();
    $('#detail').innerHTML=`<div class="empty">click any head to inspect its attention pattern</div>`;
    // auto-select an interesting one: last layer, head 0
    selectHead(DATA.n_layers-1, 0);
  }catch(e){
    $('#detail').innerHTML=`<div class="err">request failed: ${e}</div>`;
  }finally{btn.disabled=false; btn.textContent='run';}
}

$('#run').onclick=run;
$('#txt').addEventListener('keydown',e=>{if(e.key==='Enter')run();});
run();   // run the default sentence on load
</script>
</body>
</html>"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
