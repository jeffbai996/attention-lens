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
  /* ---- dark "research instrument": warm near-black canvas, soft off-white ink,
        a blue ramp for attention weight, one warm-clay accent for the selection.
        Inter everywhere (Anthropic-Sans-family grotesque); mono ONLY for raw tokens. ---- */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  :root{
    --bg:#15171c; --panel:#1c1f26; --panel-2:#232730; --raise:#272c36;
    --line:#2b303b; --line-2:#343a47;
    --ink:#eef1f6; --ink-soft:#b6beca; --dim:#838c9c; --dimmer:#5b6373;
    --blue:#5b8cff; --blue-bright:#79a3ff; --blue-soft:#22304f;
    --accent:#e0875a; --accent-soft:#3a2a20;
    --grid-empty:#21252e;
    --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{
    background:var(--bg); color:var(--ink);
    font-family:'Inter',system-ui,sans-serif; font-weight:400;
    -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility;
    min-height:100vh; line-height:1.55;
  }
  .wrap{max-width:1000px;margin:0 auto;padding:44px 26px 110px}
  a{color:var(--blue-bright);text-decoration:none}
  a:hover{text-decoration:underline}

  header{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:18px}
  h1{font-size:23px;font-weight:600;letter-spacing:-.02em;margin:0;
    display:flex;align-items:center;gap:10px;color:var(--ink)}
  h1 .dot{width:8px;height:8px;border-radius:50%;background:var(--blue);
    box-shadow:0 0 14px 1px var(--blue)}
  .sub{color:var(--dim);font-size:13px}
  .meta{margin-left:auto;color:var(--dimmer);font-family:var(--mono);
    font-size:11px;text-transform:uppercase;letter-spacing:.1em}

  .lede{font-size:15px;color:var(--ink-soft);max-width:72ch;margin:0 0 22px}
  .lede b{color:var(--ink);font-weight:600}

  details.explain{margin:0 0 26px;border:1px solid var(--line);border-radius:12px;
    background:var(--panel);overflow:hidden}
  details.explain>summary{cursor:pointer;padding:13px 17px;font-weight:500;font-size:13.5px;
    color:var(--ink-soft);list-style:none;display:flex;align-items:center;gap:9px;user-select:none}
  details.explain>summary:hover{color:var(--ink)}
  details.explain>summary::-webkit-details-marker{display:none}
  details.explain>summary .chev{color:var(--blue);transition:transform .2s;font-size:11px}
  details.explain[open]>summary .chev{transform:rotate(90deg)}
  details.explain .body{padding:4px 17px 18px;font-size:13.5px;color:var(--ink-soft);
    border-top:1px solid var(--line-2)}
  details.explain .body p{margin:12px 0}
  details.explain .body strong{color:var(--ink);font-weight:600}

  .console{display:flex;gap:10px;margin-bottom:10px}
  input[type=text]{flex:1;background:var(--panel);border:1px solid var(--line);
    color:var(--ink);font-family:var(--mono);font-size:14px;
    padding:13px 15px;border-radius:10px;outline:none;transition:border-color .15s,box-shadow .15s}
  input[type=text]::placeholder{color:var(--dimmer)}
  input[type=text]:focus{border-color:var(--blue);box-shadow:0 0 0 3px var(--blue-soft)}
  button{background:var(--blue);color:#0b1220;border:0;font-family:'Inter';
    font-weight:600;font-size:14px;padding:0 26px;border-radius:10px;cursor:pointer;
    transition:background .15s,transform .08s}
  button:hover{background:var(--blue-bright)}
  button:active{transform:translateY(1px)}
  button:disabled{opacity:.5;cursor:wait}
  .examples{font-size:12.5px;color:var(--dim);margin:0 2px 26px}
  .examples b{color:var(--ink-soft);font-weight:500}
  .ex{font-family:var(--mono);font-size:12px;color:var(--blue-bright);cursor:pointer;
    border-bottom:1px dashed var(--blue-soft)}
  .ex:hover{border-bottom-color:var(--blue-bright)}

  .predline{font-family:var(--mono);font-size:13px;color:var(--ink-soft);
    margin:0 2px 24px;min-height:20px}
  .predline b{color:var(--accent);font-weight:600;background:var(--accent-soft);
    padding:2px 8px;border-radius:6px}

  /* ---- the narration card: plain-English "what this head is doing" ---- */
  .narrate{background:linear-gradient(180deg,var(--panel-2),var(--panel));
    border:1px solid var(--line-2);border-radius:13px;padding:18px 20px;margin:0 0 16px}
  .narrate .tag{display:inline-block;font-family:var(--mono);font-size:11px;
    letter-spacing:.04em;color:var(--blue-bright);background:var(--blue-soft);
    padding:3px 9px;border-radius:6px;margin-bottom:9px}
  .narrate h3{margin:0 0 6px;font-size:16px;font-weight:600;color:var(--ink)}
  .narrate h3 b{color:var(--accent);font-family:var(--mono);font-weight:600}
  .narrate p{margin:0;font-size:14px;color:var(--ink-soft);max-width:80ch}
  .narrate .links{margin-top:12px;display:flex;flex-wrap:wrap;gap:7px}
  .narrate .lnk{font-family:var(--mono);font-size:12px;background:var(--raise);
    border:1px solid var(--line-2);border-radius:7px;padding:5px 10px;color:var(--ink-soft)}
  .narrate .lnk b{color:var(--ink)}
  .narrate .lnk .w{color:var(--blue-bright)}
  .narrate .lnk .arrow{color:var(--dimmer);margin:0 5px}

  /* ---- grid of all heads ---- */
  .gridwrap{background:var(--panel);border:1px solid var(--line);border-radius:14px;
    padding:16px 18px 18px}
  .gridtitle{font-size:12.5px;color:var(--dim);margin:0 0 14px;display:flex;
    justify-content:space-between;flex-wrap:wrap;gap:8px}
  .gridtitle b{color:var(--ink-soft);font-weight:500}
  .gridtitle .legend{display:flex;align-items:center;gap:7px;font-family:var(--mono);font-size:10px;color:var(--dim)}
  .gridtitle .legend .bar{height:8px;width:64px;border-radius:4px;
    background:linear-gradient(90deg,var(--grid-empty),var(--blue))}
  .layers{display:flex;flex-direction:column;gap:3px}
  .layer-row{display:flex;align-items:center;gap:10px}
  .layer-label{width:28px;flex:none;font-family:var(--mono);font-size:10px;
    color:var(--dim);text-align:right}
  .heads{display:grid;grid-template-columns:repeat(12,1fr);gap:3px;flex:1}
  .head{aspect-ratio:1;background:var(--grid-empty);border-radius:3px;cursor:pointer;
    position:relative;overflow:hidden;border:1.5px solid transparent;
    transition:border-color .12s,transform .1s,box-shadow .12s}
  .head:hover{border-color:var(--blue);transform:scale(1.12);z-index:2;
    box-shadow:0 4px 14px -2px rgba(91,140,255,.5)}
  .head canvas{width:100%;height:100%;display:block;image-rendering:pixelated}
  .head.sel{border-color:var(--accent);box-shadow:0 0 0 2px var(--accent-soft),0 0 14px -2px var(--accent)}
  .head .badge{position:absolute;top:1px;left:1px;font-family:var(--mono);font-size:7px;
    line-height:1;padding:1px 2px;border-radius:2px;background:rgba(91,140,255,.85);
    color:#0b1220;font-weight:600;pointer-events:none}
  .colhdr{display:grid;grid-template-columns:repeat(12,1fr);gap:3px;flex:1;margin-bottom:4px}
  .colhdr span{font-family:var(--mono);font-size:9px;color:var(--dimmer);text-align:center}
  .colhdr-row{display:flex;align-items:center;gap:10px;margin-bottom:5px}
  .axhint{font-size:11px;color:var(--dimmer);margin:12px 2px 0;display:flex;
    justify-content:space-between;font-family:var(--mono)}
  .badgekey{font-size:11.5px;color:var(--dim);margin:14px 2px 0;display:flex;gap:16px;flex-wrap:wrap}
  .badgekey span b{color:var(--blue-bright);font-family:var(--mono)}

  /* ---- detail heatmap ---- */
  .detail{margin-top:18px;background:var(--panel);border:1px solid var(--line);
    border-radius:14px;padding:22px;min-height:90px}
  .detail .empty{color:var(--dim);font-size:13.5px;text-align:center;padding:26px 0}
  .detail .hint{color:var(--ink-soft);font-size:13px;margin:0 0 16px;max-width:80ch}
  .detail .hint b{color:var(--ink);font-weight:600}
  .bigmap{display:grid;gap:2px;overflow-x:auto;padding-bottom:6px}
  .bigmap .cell{aspect-ratio:1;border-radius:2px;min-width:15px}
  .bigmap .axhead{font-family:var(--mono);font-size:10px;color:var(--ink-soft);
    white-space:nowrap;display:flex;align-items:center;justify-content:flex-end;
    padding-right:7px;min-width:0}
  .bigmap .axtop{font-family:var(--mono);font-size:10px;color:var(--ink-soft);
    writing-mode:vertical-rl;transform:rotate(180deg);justify-self:center;
    max-height:80px;overflow:hidden}
  .err{color:#ff8f8f;font-family:var(--mono);font-size:13px;background:#3a1f22;
    border:1px solid #5e2c30;border-radius:8px;padding:12px 14px}

  footer{margin-top:44px;padding-top:18px;border-top:1px solid var(--line);
    font-size:12px;color:var(--dim)}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1><span class="dot"></span>attention-lens</h1>
    <span class="sub">gpt2-small · 12 layers × 12 heads</span>
    <span class="meta" id="device">·</span>
  </header>

  <p class="lede">A transformer reads by letting each word <b>look back</b> at earlier
  words and pull in meaning from them — that looking-back is <b>attention</b>. This runs
  your sentence through GPT-2 and shows all <b>144 attention heads</b>. Click any head and
  it'll tell you, <b>in plain English</b>, what that head is doing to your sentence.</p>

  <details class="explain">
    <summary><span class="chev">▶</span>30-second primer — what a head is, and how to read it</summary>
    <div class="body">
      <p><strong>A "head"</strong> is one of 144 little circuits (12 layers × 12 heads).
      Each learns its own rule for which earlier words a word should pay attention to.</p>
      <p><strong>Each head's map is a grid:</strong> every <strong>row is a query</strong>
      (a word doing the looking), every <strong>column is a key</strong> (a word it might
      look at). A bright cell = "this row's word paid a lot of attention to that column's
      word." It's a triangle because words can only look <em>backwards</em>, never at the future.</p>
      <p><strong>Recognizable head types:</strong> a bright <em>first column</em> = an
      idle "attention-sink" head; a bright line <em>just below the diagonal</em> = a
      "previous-token" head (short-term memory); late-layer heads where the last word
      points back at the answer = "name-mover" heads doing the actual reasoning.</p>
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
    <span class="ex" data-ex="The cat sat on the mat. The cat sat on the">a repeat (induction heads)</span> ·
    <span class="ex" data-ex="Paris is the capital of France. London is the capital of">a fact</span>
  </div>
  <div class="predline" id="pred"></div>

  <div class="narrate" id="narrate" style="display:none"></div>

  <div class="gridwrap" id="gridwrap" style="display:none">
    <div class="gridtitle">
      <span>All <b>144 heads</b> · <span id="gridtok"></span> · click any to read it</span>
      <span class="legend">weak<span class="bar"></span>strong attention</span>
    </div>
    <div class="colhdr-row" id="colhdrRow" style="display:none">
      <span class="layer-label"></span>
      <div class="colhdr" id="colhdr"></div>
    </div>
    <div class="layers" id="layers"></div>
    <div class="axhint"><span>↑ layer 0 reads the input</span><span>layer 11 writes the prediction ↓</span></div>
    <div class="badgekey" id="badgekey"></div>
  </div>

  <div class="detail" id="detail">
    <div class="empty">Run a sentence to begin.</div>
  </div>

  <footer>
    GPT-2 small via <a href="https://github.com/TransformerLensOrg/TransformerLens" target="_blank" rel="noopener">TransformerLens</a>.
    Weights are read straight from the model's cache — nothing faked or post-processed.
  </footer>
</div>

<script>
const $ = s => document.querySelector(s);
let DATA = null;          // last response
let SEL = null;           // [layer, head]

// blue ramp on dark: empty(#21252e) -> bright blue(#5b8cff), eased so faint reads
function lerp(a,b,t){return Math.round(a+(b-a)*t);}
function rgbAt(w){
  const t = Math.pow(Math.max(0,Math.min(1,w)), 0.7);
  return [lerp(33,91,t), lerp(37,140,t), lerp(46,255,t)];
}
function colorize(w){const c=rgbAt(w);return `rgb(${c[0]},${c[1]},${c[2]})`;}

function drawThumb(canvas, mat){
  const n = mat.length;
  canvas.width = n; canvas.height = n;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(n, n);
  for(let q=0;q<n;q++) for(let k=0;k<n;k++){
    const c=rgbAt(mat[q][k]); const i=(q*n+k)*4;
    img.data[i]=c[0]; img.data[i+1]=c[1]; img.data[i+2]=c[2]; img.data[i+3]=255;
  }
  ctx.putImageData(img,0,0);
}

// ---- the brain: classify a head's pattern + extract its strongest links ----
function analyzeHead(mat){
  const n = mat.length;
  if(n<2) return {type:'tiny', label:'', desc:'Too few tokens to show a pattern — try a longer sentence.', links:[]};
  let sink=0, prev=0, diag=0, sinkN=0, prevN=0, diagN=0;
  for(let q=0;q<n;q++){
    sink+=mat[q][0]; sinkN++;
    if(q>=1){ prev+=mat[q][q-1]; prevN++; }
    diag+=mat[q][q]; diagN++;
  }
  sink/=sinkN; prev/=prevN; diag/=diagN;

  // strongest off-trivial links (exclude BOS col 0, exclude self, prefer informative)
  const links=[];
  for(let q=1;q<n;q++){
    let bk=-1,bw=-1;
    for(let k=1;k<=q;k++){ if(k===q) continue; if(mat[q][k]>bw){bw=mat[q][k];bk=k;} }
    if(bk>=0 && bw>0.12) links.push({q,k:bk,w:bw});
  }
  links.sort((a,b)=>b.w-a.w);
  const top = links.slice(0,4);

  // classify
  let type,label,desc;
  const lastRow = mat[n-1];
  let lastBk=-1,lastBw=-1;
  for(let k=1;k<n-1;k++){ if(lastRow[k]>lastBw){lastBw=lastRow[k];lastBk=k;} }

  if(sink>0.55 && sink>prev && sink>diag){
    type='sink'; label='attention-sink head';
    desc='Almost every word dumps its attention onto the very first token (the bright left column). This is common, low-level plumbing — a kind of "do nothing" default. Not where the interesting computation happens.';
  } else if(prev>0.45 && prev>sink){
    type='prev'; label='previous-token head';
    desc='Each word attends to the one right before it — the bright line just under the diagonal. This is the model\'s short-term memory, copying positional/recency information forward. A building block other heads rely on.';
  } else if(diag>0.5 && diag>prev && diag>sink){
    type='self'; label='self / positional head';
    desc='Words mostly attend to themselves (the diagonal). Often a positional or pass-through head that mainly preserves a token\'s own information.';
  } else if(lastBk>=0 && lastBw>0.25){
    type='mover'; label='content / mover head';
    desc='This head moves specific information between words. Notably, the final word is looking back hard at an earlier content word — on the John/Mary sentence, that\'s the "name-mover" behaviour that helps produce the answer.';
  } else {
    type='mixed'; label='mixed / distributed head';
    desc='No single clean rule — attention is spread across several earlier words. Many mid-network heads blend signals like this; the strongest individual links are listed below.';
  }
  return {type,label,desc,links:top,scores:{sink,prev,diag}};
}

const BADGE = {sink:'sink', prev:'prev', mover:'move'};

function renderGrid(){
  const L=DATA.n_layers, H=DATA.n_heads;
  const colhdr=$('#colhdr'); colhdr.innerHTML='';
  for(let h=0;h<H;h++){const s=document.createElement('span');s.textContent='H'+h;colhdr.appendChild(s);}
  $('#colhdrRow').style.display='flex';

  let nSink=0,nPrev=0,nMove=0;
  const layers=$('#layers'); layers.innerHTML='';
  for(let l=0;l<L;l++){
    const row=document.createElement('div'); row.className='layer-row';
    const lab=document.createElement('div'); lab.className='layer-label'; lab.textContent='L'+l;
    const heads=document.createElement('div'); heads.className='heads';
    for(let h=0;h<H;h++){
      const cell=document.createElement('div'); cell.className='head';
      cell.dataset.l=l; cell.dataset.h=h;
      const a=analyzeHead(DATA.patterns[l][h]);
      cell.title=`L${l} H${h} — ${a.label} · click to read`;
      const cv=document.createElement('canvas'); drawThumb(cv,DATA.patterns[l][h]); cell.appendChild(cv);
      if(BADGE[a.type]){
        const b=document.createElement('div'); b.className='badge'; b.textContent=BADGE[a.type]; cell.appendChild(b);
        if(a.type==='sink')nSink++; else if(a.type==='prev')nPrev++; else if(a.type==='mover')nMove++;
      }
      cell.onclick=()=>selectHead(l,h);
      heads.appendChild(cell);
    }
    row.appendChild(lab); row.appendChild(heads); layers.appendChild(row);
  }
  $('#badgekey').innerHTML =
    `<span><b>sink</b> ${nSink} attention-sink</span>`+
    `<span><b>prev</b> ${nPrev} previous-token</span>`+
    `<span><b>move</b> ${nMove} content/mover</span>`+
    `<span style="color:var(--dimmer)">unlabelled = mixed/distributed</span>`;
}

function selectHead(l,h){
  SEL=[l,h];
  document.querySelectorAll('.head.sel').forEach(e=>e.classList.remove('sel'));
  const cell=document.querySelector(`.head[data-l="${l}"][data-h="${h}"]`);
  if(cell) cell.classList.add('sel');
  renderNarration(); renderDetail();
  $('#narrate').scrollIntoView({behavior:'smooth',block:'nearest'});
}

function renderNarration(){
  if(!SEL||!DATA) return;
  const [l,h]=SEL;
  const a=analyzeHead(DATA.patterns[l][h]);
  const toks=DATA.tokens.map(t=>t.replace(/ /g,'·'));
  let linkHtml='';
  if(a.links.length){
    linkHtml='<div class="links">'+a.links.map(x=>
      `<span class="lnk"><b>${esc(toks[x.q])}</b><span class="arrow">→</span><b>${esc(toks[x.k])}</b> <span class="w">${x.w.toFixed(2)}</span></span>`
    ).join('')+'</div>';
  }
  $('#narrate').style.display='block';
  $('#narrate').innerHTML=
    `<span class="tag">${a.label}</span>`+
    `<h3>layer <b>${l}</b> · head <b>${h}</b> — what it's doing</h3>`+
    `<p>${a.desc}</p>`+
    (linkHtml? `<p style="margin-top:12px;font-size:12.5px;color:var(--dim)">strongest links (which word looks at which, and how hard):</p>`+linkHtml : '');
}

function renderDetail(){
  const d=$('#detail'); if(!SEL||!DATA){return;}
  const [l,h]=SEL;
  const mat=DATA.patterns[l][h];
  const toks=DATA.tokens.map(t=>t.replace(/ /g,'·'));
  const n=toks.length;
  let html=`<p class="hint"><b>The full map.</b> Rows = query words (doing the looking), columns = key words (being looked at). Brighter = more attention. The blank upper-right triangle is the causal mask (no peeking ahead). Hover a cell for the exact weight.</p>`;
  const cols=`minmax(80px,auto) repeat(${n}, minmax(15px,1fr))`;
  html+=`<div class="bigmap" style="grid-template-columns:${cols}">`;
  html+=`<div></div>`;
  for(let k=0;k<n;k++) html+=`<div class="axtop">${esc(toks[k])}</div>`;
  for(let q=0;q<n;q++){
    html+=`<div class="axhead">${esc(toks[q])}</div>`;
    for(let k=0;k<n;k++){
      const w=mat[q][k];
      html+=`<div class="cell" style="background:${colorize(w)}" title="${esc(toks[q])} → ${esc(toks[k])}: ${w.toFixed(3)}"></div>`;
    }
  }
  html+=`</div>`;
  d.innerHTML=html;
}

function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

// auto-pick the most illustrative head to open first: prefer a clear mover, else prev, else last layer
function pickInterestingHead(){
  let best=null;
  for(let l=DATA.n_layers-1;l>=0;l--) for(let h=0;h<DATA.n_heads;h++){
    const a=analyzeHead(DATA.patterns[l][h]);
    if(a.type==='mover'){ if(!best){best=[l,h];} }
  }
  if(best) return best;
  for(let l=2;l<DATA.n_layers;l++) for(let h=0;h<DATA.n_heads;h++){
    if(analyzeHead(DATA.patterns[l][h]).type==='prev') return [l,h];
  }
  return [DATA.n_layers-1,0];
}

async function run(){
  const text=$('#txt').value.trim(); if(!text)return;
  const btn=$('#run'); btn.disabled=true; btn.textContent='running…';
  $('#pred').textContent='';
  try{
    const r=await fetch('/api/attn',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
    const j=await r.json();
    if(j.error){$('#detail').innerHTML=`<div class="err">error: ${esc(String(j.error))}</div>`;return;}
    DATA=j; SEL=null;
    $('#device').textContent='gpt2-small';
    $('#pred').innerHTML=`the model's next-token guess → <b>${esc(j.prediction.replace(/ /g,'·'))}</b>`;
    $('#gridwrap').style.display='block';
    $('#gridtok').textContent=`${DATA.tokens.length} tokens`;
    renderGrid();
    const pick=pickInterestingHead();
    selectHead(pick[0],pick[1]);
  }catch(e){
    $('#detail').innerHTML=`<div class="err">request failed: ${esc(String(e))}</div>`;
  }finally{btn.disabled=false; btn.textContent='run';}
}

$('#run').onclick=run;
$('#txt').addEventListener('keydown',e=>{if(e.key==='Enter')run();});
document.querySelectorAll('.ex').forEach(el=>el.onclick=()=>{$('#txt').value=el.dataset.ex; run();});
run();
</script>
</body>
</html>"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
