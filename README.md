# attention-lens

A toy for watching GPT-2 small's attention heads light up live. Type a sentence,
see all 144 heads (12 layers × 12 heads) render their attention patterns as a grid,
click any head to blow it up — and the tool **tells you in plain English what that
head is doing to your sentence**.

It auto-classifies each head (attention-sink / previous-token / content-mover /
mixed), badges them right on the grid so the 144-head wall reads as a labelled map
instead of noise, and spells out the strongest word→word attention links with their
weights. On load it auto-opens a genuinely illustrative head and explains it, so you
get a worked example immediately instead of staring at an abstract matrix.

This is the **playground** version of mech interp. It wires up TransformerLens
(`from_pretrained`, `run_with_cache`, `cache["pattern", layer]`) so that when you
move to the real IOI-circuit work, the model's already running and the cache is
already flowing — zero activation energy.

Dark UI, Inter type, a blue attention ramp; single-file Flask app.

## Run it

```bash
cd attention-lens
python -m venv venv && source venv/bin/activate   # or: uv venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

First run downloads GPT-2 small (~500MB) via TransformerLens. Then open
**http://localhost:5000**. With a CUDA GPU it loads in seconds and each run is
instant; on CPU it still works, just slower.

It binds `0.0.0.0:5000`, so if you run it on a remote box you can reach it from
another machine on the same network at `http://<that-box>:5000`. (Binding to all
interfaces is for LAN convenience — don't expose it to the public internet
without putting auth/a reverse proxy in front; see the note below.)

## What to look for

Three example sentences are one-click in the UI. The head types the tool labels for you:

- **Attention-sink heads** (`sink` badge) — a bright first column: nearly everything
  dumps attention onto the first token. Common, low-level, basically idle plumbing.
- **Previous-token heads** (`prev` badge) — a bright line just below the diagonal:
  each token attends to the one right before it. The model's short-term memory.
- **Content / mover heads** (`move` badge) — they move information between specific
  words. On the John/Mary sentence, later-layer movers are where the final token
  looks back at " Mary" — (part of) the circuit that produces the answer.
- **Induction heads** — the reasoning primitive. On repeated text ("the cat sat. the
  cat ___") they attend from the current token back to *what came after the last
  occurrence*. Try the repeat example and hunt the mid layers (≈5–6).

The auto-classifier is a heuristic (column/diagonal/last-row mass) — it's a fast
guide, not ground truth. Once a head makes you go "wait, *that's* the one doing the
work," you're one `pip install` away from the real IOI project (activation patching
to *prove* it, not just eyeball it).

## Files
- `app.py` — the whole thing (Flask server + model + the single-page viz + the
  in-browser head classifier, one file)
- `test_mock.py` — verifies the data pipeline + routes with a mock model (no torch needed)
- `dev_screenshot.py` — dev-only: render the UI headless and screenshot it (mocks the
  model, so no GPU/download needed)
- `requirements.txt` · `requirements-dev.txt`

## Tests

```bash
pip install flask numpy        # the test mocks torch + transformer_lens, so those aren't needed
python test_mock.py
```

The test stubs out TransformerLens with a mock that mirrors its API, then checks
the data pipeline (token handling, `[layer][head][q][k]` shape, the causal mask)
and the Flask routes — so you can verify the plumbing without downloading the model
or having a GPU.

## A note on serving

The app binds `0.0.0.0:5000` and runs Flask's development server with `debug=False`.
That's fine for local/LAN use. **Do not expose it directly to the public internet** —
there's no authentication and `get_attention` runs a full forward pass per request, so
an open endpoint is a trivial resource-exhaustion target. If you need remote access,
put it behind a reverse proxy with auth, or tunnel to it (Tailscale, SSH, etc.).

## Developing the UI

To iterate on the frontend without a GPU or the model download, there's a Playwright
harness that mocks the model, renders the page headless, and screenshots it:

```bash
pip install -r requirements-dev.txt
python -m playwright install chromium
python dev_screenshot.py            # -> shots/full.png, shots/detail.png
```

The mock (shared with `test_mock.py`) gives a few heads recognizable structure
(sink columns, sub-diagonals) so the viz and the classifier render realistically.

## Extending it

A natural next step: a dropdown to switch models (`gpt2`, `gpt2-medium`,
`pythia-160m`) and reload without restarting the server. The viz rendering is
model-agnostic — it reads `n_layers`/`n_heads` off the response — so adding more
models is mostly a load-and-cache concern on the backend. The head classifier in
`app.py` (`analyzeHead`) is heuristic and an easy place to improve next — e.g. real
induction-head detection on repeated-token sentences.

## License

MIT — see [LICENSE](LICENSE).
