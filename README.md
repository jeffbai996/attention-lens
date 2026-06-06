# attention-lens

A toy for watching GPT-2 small's attention heads light up live. Type a sentence,
see all 144 heads (12 layers × 12 heads) render their attention patterns as a grid,
click any head to blow it up and see which tokens attend to which.

This is the **playground** version of mech interp. It wires up TransformerLens
(`from_pretrained`, `run_with_cache`, `cache["pattern", layer]`) so that when you
move to the real IOI-circuit work, the model's already running and the cache is
already flowing — zero activation energy.

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

## What to actually look for (the fun part)

Type the default sentence ("When John and Mary went to the store, John gave a drink to")
and the model predicts " Mary". Now go hunting in the grid:

- **Layer 0 heads** — mostly boring/positional. Many attend heavily to the first
  token (the "attention sink" / BOS) or to the immediately previous token. That's
  the low-level plumbing.
- **"Previous-token heads"** — look for heads whose pattern is a bright line just
  below the diagonal. Each token attending to the one right before it.
- **"Induction heads"** (usually layers 5–6ish) — the reasoning primitive. On
  repeated text ("the cat sat. the cat ___") they attend from the current token
  back to *what came after the last occurrence* of it. Try a sentence with a
  repeated word and watch for it.
- **The IOI name-mover heads** (later layers) — on the John/Mary sentence, look
  for heads in the last few layers where the final token attends strongly back to
  " Mary". Those are (part of) the circuit that does the actual answer. You're
  *seeing the IOI circuit* before you formally replicate it.

That last bullet is the bridge: once a head's pattern makes you go "wait, *that's*
the one doing the work" — you're one `pip install` away from the real IOI project
(activation patching to *prove* it, not just eyeball it).

## Files
- `app.py` — the whole thing (Flask server + model + the single-page viz, one file)
- `test_mock.py` — verifies the data pipeline + routes with a mock model (no torch needed)
- `requirements.txt`

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

## Extending it

A natural next step: a dropdown to switch models (`gpt2`, `gpt2-medium`,
`pythia-160m`) and reload without restarting the server. The viz rendering is
model-agnostic — it reads `n_layers`/`n_heads` off the response — so adding more
models is mostly a load-and-cache concern on the backend.

## License

MIT — see [LICENSE](LICENSE).
