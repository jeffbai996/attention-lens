"""Dev-only: boot the app with a mocked model and screenshot the UI with Playwright.

Lets you iterate on the frontend (HTML/CSS/copy) without a GPU or the ~500MB
model download — it reuses the same TransformerLens mock as test_mock.py, runs the
Flask app in a background thread, drives it with a headless browser, and writes
screenshots to ./shots/.

    python dev_screenshot.py

Not part of the shipped app; requires `pip install playwright && playwright install chromium`.
"""
import sys, types, threading, time, os
import numpy as np

# ---- same mock surface as test_mock.py (torch + transformer_lens) ----
fake_torch = types.ModuleType("torch")
fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
class _NoGrad:
    def __enter__(self): return self
    def __exit__(self, *a): return False
fake_torch.no_grad = lambda: _NoGrad()
sys.modules["torch"] = fake_torch

N_LAYERS, N_HEADS = 12, 12

class _Tensor:
    def __init__(self, arr): self.arr = np.asarray(arr)
    def __getitem__(self, idx): return _Tensor(self.arr[idx])
    def cpu(self): return self
    def float(self): return self
    def tolist(self): return self.arr.tolist()
    def argmax(self): return int(self.arr.argmax())
    @property
    def shape(self): return self.arr.shape

class _Cache:
    def __init__(self, seq):
        self.data = {}
        for l in range(N_LAYERS):
            heads = []
            for h in range(N_HEADS):
                # give a few heads recognizable structure so the viz looks real
                if h % 4 == 0:      # previous-token-ish: sub-diagonal
                    m = np.eye(seq, k=-1) + 0.02
                elif h % 4 == 1:    # attention-sink: first column hot
                    m = np.zeros((seq, seq)); m[:, 0] = 1.0; m += 0.02
                else:
                    m = np.tril(np.random.rand(seq, seq))
                m = np.tril(m)
                m = m / m.sum(axis=1, keepdims=True)
                heads.append(m)
            self.data[("pattern", l)] = _Tensor(np.stack(heads)[None])
    def __getitem__(self, key): return self.data[key]

class MockModel:
    def __init__(self): self.cfg = types.SimpleNamespace(n_layers=N_LAYERS, n_heads=N_HEADS)
    def eval(self): pass
    def to_tokens(self, text):
        self._seq = len(text.split()) + 1
        return _Tensor(np.arange(self._seq)[None])
    def to_str_tokens(self, text): return ["<|endoftext|>"] + [" " + w for w in text.split()]
    def to_string(self, ids): return " Mary"
    def run_with_cache(self, tokens, return_type=None):
        return None, _Cache(tokens.shape[1])
    def __call__(self, tokens, return_type=None):
        return _Tensor(np.random.rand(1, tokens.shape[1], 50257))

fake_tl = types.ModuleType("transformer_lens")
fake_tl.HookedTransformer = types.SimpleNamespace(from_pretrained=lambda name, device=None: MockModel())
sys.modules["transformer_lens"] = fake_tl

# ---- import the app and run it in a thread ----
import importlib.util
_APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
spec = importlib.util.spec_from_file_location("app", _APP)
app_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_mod)

PORT = 5057
threading.Thread(
    target=lambda: app_mod.app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False),
    daemon=True,
).start()
time.sleep(1.5)

from playwright.sync_api import sync_playwright

os.makedirs("shots", exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 1600}, device_scale_factor=2)
    page.goto(f"http://127.0.0.1:{PORT}/", wait_until="networkidle")
    page.wait_for_timeout(1200)          # let the default sentence render
    page.screenshot(path="shots/full.png", full_page=True)
    print("wrote shots/full.png")
    # also a detail-panel close-up (a head is auto-selected on load)
    try:
        page.locator(".detail").screenshot(path="shots/detail.png")
        print("wrote shots/detail.png")
    except Exception as e:
        print("detail shot skipped:", e)
    browser.close()
print("done")
