"""Test the app's logic with a mock that mimics TransformerLens's API exactly.
Verifies: token handling, cache["pattern", layer] shape, causal mask, JSON shape,
prediction extraction, and that the Flask routes respond correctly.
"""
import sys, types, json

# ---- build a fake `torch` and `transformer_lens` with the same surface the app uses ----
import numpy as np

fake_torch = types.ModuleType("torch")
fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
class _NoGrad:
    def __enter__(self): return self
    def __exit__(self,*a): return False
fake_torch.no_grad = lambda: _NoGrad()
sys.modules["torch"] = fake_torch

N_LAYERS, N_HEADS = 12, 12

class _Tensor:
    """Minimal stand-in supporting the ops the app calls."""
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
        # causal lower-triangular attention, row-normalized, per (layer,head)
        self.data={}
        for l in range(N_LAYERS):
            heads=[]
            for h in range(N_HEADS):
                m=np.tril(np.random.rand(seq,seq))
                m=m/m.sum(axis=1,keepdims=True)
                heads.append(m)
            self.data[("pattern",l)]=_Tensor(np.stack(heads)[None])  # [1,head,q,k]
    def __getitem__(self,key): return self.data[key]

class MockModel:
    def __init__(self): self.cfg=types.SimpleNamespace(n_layers=N_LAYERS,n_heads=N_HEADS)
    def eval(self): pass
    def to_tokens(self,text): 
        self._seq=len(text.split())+1   # +1 BOS
        return _Tensor(np.arange(self._seq)[None])
    def to_str_tokens(self,text): return ["<|endoftext|>"]+[" "+w for w in text.split()]
    def to_string(self,ids): return " Mary"
    def run_with_cache(self,tokens,return_type=None):
        seq=tokens.shape[1]; return None,_Cache(seq)
    def __call__(self,tokens,return_type=None):
        seq=tokens.shape[1]; return _Tensor(np.random.rand(1,seq,50257))

fake_tl=types.ModuleType("transformer_lens")
fake_tl.HookedTransformer=types.SimpleNamespace(from_pretrained=lambda name,device=None: MockModel())
sys.modules["transformer_lens"]=fake_tl

# ---- now import the app and exercise it ----
import importlib.util, os
_APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
spec=importlib.util.spec_from_file_location("app", _APP_PATH)
app_mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_mod)

# 1) data pipeline
d=app_mod.get_attention("When John and Mary went to the store, John gave a drink to")
seq=len(d["tokens"])
assert d["n_layers"]==12 and d["n_heads"]==12, "layer/head count wrong"
assert len(d["patterns"])==12, "wrong #layers in patterns"
assert len(d["patterns"][0])==12, "wrong #heads"
assert len(d["patterns"][0][0])==seq and len(d["patterns"][0][0][0])==seq, "pattern not seq x seq"
# causal check: row q should have ~zero weight on keys > q (upper triangle)
import numpy as np
p=np.array(d["patterns"][3][5])
upper=np.triu(p,k=1)
assert upper.sum()<1e-9, "causal mask violated (attends to future)"
assert d["prediction"]==" Mary", "prediction extraction broke"
print(f"OK pipeline: seq={seq}, patterns shape [12][12][{seq}][{seq}], causal verified, pred='{d['prediction']}'")

# 2) flask routes
client=app_mod.app.test_client()
r=client.get("/"); assert r.status_code==200 and b"attention-lens" in r.data, "index route broke"
print("OK GET / serves the page")
r=client.post("/api/attn",json={"text":"hello world"})
j=r.get_json(); assert r.status_code==200 and "patterns" in j, "api route broke"
print(f"OK POST /api/attn returns patterns (seq={len(j['tokens'])})")
r=client.post("/api/attn",json={"text":""})
assert r.status_code==400, "empty-text guard broke"
print("OK empty-text guard returns 400")
r=client.post("/api/attn",json={"text":"x"*(app_mod.MAX_CHARS+1)})
assert r.status_code==400, "over-length guard broke"
print(f"OK over-length guard returns 400 (>{app_mod.MAX_CHARS} chars)")
r=client.post("/api/attn",data="not json",content_type="text/plain")
assert r.status_code==400, "non-JSON body should 400, not 500/crash"
print("OK non-JSON body returns 400 (get_json silent)")

print("\nALL CHECKS PASSED — pipeline + routes are sound. Ready for real GPT-2 on a GPU.")
