"""Limited structural checks for the deposited output bundle, not pipeline replication."""
from pathlib import Path
import json, csv
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
def rows(name):
    with (ROOT / "results" / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
emb = np.load(ROOT / "embeddings" / "embeddings.npy", mmap_mode="r", allow_pickle=False)
lifecycle = rows("lifecycle_results.csv")
clustering = rows("clustering_evaluation.csv")
with (ROOT / "results" / "portfolio_analysis.json").open(encoding="utf-8") as f:
    portfolio = json.load(f)
with (ROOT / "results" / "trajectories.json").open(encoding="utf-8") as f:
    trajectories = json.load(f)
assert emb.shape == (7747, 768), emb.shape
assert len(lifecycle) == 11
assert sum(int(x["n_grants"]) for x in lifecycle) == emb.shape[0]
assert len(clustering) == 17 and [int(x["k"]) for x in clustering] == list(range(4, 21))
assert int(max(clustering, key=lambda x:float(x["score"]))["k"]) == 11
assert sum(sum(map(int, yearly.values())) for yearly in trajectories.values()) == 7655
sizes = np.array([int(x["n_grants"]) for x in lifecycle])
hhi = float(np.sum((100 * sizes / sizes.sum()) ** 2))
assert abs(hhi - float(portfolio["HHI"])) < .1, (hhi, portfolio["HHI"])
shares = {name: sum(int(x["n_grants"]) for x in lifecycle if x["lifecycle_stage"]==name)/7747 for name in portfolio["ideal_stage_shares"]}
balance = 1 - .5 * sum(abs(shares[name]-target) for name,target in portfolio["ideal_stage_shares"].items())
assert abs(balance - float(portfolio["balance_score"])) < .001
print("PASS: 7,747 x 768 embedding matrix")
print("PASS: 11 cluster sizes sum to 7,747; k=11 is the maximum supplied score")
print("PASS: year-by-cluster counts sum to 7,655")
print(f"PASS: HHI={hhi:.3f}; portfolio balance={balance:.6f}")
print("LIMITATION: original raw data and source code are not present, so these are not end-to-end replication tests.")
