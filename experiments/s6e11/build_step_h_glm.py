"""goodpjw2008 step H v35 (reported public AUC .95504), blended with our GLM."""
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    base_path = Path(".kaggle-cache/notebooks/goodpjw2008/s6e11-board-model-realmlp-lb-0-95560/output/versions/35/submission_step_h.csv")
    assert sha256(base_path.read_bytes()).hexdigest() == "ae0cafb7f8dfad06c919d86e002c10bea3ae405da5eb27dddcfbe331e55c2b0e"
    base = pd.read_csv(base_path)
    template = pd.read_csv(".kaggle-cache/competitions/playground-series-s6e11/sample_submission.csv")
    assert base.columns.equals(template.columns) and base.id.equals(template.id) and base.id.is_unique
    assert base.is_fraud.between(0, 1).all()
    with np.load("runs/self_rate_glm_full_test.npz") as model:
        assert np.array_equal(model["id"], base.id) and np.isfinite(model["test"]).all()
        glm_rank = pd.Series(model["test"]).rank(pct=True)
    base_rank = base.is_fraud.rank(pct=True)
    blend = base.assign(is_fraud=.95 * base_rank + .05 * glm_rank)
    assert len(blend) == 299984 and blend.is_fraud.between(0, 1).all()
    blend.to_csv("runs/step_h95_self_rate_glm05.csv", index=False)
    print(len(blend), "rows; rank correlation between members:", base_rank.corr(glm_rank))


if __name__ == "__main__":
    main()
