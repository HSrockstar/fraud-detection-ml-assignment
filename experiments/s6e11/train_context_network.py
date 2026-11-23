"""One neural classifier: additive spline logit with an optional rank-eight interaction."""
from pathlib import Path
import gc

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from compare_tree_member import bootstrap_delta
from train_rate_glm import rates


AMOUNTS = ("amount_usd", "account_balance_usd", "distance_from_home_km",
           "hours_since_last_txn", "merchant_risk_score", "velocity_score")
MODES = ("additive", "interaction")


def blocks(table):
    def codes(s):
        return pd.factorize(s, use_na_sentinel=False)[0].astype(np.int32)
    monetary = []
    for name in AMOUNTS:
        v = table[name]
        cents = np.rint(v*100).astype(np.int64)
        whole = cents//100
        length = np.floor(np.log10(np.maximum(whole, 1))).astype(int)+1
        lead = whole//(10**np.maximum(length-2, 0))
        for key in (cents,whole,cents%100,cents//1000,cents//10%10,cents%10,
                    np.floor(10*np.log1p(v)),length*100+lead,length*100+cents%100):
            monetary.append(codes(key))
    context = [codes(table[c]) for c in table if c not in AMOUNTS]
    return monetary, context


def encoded_inputs(table, y, fit, score, monetary, context):
    keys = monetary+context
    fit_rate = np.empty((len(fit),len(keys)),dtype=np.float32)
    for a,b in StratifiedKFold(5,shuffle=True,random_state=42).split(fit,y[fit]):
        fit_rate[b] = rates(keys,fit[a],fit[b],y)
    val_rate = rates(keys,fit,score,y)
    # Location and spread estimated strictly on the fit rows.
    inputs=[]
    raw = np.log1p(table[list(AMOUNTS)].to_numpy(dtype=np.float32))
    context_raw = table.drop(columns=list(AMOUNTS)).select_dtypes(include=["number","bool"]).to_numpy(dtype=np.float32)
    for idx,te in [(fit,fit_rate),(score,val_rate)]:
        inputs.append(np.column_stack([raw[idx],te[:,:len(monetary)],context_raw[idx],te[:,len(monetary):]]))
    mean,scale=inputs[0].mean(axis=0,dtype=np.float64),inputs[0].std(axis=0,dtype=np.float64)
    for i,x in enumerate(inputs):
        inputs[i]=torch.as_tensor(np.clip((x-mean)/np.maximum(scale,.05),-8,8).astype(np.float32),device="cuda")
    return inputs[0], inputs[1], raw.shape[1] + len(monetary)


class ContextNet(nn.Module):
    def __init__(self, dim, boundary, interaction, prior):
        super().__init__()
        self.boundary = boundary * 6
        self.register_buffer("knots",torch.tensor([-2.,-1.,0.,1.,2.]))
        self.additive = nn.Linear(dim*6,1)
        nn.init.zeros_(self.additive.weight)
        nn.init.constant_(self.additive.bias,float(np.log(prior/(1-prior))))
        self.interaction = interaction
        if interaction:
            self.amount = nn.Linear(self.boundary,8,bias=False)
            self.context = nn.Linear(dim*6-self.boundary,8,bias=False)
            nn.init.normal_(self.amount.weight,std=.01)
            nn.init.normal_(self.context.weight,std=.01)

    def forward(self,x):
        # Each field learns its own piecewise-linear risk curve.
        basis = torch.cat([x.unsqueeze(-1),torch.relu(x.unsqueeze(-1)-self.knots)],dim=-1).flatten(1)
        value=self.additive(basis).squeeze(1)
        if self.interaction:
            a=self.amount(basis[:,:self.boundary])
            c=self.context(basis[:,self.boundary:])
            value=value+(torch.tanh(a)*torch.tanh(c)).sum(1)
        return value


def train(x, v, labels, boundary, mode):
    torch.manual_seed(42)
    model=ContextNet(x.shape[1],boundary,mode=="interaction",labels.mean()).cuda()
    target=torch.tensor(labels,dtype=torch.float32,device="cuda")
    opt=torch.optim.AdamW(model.parameters(),lr=.0012,weight_decay=.02)
    gen=torch.Generator(device="cuda").manual_seed(2026)
    for epoch in range(12):
        opt.param_groups[0]["lr"]=.00015+.00105*(1+np.cos(np.pi*epoch/11))/2
        for batch in torch.randperm(len(x),device="cuda",generator=gen).split(2048):
            opt.zero_grad(set_to_none=True)
            loss=nn.functional.binary_cross_entropy_with_logits(model(x[batch]),target[batch])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return torch.cat([model(batch).sigmoid() for batch in v.split(4096)]).cpu().numpy()


def main():
    out=Path("runs/context_network")
    out.mkdir(exist_ok=True)
    d=Path(".kaggle-cache/competitions/playground-series-s6e11")
    tr=pd.read_csv(d/"train.csv"); test=pd.read_csv(d/"test.csv")
    ids=tr.id.to_numpy(); y=tr.is_fraud.to_numpy(dtype=int)
    folds=pd.read_csv("splits/seed42.csv").set_index("id").loc[ids,"fold"].to_numpy()
    tab=pd.concat([tr[test.columns],test],ignore_index=True).drop(columns="id")
    money,context=blocks(tab)
    for f in range(5):
        path=out/f"fold{f}.npz"
        if path.exists(): continue
        a=np.flatnonzero((folds>=0)&(folds!=f)); b=np.flatnonzero(folds==f)
        x,v,boundary=encoded_inputs(tab,y,a,b,money,context)
        preds={}
        for mode in MODES:
            preds[mode]=train(x,v,y[a],boundary,mode)
            print(f,mode,roc_auc_score(y[b],preds[mode]),flush=True)
        tmp=path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp,id=ids[b],**preds);tmp.replace(path)
        del x,v
        gc.collect()
    dev=folds>=0
    pred={}
    for mode in MODES:
        rows=[]
        for f in range(5):
            with np.load(out/f"fold{f}.npz") as z:
                rows.append(pd.Series(z[mode],index=z["id"]))
        pred[mode]=pd.concat(rows,verify_integrity=True).loc[ids[dev]].to_numpy()
    scores={k:roc_auc_score(y[dev],v) for k,v in pred.items()}
    print("OOF",scores,"interaction delta CI",bootstrap_delta(pred["additive"],pred["interaction"],y[dev].astype(bool)),flush=True)
    mode=max(scores,key=scores.get)
    x,v,boundary=encoded_inputs(tab,y,np.arange(len(y)),np.arange(len(y),len(tab)),money,context)
    p=train(x,v,y,boundary,mode)
    sample=pd.read_csv(d/"sample_submission.csv")
    assert sample.id.is_unique and sample.id.equals(test.id) and np.isfinite(p).all()
    sample.is_fraud=p
    sample.to_csv(out/f"single_{mode}.csv",index=False)
    print("full refit",mode,len(sample),flush=True)


if __name__=="__main__":
    main()
