import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tabm")
pytest.importorskip("rtdl_revisiting_models")

from fraud.neural import NeuralPreprocessor, batch_probabilities, fit_network, predict_frame, restore_network
from fraud.neural_experiments import load_bundle, save_bundle


def sample_frame(n=256):
    rng=np.random.default_rng(42)
    x=pd.DataFrame({"amount":rng.lognormal(1,1,n),"risk":rng.normal(size=n),
                    "flag":rng.random(n)>0.7,"category":np.resize(np.array(["A","B","C"]),n)})
    y=((x.risk>0.7)&x.flag).astype(int).to_numpy()
    return x,y


def test_neural_preprocessing_unknown_category_and_boolean_are_safe():
    x,_=sample_frame()
    prep=NeuralPreprocessor(42).fit(x)
    vocab=dict(prep.vocabularies["category"])
    valid=pd.DataFrame({"amount":[np.nan,1e20],"risk":[0.,100.],"flag":[False,True],"category":["UNSEEN","A"]})
    numeric,categorical=prep.transform(valid)
    assert categorical[0,0]==0
    assert prep.vocabularies["category"]==vocab
    assert numeric[:,-1].tolist()==[0.,1.]
    assert np.isfinite(numeric).all()
    with pytest.raises(ValueError,match="缺少输入列"):
        prep.transform(valid.drop(columns="amount"))


@pytest.mark.skipif(not torch.cuda.is_available(),reason="GPU 测试需要 CUDA")
def test_member_probabilities_are_averaged_after_sigmoid():
    class Members(torch.nn.Module):
        def forward(self,numeric,categorical):
            return torch.tensor([-5.,1.],device=numeric.device).reshape(1,2,1).expand(len(numeric),-1,-1)
    data=[torch.zeros((3,1),device="cuda"),torch.zeros((3,1),device="cuda",dtype=torch.long)]
    probability=batch_probabilities(Members().to("cuda"),data,2)
    expected=(1/(1+np.exp(5))+1/(1+np.exp(-1)))/2
    assert np.allclose(probability,expected)
    assert not np.allclose(probability,1/(1+np.exp(2)))


@pytest.mark.skipif(not torch.cuda.is_available(),reason="GPU 测试需要 CUDA")
@pytest.mark.parametrize("kind",["mlp","ft_transformer","tabm"])
def test_architecture_cuda_training_and_checkpoint_round_trip(kind,tmp_path):
    x,y=sample_frame()
    params={"hidden":[32,16,8],"embedding_dim":4,"dropout":0.1,"batch_size":64,"lr":0.001,"weight_decay":0.0001,
            "d_block":32,"n_blocks":2,"attention_n_heads":4,"attention_dropout":0.1,"ffn_dropout":0.1,
            "k":4,"n_bins":8,"num_embedding_dim":4}
    model,bundle=fit_network(kind,params,x,y,42,epochs=1)
    probability=predict_frame(model,bundle["preprocessor"],x,128)
    assert probability.shape==(len(x),)
    assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
    assert bundle["peak_memory_mib"]>0 and "RTX" in bundle["training_device"]
    path=tmp_path/f"{kind}.pt"
    save_bundle(path,bundle)
    restored=load_bundle(path)
    replay=predict_frame(restore_network(restored),restored["preprocessor"],x,128)
    assert np.allclose(probability,replay,atol=1e-6)
