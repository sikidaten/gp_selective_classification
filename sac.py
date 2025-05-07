from typing import List, Dict
from torchmetrics import Metric
import torch
import numpy as np


class SelectiveAccuracyConstraint(Metric):
    """Selective AccuracyConstraint"""

    def __init__(self, risk=0.99):
        super().__init__()
        self.risk = risk
        self.add_state("conf", default=list())
        self.add_state("iscorrect", default=list())

    def update(self, conf, iscorrect):
        self.conf.append(conf.detach().cpu().numpy())
        self.iscorrect.append(iscorrect.detach().cpu().numpy())

    def compute(self) -> Dict[str, float]:
        conf = np.concatenate(self.conf)
        iscorrect = np.concatenate(self.iscorrect)
        sortidx = conf.argsort()[::-1]
        iscorrect = iscorrect[sortidx]
        cumcumisc = np.cumsum(iscorrect, 0)
        cumcumisc = cumcumisc / \
            np.arange(1, cumcumisc.shape[-1]+1).astype(np.float32)
        # print(cumcumisc)
        if (cumcumisc >= self.risk).sum() < 1:
            return 0
        # because torch.argmax return FIRST index
        rmaxidx = (np.flip(cumcumisc, axis=(-1,)) >=
                   self.risk).astype(np.float32).argmax()
        numcoverage = iscorrect.shape[-1] - (rmaxidx+1)+1
        return numcoverage/iscorrect.shape[-1]


if __name__ == "__main__":
    sacmtr = SelectiveAccuracyConstraint(risk=1)
    iscorrect = torch.tensor([1, 1, 0, 1])
    conf = torch.tensor([0.9, 0.8, 0.7, 0.3])
    # sacmtr.update(conf, iscorrect)
    sacmtr.update(conf, iscorrect)
    sacv = sacmtr.compute()
    print(sacv)