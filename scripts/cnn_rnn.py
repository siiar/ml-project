%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

from fastai import *
from fastai.vision import *
import pretrainedmodels
from fastai.callbacks.tracker import *
from fastai.callbacks.hooks import *
from torch.utils.data import Sampler, BatchSampler
import json

from utils import *

PATH = Path('data/txt')
#Create ImageDataBunch using the data block API

sz_lrg, sz_small = 192, 96
bs = 520
batch_stats_lrg = pd.read_pickle(f'data/batch_stats_{sz_lrg}.pkl')
batch_stats_lrg = batch_stats_lrg[0][:, None, None], batch_stats_lrg[1][:, None, None]

batch_stats_small = pd.read_pickle(f'data/batch_stats_{sz_small}.pkl')
batch_stats_small = batch_stats_small[0][:, None, None], batch_stats_small[1][:, None, None]

country2code = {v: k for k, v in enumerate(pd.read_pickle('data/countrycodes.pkl'))}
codes_count = len(country2code); codes_count

seq_max_len = 68

def create_func(path):
    with open(path) as f: j = json.load(f)
    
    drawing_lrg = list2drawing(j['drawing'], size=sz_lrg, time_color=True)
    tensor_lrg = drawing2tensor(drawing_lrg)
    tensor_lrg.div_(255).sub_(batch_stats_lrg[0]).div_(batch_stats_lrg[1])
    
    drawing_small = list2drawing(j['drawing'], size=sz_small, time_color=True)
    tensor_small = drawing2tensor(drawing_small)
    tensor_small.div_(255).sub_(batch_stats_small[0]).div_(batch_stats_small[1])
    
    ary = drawing2seq(j['drawing'])
    seq = np.zeros((seq_max_len, 3))
    seq[-ary.shape[0]:, :] = ary[:seq_max_len, :]
    
    country_code = country2code[j['countrycode']] if j['countrycode'] in country2code.keys() else 0
    return tensor_lrg, tensor_small, seq.astype(np.float32), country_code

%%time
item_list = ItemList.from_folder(PATH/'train', create_func=create_func)
item_lists = item_list.random_split_by_pct(0.002, seed=0)


classes = pd.read_pickle('data/classes.pkl')

label_lists = item_lists.label_from_folder(classes=classes)

test_items = ItemList.from_folder(PATH/'test', create_func=create_func)
label_lists.add_test(test_items);

train_dl = DataLoader(
    label_lists.train,
    num_workers=64,
    batch_sampler=BatchSampler(RandomSamplerWithEpochSize(label_lists.train, 200_000), bs, True)
)
valid_dl = DataLoader(label_lists.valid, bs, False, num_workers=12)
test_dl = DataLoader(label_lists.test, bs, False, num_workers=12)

data_bunch = ImageDataBunch(train_dl, valid_dl, test_dl)

#TRAINING
name = 'resnet50-incv4'
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, k)
        self.bn = nn.BatchNorm1d(c_out)
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.bn(x)
        return x

class RNNBlock(nn.Module):
    def __init__(self, inp_sz, hidden_sz=128, num_l=1, bi=False, last_h=False):
        super().__init__()
        self.rnn = nn.LSTM(input_size=inp_sz, hidden_size=hidden_sz, num_layers=num_l, bidirectional=bi)
        self.last_h = last_h
    def forward(self, x):
        x = self.rnn(x)[0]
        if self.last_h: x = x[-1]
        x = F.relu(x)
        return x

cnn_lstm = nn.Sequential(*[
    Lambda(lambda x: x.transpose(2,1)),
    ConvBlock(3, 48, 5),
    ConvBlock(48, 64, 5),
    ConvBlock(64, 96, 5),
    Lambda(lambda x: x.permute(2,0,1)),
    RNNBlock(96),
    RNNBlock(128, last_h=True),
    Lambda(lambda x: x.squeeze()),
])

COUNTRY_EMB_SZ = 20

class MixedInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_lrg = nn.Sequential(create_body(models.resnet50(True), -2), nn.AdaptiveAvgPool2d(1))
        self.conv_small = create_body(pretrainedmodels.inceptionv4(), -2)
        self.cnn_lstm = cnn_lstm
        self.country_emb = nn.Embedding(codes_count, COUNTRY_EMB_SZ)
        head_inp_sz = num_features_model(self.conv_lrg) + \
            model_sizes(self.conv_small, (96, 96), False)[-1][1] + self.country_emb.embedding_dim + cnn_lstm[6].rnn.hidden_size
#         self.head = create_head(head_inp_sz, 340)[2:]
        self.head = create_head(head_inp_sz, 340, ps=0, lin_ftrs=[4000, 2000])[2:]
        
    def forward(self, drawings_lrg, drawings_small, seqs, country_codes):
        x = torch.cat(
            (
                self.conv_lrg(drawings_lrg).squeeze(),
                self.conv_small(drawings_small).squeeze(),
                self.cnn_lstm(seqs.float()),
                self.country_emb(country_codes)
            ),
            -1
        )
        return self.head(x)

learn = Learner(data_bunch, MixedInputModel(), metrics=[accuracy, map3])


learn.model = nn.DataParallel(learn.model)

learn.lr_find()


learn.recorder.plot()


#Replacing the classifier
state_dict = torch.load('models/resnet50-incv4-2e-05.pth')
keys = list(state_dict.keys())

for k in keys:
    if 'module.head' in k: del state_dict[k]
learn.model.load_state_dict(state_dict, False)

#FREEZING THE SUBMODELS
for m in [learn.model.module.cnn_lstm, learn.model.module.conv_lrg, learn.model.module.conv_small]:
    requires_grad(m, False)

learn.lr_find()

learn.recorder.plot()

learn.fit_one_cycle(4, 5e-3)

for l in flatten_model(learn.model):
    if type(l) == nn.BatchNorm1d or type(l) == nn.BatchNorm2d:
        l.track_running_stats = False



learn.fit(1, 5e-4)
#PREDICT
preds, _ = learn.get_preds(ds_type=DatasetType.Test)


from IPython.display import FileLink
FileLink('preds/test_preds')


create_submission(preds, data_bunch.test_dl, name, classes)

pd.read_csv(f'subs/{name}.csv.gz').head()
