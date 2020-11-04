import pdb
from fastai.vision.all import *
from fastai.data.all import *
import pandas as pd
from pathlib import Path

data_dir = Path('data').absolute()
train_val_file = data_dir/'subset_train_val_data.csv'

train_val_folder = get_image_files(data_dir/"train")
train_val_data = pd.read_csv(train_val_file)


data_block = DataBlock(blocks=(ImageBlock, CategoryBlock),
                       splitter=ColSplitter(),
                       get_x=ColReader(0, pref=data_dir),
                       get_y=ColReader(1),
                       item_tfms=Resize(224)
                       )
dls = data_block.dataloaders(train_val_data, bs=4)

mixup = MixUp()
learn = cnn_learner(dls, resnet18, metrics=error_rate,
                    cbs=[mixup])

learn.fit_one_cycle(1, 3e-3)
