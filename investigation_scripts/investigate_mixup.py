import pdb
from fastai.vision.all import *
from fastai.data.all import *
import pandas as pd
from pathlib import Path

data_dir = Path('data').absolute()
train_val_file = data_dir/'subset_train_val_data.csv'

train_val_folder = get_image_files(data_dir/"train")
train_val_data = pd.read_csv(train_val_file)

# path = untar_data(URLs.MNIST_TINY)
# test_path = path / 'test'

# mnist = DataBlock(blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
#                   get_items=get_image_files,
#                   splitter=GrandparentSplitter(),
#                   get_y=parent_label)


data_block = DataBlock(blocks=(ImageBlock, CategoryBlock),
                       splitter=ColSplitter(),
                       get_x=ColReader(0, pref='data/'),
                       get_y=ColReader(1),
                       item_tfms=Resize(224),
                       )

dls = data_block.dataloaders(train_val_data, bs=16)
# dls = mnist.dataloaders(path, bs=16)


mixup = MixUp()
learn_mixup = cnn_learner(dls, resnet18, metrics=error_rate,
                          cbs=[mixup])

learn_mixup.fit_one_cycle(1, 3e-3)

pdb.set_trace()
print(learn_mixup.predict(get_image_files(data_dir/'test')[0]))

# learn = cnn_learner(dls, resnet18, metrics=error_rate)

# learn.fit_one_cycle(1, 3e-3)

# dl = dls.test_dl(get_image_files(data_dir/'test')[0:1], with_decoded=True)
# learn.get_preds(dl=dl)
# print(learn.predict(get_image_files(data_dir/'test')[0]))
pdb.set_trace()
print('wait')

# need to go to /usr/local/lib/python3.6/dist-packages/fastai/learner.py
#   to set breakpoints

# in function one_batch(), the data has not been mixed up yet
# in function do_one_batch(), the image data has been mixed up but the label data has not?
