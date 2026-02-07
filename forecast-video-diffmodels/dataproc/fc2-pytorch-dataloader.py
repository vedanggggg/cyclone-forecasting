import glob
import torch
import numpy as np
import pickle
import os

from tqdm import tqdm

from utils import *

BASE_DIR = "/rds/general/user/zr523/home/researchProject/satellite/dataloader/64_FC"
c_dataloader_fns = glob.glob(BASE_DIR + "/*.dat")

BASE_DIR = os.environ.get("DATALOADER_DIR", BASE_DIR)
TEST_SET_PATH = os.environ.get("TEST_SET_PATH", os.path.join(os.path.dirname(__file__), "test_set.pkl"))

c_dataloader_fns = glob.glob(BASE_DIR + "/*.dat")
test_set = pickle.load(open(TEST_SET_PATH, "rb"))
train_dataloader = ModelDataLoader(batch_size=4, mode="fc", augment=False#True
                                   )
test_dataloader  = ModelDataLoader(batch_size=4, mode="fc", test=True)

for fn in tqdm(c_dataloader_fns):
    with open(fn, "rb") as file: 
        region, name = fn.split('/')[-1][:-4].split('_')
        print(name)
        print(file)
        if name in test_set[region]:
            test_dataloader.add_dataloader(pickle.load(file))
        else:
            train_dataloader.add_dataloader(pickle.load(file))

_ = len(train_dataloader) ; _ = len(test_dataloader)
