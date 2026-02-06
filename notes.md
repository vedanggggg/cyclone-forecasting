i am going to modify the test train script only for the 15 cyclones i have preprocessed until now. we will have to change this later accoridng to the original repo.

i enocountered a inconsistency issue while trying to preprocess as well. The author’s ModelDataLoader expects img_64 to be (N, H, W) (no channel dim).Some of my files were saved as (N, 1, H, W). i have changed them now to match with the authors.
