import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from FaceMaskDataset import FaceMaskDataset
from torch.utils.data import DataLoader
from FaceMaskModel import MobileNetV3

# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

# Reading input folder
# files = os.listdir(args.input_folder)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
SHUFFLES = False

test_dataset = FaceMaskDataset(root_dir=args.input_folder, have_label=True, phase='eval')
test_dataloader = DataLoader(test_dataset, BATCH_SIZE, SHUFFLES)

#####
# TODO - your prediction code here
model = MobileNetV3(n_class=2, input_size=224, dropout=0.0, mode='large', width_mult=1.0)
model.load_state_dict(torch.load("mask_model.pt"))
model = model.to(DEVICE)
model.eval()

filenames = np.empty(0, dtype='|S12')
y_pred = np.empty(0)
for fnames, images, labels in test_dataloader:
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)
                    
    with torch.set_grad_enabled(False):
        outputs = model(images)

        _, pred = torch.max(outputs, 1)
        y_pred = np.append(y_pred, pred.to('cpu').numpy())
        filenames = np.append(filenames, fnames)

# Example (A VERY BAD ONE):
prediction_df = pd.DataFrame(zip(filenames, y_pred), columns=['id', 'label'])
####

# TODO - How to export prediction results
prediction_df.to_csv("prediction.csv", index=False, header=False)


# ### Example - Calculating F1 Score using sklrean.metrics.f1_score
# from sklearn.metrics import f1_score
# y_true = prediction_df['id'].apply(lambda x: int(x[7:8])).values
# f1 = f1_score(y_true, y_pred, average='binary')		# Averaging as 'binary' - This is how we will evaluate your results.

# print("F1 Score is: {:.2f}".format(f1))


