#!/usr/bin/env python3

# Max-Heinrich Laves
# Institute of Mechatronic Systems
# Leibniz Universität Hannover, Germany
# 2019

import torch
import numpy as np
from skimage import io
from sklearn import metrics

torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from models import BaselineResNet, BayesianResNet
from torch.utils.data import DataLoader
import argparse
import tqdm
from data_generator import KermanyDataset
from utils import accuracy


# enable cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Train classifier with uncertainty awareness on OCT dataset.')
parser.add_argument('--model', metavar='M', type=str, help='Specify the model M',
                    choices=['baseline', 'bayesian'])
parser.add_argument('--snapshot', metavar='S', type=str, help='Specify the model snapshot')
parser.add_argument('--bs', metavar='N', type=int, help='Specify the batch size N', default=1,
                    choices=list(range(1, 65)))
parser.add_argument('--mc', metavar='C', type=int, help='Number of Monte Carlo Samples', default=100,
                    choices=list(range(1, 1001)))
parser.add_argument('--p', metavar='C', type=float, help='Dropout probability', default=0.5,
                    choices=[0.1, 0.25, 0.5, 0.75])
args = parser.parse_args()

print("Test model:", args.model)
print("Test with batch_size:", str(args.bs))
print("dropout_p:", str(args.p))
print("num_mc:", str(args.mc))

# properties
batch_size = args.bs
val_batch_size = batch_size
num_classes = 4
bayesian_dropout_p = args.p
num_mc = args.mc

color = True
resize_to = (224, 224)

dataset_test = KermanyDataset("/home/laves/Downloads/OCT2017_3/train",
                              crop_to=(384, 384), resize_to=resize_to, color=color)
dataloader_test = DataLoader(dataset_test, batch_size=val_batch_size, shuffle=False)

assert len(dataset_test) > 0

print("Test dataset length:", len(dataset_test))
print('')

# create a model
model = torch.nn.Module()
if args.model == 'baseline':
    model = BaselineResNet(num_classes=num_classes).to(device)
elif args.model == 'bayesian':
    model = BayesianResNet(num_classes=num_classes).to(device)
else:
    assert False

# load weights for flow estimation from best last stage
checkpoint = torch.load(args.snapshot, map_location=device)
print("Loading previous weights at epoch " + str(checkpoint['epoch']) + " from " + args.snapshot)
model.load_state_dict(checkpoint['state_dict'])

# create loss function
criterion = torch.nn.CrossEntropyLoss()

print('')

# go through test set
print("Going through test set.")
model.eval()
y_true_np = []
y_pred_np = []

with torch.no_grad():

    test_losses = []
    test_accuracies = []
    correct_var = []
    incorrect_var = []

    batches = tqdm.tqdm(dataloader_test)
    for i, (x, y) in enumerate(batches):
        x, y = x.to(device), y.to(device)

        if args.model == 'baseline':
            y_pred = model(x)
            mean = y_pred

            y_true_np.append(y.data.cpu().numpy())
            y_pred_np.append(mean.argmax().data.cpu().numpy())
        elif args.model == 'bayesian':
            y_pred = model(x, dropout=True, p=bayesian_dropout_p)
            mc_output = y_pred.softmax(1).unsqueeze(1)

            for mc in range(num_mc - 1):
                y_pred = model(x, dropout=True, p=bayesian_dropout_p).softmax(1).unsqueeze(1)
                mc_output = torch.cat((mc_output, y_pred), dim=1)
                
            mean = mc_output.mean(dim=1)
            var = mc_output.var(dim=1)

            y_true_np.append(y.data.cpu().numpy())
            y_pred_np.append(mc_output.data.cpu().numpy())
        else:
            assert False

        if args.model != 'baseline':
            for j in range(batch_size):
                if mean[j].argmax() == y[j]:
                    correct_var.append(var[j].data.cpu().numpy())
                else:
                    incorrect_var.append(var[j].data.cpu().numpy())

        test_loss = criterion(mean, y)

        # sum epoch loss
        test_losses.append(test_loss.item())

        # calculate batch train accuracy
        batch_acc = accuracy(mean, y)
        test_accuracies.append(batch_acc)

        if len(correct_var) == 0 or len(incorrect_var) == 0:
            continue

        # print current loss
        batches.set_description("cv: {:.4f}, iv: {:.4f}".format(np.mean(np.array(correct_var)),
                                                                np.mean(np.array(incorrect_var))))

        # assert batch_size == 1
        # io.imsave("./imgout/"+str(i)+"_y="+str(y[0].data.cpu().item())+"_yp="+str(mean[0].argmax().item())+"_u="+str(np.mean(var[0].data.cpu().numpy()))+".png",
        #          x.data.cpu().numpy()[0, 0])

print("test mean loss:", np.mean(test_losses))
print("test mean accuracies:", np.mean(test_accuracies))

if args.model != 'baseline':
    print("mean correct uncert U_s  ", np.mean(np.array(correct_var)))
    print("")
    print("mean incorrect uncert U_s", np.mean(np.array(incorrect_var)))

y_pred_np = np.array(y_pred_np).squeeze()
y_true_np = np.array(y_true_np).squeeze()

np.save(args.model+"_mc_output_" + str(num_mc) + "_" + str(bayesian_dropout_p) + ".npy", y_pred_np)
np.save(args.model+"_true_output_" + str(num_mc) + "_" + str(bayesian_dropout_p) + ".npy", y_true_np)

print(y_pred_np.shape)
print(y_true_np.shape)
print(metrics.classification_report(y_true_np, y_pred_np.mean(axis=1).argmax(axis=1)))
