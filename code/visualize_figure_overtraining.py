import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import utils
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
import numpy as np
from copy import deepcopy

matplotlib.rc('font', family='DejaVu Sans', size=20)

import models.sketchanetfbin as sketchanetfbin
import models.sketchanetwbin as sketchanetwbin

alphas_list = []
betas_list = []
k_list = []
k_list_xnor = []
half_val = -1
old_alphas_list = []

cnt = 0
for i in range(0, 410):

    resume = "savedmodels/sketchanetwbin_tuberlin_epoch_" + str(i) + ".pth.tar"
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        best_prec1 = checkpoint['best_prec1']
        model = sketchanetwbin.Net(250)
        model = model.cuda()
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))

        print(model)
        cnt += 1

        for m in model.modules():
            if isinstance(m, nn.Conv2d) and m.weight.data[0].numel() > 12000:
                total_elems = m.weight.data[0].numel()
                half_val = m.weight.data[0].numel() / 2
                k, k1, k2, alpha, beta, weights = utils.DualBinarizeConvParams(deepcopy(m), False)
                k_xnor, origalpha = utils.binarizeConvParams(deepcopy(m), False)
                print(alpha.cpu().numpy()[0])
                print(beta.cpu().numpy()[0])
                print(k.cpu().numpy()[0])
                print(weights.cpu().numpy())

                k_list_xnor.append(k_xnor/total_elems)

                alphas_list.append(alpha.cpu().numpy()[0])
                betas_list.append(beta.cpu().numpy()[0])
                k_list.append(k.cpu().numpy()[0]/total_elems)


                x = range(len(weights.cpu().numpy()))
                y = weights.cpu().numpy()
                y = np.round(y, 4)
                fig = plt.figure(facecolor="white", figsize=(12, 6), dpi=80)
                plt.subplots_adjust(bottom=.15)

                bins = 200
                if cnt == 1:
                    bins = 100

                n, bin_edges = np.histogram(y, bins)
                # Normalize it, so that every bins value gives the probability of that bin
                bin_probability = n/float(n.sum())
                # Get the mid points of every bin
                bin_middles = (bin_edges[1:]+bin_edges[:-1])/2.
                # Compute the bin-width
                bin_width = bin_edges[1]-bin_edges[0]
                # Plot the histogram as a bar plot
                #plt.bar(bin_middles, bin_probability, width=bin_width)
                plt.plot(bin_middles, bin_probability, ls='steps', color='darkslategrey')

                plt.xlim([-0.1, 0.1])
                ymax = 0.005
                width = 2
                plt.grid()
                plt.title('Distribution of weights')
                plt.locator_params(nbins=5, axis='y')
                plt.plot([alpha.cpu().numpy()[0], alpha.cpu().numpy()[0]], [0, ymax], '-', color='darkred', linewidth=width, label='α (Ours)')
                plt.plot([beta.cpu().numpy()[0], beta.cpu().numpy()[0]], [0, ymax], '-',  color='darkgreen', linewidth=width, label='β (Ours)')
                #origalpha = utils.binarizeConvParams(m, False)
                old_alphas_list.append(origalpha.cpu().numpy()[0])
                plt.plot([origalpha.cpu().numpy()[0], origalpha.cpu().numpy()[0]], [0, ymax], '-',  color='royalblue', linewidth=width, label="α,-α (XNOR)")
                plt.plot([-origalpha.cpu().numpy()[0], -origalpha.cpu().numpy()[0]], [0, ymax], '-',  color='royalblue', linewidth=width)
                plt.xlabel('Value')
                plt.ylabel('Probability')
                plt.legend()
                plt.show()
                print(origalpha.cpu().numpy()[0])
                print(k.cpu().numpy()[0])

                break


plt.figure(facecolor='white')
plt.plot([i*10 for i in range(len(k_list_xnor))], k_list_xnor, 'g-', label='XNOR')
plt.plot([i*10 for i in range(len(k_list))], k_list, 'r-', label='Ours')
plt.ylim([0, 1])
plt.xlabel('Epoch')
plt.ylabel('Normalized K-Value')
plt.title('Variation of K-Value during training')
plt.grid()
#plt.plot(range(len(alphas_list)), alphas_list, 'r-', label='Alpha')
#plt.plot(range(len(betas_list)), betas_list, 'g-', label='Beta')
#plt.plot(range(len(old_alphas_list)), old_alphas_list, 'b-', label='Old Alpha')
#plt.plot(range(len(old_alphas_list)), [-x for x in old_alphas_list], 'b-')
plt.legend(loc=1)
plt.show()
