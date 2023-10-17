import argparse
import os
import pickle
import random
import re
from collections import OrderedDict
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import roc_auc_score
from torch.utils import model_zoo
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from tqdm import tqdm

import datasets.dataset as dataset

# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser('Fourier_Heat_Map')
    parser.add_argument('--data_path', type=str, default='/data/spot-diff/VisA_pytorch/1cls')
    parser.add_argument('--save_path', type=str, default='./result')
    parser.add_argument('--model', type=str, choices=['padim','stylized_padim'], default='stylized_padim')
    parser.add_argument('--dataset_name', type=str, choices=['mvtecad','visa'], default='visa')
    parser.add_argument('--resize', type=int,  default=256)
    parser.add_argument('--crop', type=int,  default=224)
    parser.add_argument('--eps', type=int,  default=8)
    parser.add_argument('--size', type=int,  default=8)

    return parser.parse_args()


def main():
    args = parse_args()
    #load model
    if args.model == 'padim':
        model = resnet50(pretrained=True, progress=True)
        t_d = 1792
        d = 550
    elif args.model == 'stylized_padim':
        model = resnet50(pretrained=False, progress=True)
        PATH = model_zoo.load_url('https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar') # trained by Stylized Imagenet weight
        new_state_dict = fix_key(PATH['state_dict'])
        model.load_state_dict(new_state_dict)
        t_d = 1792
        d = 550

    model.to(device)
    model.eval()

    random.seed(1024)
    torch.manual_seed(1024)

    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    idx = torch.tensor(sample(range(0, t_d), d))

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)


    os.makedirs(os.path.join(args.save_path, args.model, 'temp_train_pkl'), exist_ok=True)

    if args.dataset_name == 'mvtecad':
        CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid','hazelnut', 'leather', 'metal_nut',
                       'pill', 'screw','tile', 'toothbrush', 'transistor', 'wood', 'zipper']
        CLASS_NAMES = ['pill']

    elif args.dataset_name == 'visa':
        CLASS_NAMES = ['pcb1', 'pcb2', 'pcb3','pcb4','macaroni1', 'macaroni2','candle', 'capsules', 'cashew','chewinggum', 'fryum','pipe_fryum']
        CLASS_NAMES = ['chewinggum']


    for class_name in CLASS_NAMES:
        train_dataset = dataset.Dataset_process(args.data_path, class_name=class_name, is_train=True,resize=args.resize, cropsize=args.crop,
                                                dataset_name = args.dataset_name, class_names = CLASS_NAMES)
        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)

        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        # extract train set features
        train_feature_filepath = os.path.join(args.save_path, args.model, 'temp_train_pkl', 'train_%s.pkl' % class_name)
        if not os.path.exists(train_feature_filepath):
            for (x, _, _) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
                # model prediction
                with torch.no_grad():
                    _ = model(x.to(device))
                # get intermediate layer outputs
                for k, v in zip(train_outputs.keys(), outputs):
                    train_outputs[k].append(v.cpu().detach())

                # initialize hook outputs
                outputs = []

            for k, v in train_outputs.items():
                train_outputs[k] = torch.cat(v, 0)



            # Embedding concat
            embedding_vectors = train_outputs['layer1']
            for layer_name in ['layer2', 'layer3']:
                embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])

            # randomly select d dimension
            embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
            # calculate multivariate Gaussian distribution
            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W)
            mean = torch.mean(embedding_vectors, dim=0).numpy()
            cov = torch.zeros(C, C, H * W).numpy()
            I = np.identity(C)
            for i in range(H * W):
                # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
                cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
            # save learned distribution
            train_outputs = [mean, cov]
            with open(train_feature_filepath, 'wb') as f:
                pickle.dump(train_outputs, f)
        else:
            print('load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                train_outputs = pickle.load(f)

        #fabrication of fourier heat map
        size_h = np.int8(args.crop/ args.size)
        errors = np.zeros((size_h,size_h))


        for xx in range(size_h):
            for yy in range(size_h):
                test_dataset = dataset.Dataset_process(args.data_path, class_name=class_name, is_train=False,resize=args.resize, cropsize=args.crop,eps = args.eps, size=args.size
                                                       ,xx=xx,yy=yy,dataset_name = args.dataset_name, class_names = CLASS_NAMES)
                test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)
                test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
                gt_list = []
                for  (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
                    gt_list.extend(y.cpu().detach().numpy())
                    # model prediction
                    with torch.no_grad():
                        _ = model(x.to(device))
                    # get intermediate layer outputs
                    for k, v in zip(test_outputs.keys(), outputs):
                        test_outputs[k].append(v.cpu().detach())

                    # initialize hook outputs
                    outputs = []

                for k, v in test_outputs.items():
                    test_outputs[k] = torch.cat(v, 0)

                # Embedding concat
                embedding_vectors = test_outputs['layer1']
                for layer_name in ['layer2', 'layer3']:
                    embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

                # randomly select d dimension
                embedding_vectors = torch.index_select(embedding_vectors, 1, idx)

                # calculate distance matrix
                B, C, H, W = embedding_vectors.size()
                embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
                dist_list = []
                for i in range(H * W):
                    mean = train_outputs[0][:, i]
                    conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
                    dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
                    dist_list.append(dist)

                dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

                # upsample
                dist_list = torch.tensor(dist_list)
                score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                                        align_corners=False).squeeze().numpy()

                # apply gaussian smoothing on the score map
                for i in range(score_map.shape[0]):
                    score_map[i] = gaussian_filter(score_map[i], sigma=4)

                # Normalization
                max_score = score_map.max()
                min_score = score_map.min()
                scores = (score_map - min_score) / (max_score - min_score)

                # calculate image-level ROC AUC score
                img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
                gt_list = np.asarray(gt_list)
                img_roc_auc = roc_auc_score(gt_list, img_scores)
                print('image ROCAUC: %.3f' % (img_roc_auc))

                errors[xx,yy] = 1 - img_roc_auc

        save_dir = os.path.join(args.save_path, args.model, 'fourier_heat_map',class_name)
        os.makedirs(save_dir, exist_ok=True)
        plot_fig(errors,save_dir,class_name,label_name='_error')
        np.savetxt(os.path.join(save_dir, class_name+'_error.csv'),errors,delimiter=',')



def plot_fig(scores, save_dir, class_name,label_name='bottle'):

    fig, ax=plt.subplots()
    ax =sns.heatmap(
        scores,
        vmin=0,
        vmax=0.5,
        cmap="jet",
        cbar=True,
        xticklabels=False,
        yticklabels=False,
    )
    sns.set(font_scale = 2)
    ax.collections[0].colorbar.set_label('Error')
    plt.savefig(os.path.join(save_dir, class_name +label_name), dpi=100)
    plt.clf()
    plt.close()

def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z

def fix_key(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    return new_state_dict


if __name__ == '__main__':
    main()
