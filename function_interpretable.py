import torch
from CLIP.clip import *
from PIL import Image
import torchvision
from tqdm import tqdm 
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from torch.utils.data import DataLoader
import pickle
import matplotlib.pyplot as plt
import glob
import os
import cv2
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageFont, ImageDraw 
from torchvision import utils
from itertools import combinations_with_replacement, combinations, permutations
import itertools
import math
from matplotlib.ticker import MaxNLocator, MultipleLocator
import matplotlib.patches as mpatches 
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from attribute_list import *
from scipy.stats import gaussian_kde



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    
def temp_plot_CE_difference_1d(CE_result,text_list, experiment_name,img_dir, img_g_dir, n_point):
    plot_path = f"figs/{experiment_name}/{img_dir.split('/')[-1]}_{img_g_dir.split('/')[-1]}"
    
    
    if not os.path.exists(plot_path):
            os.makedirs(plot_path)

    font_size = 8

    pair_text_list_np = np.expand_dims(np.array(text_list),1)
    result = np.concatenate((CE_result,pair_text_list_np),axis=1) #KL, JSD, MI, text_attr
    result_sorted_KL = np.array(sorted(result, key = lambda item:np.float64(item[0]))[::-1])
    result_sorted_difference = np.array(sorted(result, key = lambda item:np.float64(item[1]))[::-1])

    stats_KL = result_sorted_KL[:,:2].astype('float64')
    stats_difference = result_sorted_difference[:,:2].astype('float64')

    KL, count_difference = stats_KL[:,0], stats_difference[:,1]

    
    def save_img_1d(outputs, types, result_sorted):
        plt.close()
        fig = plt.figure()
        ax2 = fig.add_subplot(111)
        variance = np.var(outputs, dtype=np.float32)
        ax2.set_xticks(np.arange(len(text_list)))
        ax2.set_xticklabels(result_sorted[:,2], rotation=90, fontsize =font_size)
       # ax2.set_ylim(min(outputs) - 0.1, max(outputs) + 0.1)
        ax2.bar(result_sorted[:, 2],outputs)
        ymin, ymax = ax2.get_ylim()
        #ax2.set_yticks(yticks)
        interval = ymax / 10
        exponent = int(math.log10(abs(ymax)))

        interval = 10**(exponent-1)


        ax2.yaxis.set_major_locator(MaxNLocator(nbins=10, integer=True, prune='both', min_n_ticks=10, symmetric=True, steps=[1, 2, 5, 10]))
        ax2.yaxis.set_minor_locator(MultipleLocator(interval))

        plt.title(f"{types}, task: set1 : {img_dir.split('/')[-1]}, set2 : {img_g_dir.split('/')[-1]},\n \
        total_difference ={np.round(np.sum(outputs)/(result_sorted[:,:2].shape[0]), 10)*10000000}, \n \
        var = {variance}", fontsize = 5) #
        #total_difference ={np.round(np.sum(outputs)/(stats.shape[0]), 10)*100000} ", fontsize = 5)
        plt.ylabel('each attribute-pair difference between 2 dataset', fontsize=5)
        
        plt.tight_layout()
        plt.show()    #plt.savefig(f"{plot_path}/{types}.png", dpi = 500)
        #import pdb; pdb.set_trace()
        plt.savefig(f"{plot_path}/n_point_{n_point}_{types}.png", dpi = 500)


    save_img_1d(KL, "SAD", result_sorted_KL)
    #save_img_1d(JSD, "JSD_1d")
    save_img_1d(count_difference, "mean_diff", result_sorted_difference)
    #save_img(PD, "PD_1d")

    return


def temp_plot_CE_difference_2d(CE_results,text_list, experiment_name,img_dir, img_g_dir, n_point):
    plot_path = f"figs/{experiment_name}/{img_dir.split('/')[-1]}_{img_g_dir.split('/')[-1]}"
    
    if not os.path.exists(plot_path):
            os.makedirs(plot_path)

    if len(text_list)>7: font_size=2
    else : font_size=5

    pair_text_list =[f"P({a[0]}, {a[1]})" for a in combinations(text_list,2)]
    pair_text_list_np = np.expand_dims(np.array(pair_text_list),1)


    CE_result = CE_results.cpu().numpy()
    result = np.concatenate((CE_result,pair_text_list_np),axis=1) #KL, JSD, MI, text_attr
    result_sorted = np.array(sorted(result, key = lambda item:np.float64(item[0]))[::-1])

    stats = result_sorted[:,:2].astype('float64')

    KL, JSD= stats[:,0], stats[:,1]

    def save_img(outputs, types):
        plt.close()
        fig = plt.figure()
        ax2 = fig.add_subplot(111)

        ax2.set_xticks(np.arange(len(pair_text_list)))
        ax2.set_xticklabels(result_sorted[:,2], rotation=90, fontsize =font_size)

        ax2.bar(result_sorted[:, 2],outputs)

        ymin, ymax = ax2.get_ylim()
        #ax2.set_yticks(yticks)
        interval = ymax / 10
        exponent = int(math.log10(abs(ymax)))

        interval = 10**(exponent-1)
        variance = np.var(outputs, dtype=np.float32)


        ax2.yaxis.set_major_locator(MaxNLocator(nbins=10, integer=True, prune='both', min_n_ticks=10, symmetric=True, steps=[1, 2, 5, 10]))
        ax2.yaxis.set_minor_locator(MultipleLocator(interval))



        plt.title(f"{types}, task: set1 : {img_dir.split('/')[-1]}, set2 : {img_g_dir.split('/')[-1]},\n \
        total_difference ={np.round(np.sum(outputs)/(stats.shape[0]), 10)*10000000} ,\n \
        var = {variance}", fontsize = 5)
        plt.ylabel('each attribute-pair difference between 2 dataset', fontsize=5)
        
        plt.tight_layout()
        plt.show()    #plt.savefig(f"{plot_path}/{types}.png", dpi = 500)
        plt.savefig(f"{plot_path}/n_point_{n_point}_{types}.png", dpi = 2000)

    save_img(KL, "PAD")


    return 


def load_candidate(experiment_name, n_attr, attr_type):
    text_list = []

    if attr_type!= "BLIP":
        return attr_dict[experiment_name][:n_attr], 0

    else :  #using BLIP
        candidate_pkl_path = f"pickles/attr_candidates/{experiment_name.split('_')[0]}.pkl"
         
        with open(candidate_pkl_path , 'rb') as dd:
            mydict = pickle.load(dd)
            for index,ele in enumerate(mydict): 
                text_list.append(ele[0])                


        return text_list[:n_attr], mydict[:n_attr]
    
def get_text_mean_rebuttal(text_list):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    with torch.no_grad():
        text = clip.tokenize(text_list).to(device)
        for index, single_txt in tqdm(enumerate(text), total =text.shape[0]):
            text_feature = model.encode_text(single_txt.unsqueeze(0))
            text_features = text_feature if index==0 else torch.vstack((text_features,text_feature))

        text_mean = torch.mean(text_features, axis=0)
                            

    return text_mean 


def get_img_mean(img_dir, img_pickle_path):
    if os.path.isfile(img_pickle_path):
        print(f"{img_pickle_path} exists, pickle loading...")   
        img_mean = np.load(img_pickle_path)
        return torch.Tensor(img_mean).to('cuda')

    else:
        print(f"{img_pickle_path} does not exists")
        pickle_name = img_pickle_path.split("/")[:-1]
        if not os.path.exists("/".join(pickle_name)):
            os.makedirs("/".join(pickle_name))


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    real_dataset = torchvision.datasets.ImageFolder(root=f"{img_dir}", transform=preprocess) #input shape = (224, 224)
    bs=2000
    real_dataloader = torch.utils.data.DataLoader(
                real_dataset, batch_size=bs, shuffle=False, drop_last=False, num_workers=2)

    img_list_box = torch.zeros((len(real_dataloader.dataset),512),dtype=torch.float16).to(device)
    with torch.no_grad():   
        for index, (datas,_) in tqdm(enumerate(real_dataloader), total = len(real_dataloader)):
            image_features = model.encode_image(datas.to(device))
            #img_list= image_features if index ==0 else torch.vstack((img_list,image_features))
            img_list_box[index*bs:(index+1)*bs]=image_features
    #img_mean = torch.mean(img_list, axis=0)
    img_mean = torch.mean(img_list_box, axis=0)

    img_mean_np = img_mean.detach().cpu().numpy()
    img_mean_np.astype('float16')
   # import pdb; pdb.set_trace()
    np.save(img_pickle_path, img_mean_np)

    return img_mean

def get_img_stats_attr_mean(img_dir, img_mean, text_mean, text_list, precomputed_np_path):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    real_dataset = torchvision.datasets.ImageFolder(root=f"{img_dir}", transform=preprocess) #input shape = (224, 224)
    bs=1000 #1000
    real_dataloader = torch.utils.data.DataLoader(
                real_dataset, batch_size=bs, shuffle=False, num_workers=6)
    allFiles, _ = map(list, zip(*real_dataloader.dataset.samples))
    
    if os.path.isfile(precomputed_np_path):
        print(f"{precomputed_np_path} exists, pickle loading...")   
        img_mean = np.load(precomputed_np_path)
        return torch.Tensor(img_mean).to('cuda'), allFiles

    else:
        print(f"{precomputed_np_path} does not exists")
        pickle_name = precomputed_np_path.split("/")[:-1]
        if not os.path.exists("/".join(pickle_name)):
            os.makedirs("/".join(pickle_name))
            
    img_mean = img_mean.to(torch.float16)
    img_mean = img_mean.unsqueeze(0).to(device)


    text_list_prompt = ["a photo of "+ a for a in text_list]
    text = clip.tokenize(text_list_prompt).to(device)
    attribute_num = len(text_list_prompt)
    
    #import pdb; pdb.set_trace()
    with torch.no_grad():
        text_features = model.encode_text(text)       
        
        similarity_per_attribute  = torch.zeros((len(real_dataloader.dataset),attribute_num),dtype=torch.float16).to(device)
        for index, (datas,_) in tqdm(enumerate(real_dataloader), total = len(real_dataloader)):

            image_features = model.encode_image(datas.to(device))  
            #import pdb; pdb.set_trace()                                
            img_feat = image_features - img_mean
            text_feat = text_features - text_mean

            img_feat /= img_feat.norm(dim=-1, keepdim=True)    
            for i in range(attribute_num):
    
                text_feat_temp =text_feat[i]
                text_feat_temp /= text_feat_temp.norm(dim=-1, keepdim=True)
                sim = (100.0 * img_feat @ text_feat_temp.T)

                similarity_per_attribute[index*bs:(index+1)*bs,i]=sim


    #import pdb; pdb.set_trace()
    similarity_per_attribute_np = similarity_per_attribute.detach().cpu().numpy()
    np.save(precomputed_np_path, similarity_per_attribute_np)

    #return similarity_per_attribute, allFiles
    return torch.Tensor(similarity_per_attribute_np).to(device), allFiles

def img_stats_into_density_1d(oriBLIPscore_img_stats,inputBLIPscore_img_stats, text_list, n_bins):
    total_num = len(text_list)
    given_box = np.zeros((total_num,2))  #KL(P,Q) JSD, MI(P,Q)
    #grids = grid_for_interpolation(n_bins)  #(n_box,4 point, each x,yin point)
    num_boxes = n_bins*n_bins
    eps = 1e-10
    sliced_array = np.linspace(-35, 35, num_boxes)

    for index, text in tqdm(enumerate(text_list), total = total_num):
        prob_ori = np.zeros(num_boxes)
        prob_input = np.zeros(num_boxes)
        attr_ori = oriBLIPscore_img_stats[:,index].detach().cpu().numpy()
        attr_input = inputBLIPscore_img_stats[:,index].detach().cpu().numpy()

        pdf_ori, pdf_input = kde_1d(attr_ori, attr_input)

        prob_ori_1d, prob_input_1d = into_grid_prob_1d(pdf_ori, pdf_input, n_bins)
        prob_ori_1d, prob_input_1d = prob_ori_1d+eps, prob_input_1d+eps
        # ori_mean = np.mean(prob_ori_1d * sliced_array) 
        # input_mean = np.mean(prob_input_1d * sliced_array)
        # difference_1d = input_mean -ori_mean
        difference_1d = np.mean(attr_input) - np.mean(attr_ori)

        #import pdb; pdb.set_trace()
        CE_P = -prob_ori_1d * np.log(prob_ori_1d)
        CE_Q = -prob_input_1d *np.log(prob_input_1d)
        CE_P_Q= -prob_ori_1d * np.log(prob_input_1d) 
        CE_Q_P= -prob_input_1d * np.log(prob_ori_1d) 

        KL_P_Q =  np.mean(CE_P_Q-CE_P) #H(P,Q) - H(P)
        if KL_P_Q <0: KL_P_Q=0
        KL_Q_P = np.mean(CE_Q_P-CE_Q) #H(P,Q) - H(Q)
        if KL_Q_P <0: KL_Q_P=0

        #import pdb; pdb.set_trace()
        JSD = (KL_P_Q+KL_Q_P)/2
        #given_box[index]= [KL_P_Q,JSD]  #KL(P,Q) JSD, MI(P,Q)
        given_box[index]= [KL_P_Q,difference_1d]  #KL(P,Q) JSD, 1d difference
    return given_box


def into_grid_prob_1d(pdf_ori_1d, pdf_input_1d,n_bins):
    num_boxes = n_bins*n_bins
    sliced_array = np.linspace(-35, 35, num_boxes)

    points_ori = pdf_ori_1d(sliced_array) * (70/num_boxes)
    points_input = pdf_input_1d(sliced_array) * (70/num_boxes)

    return points_ori, points_input

def kde_1d(attr1, attr2):

    kde_A = gaussian_kde(attr1)
    kde_B = gaussian_kde(attr2)

    return kde_A, kde_B


def img_stats_into_density_2d(oriBLIPscore_img_stats,inputBLIPscore_img_stats, text_list, n_bins):
    pooling = torch.nn.AvgPool2d(kernel_size=2, stride=1)
    #corr_total_num = len([a for a in combinations(np.arange(cov_map.shape[0]),2)])
    corr_total_num = len([a for a in combinations(np.arange(len(text_list)),2)])


    given_box = torch.zeros((corr_total_num,2)) #n_pair, n_filename, [score, coef^2*score, coef,coef^2] 
    #grids = grid_for_interpolation(n_bins)  #(n_box,4 point, each x,yin point)
    num_boxes = n_bins*n_bins
    eps = 1e-10

    for corr_index, pair in tqdm(enumerate(combinations(np.arange(len(text_list)),2)), total = corr_total_num):
        i,j = pair[0], pair[1]

        attr1=oriBLIPscore_img_stats[:,i].detach().cpu().numpy()
        attr2=oriBLIPscore_img_stats[:,j].detach().cpu().numpy()

        attr1_input=inputBLIPscore_img_stats[:,i].detach().cpu().numpy()
        attr2_input=inputBLIPscore_img_stats[:,j].detach().cpu().numpy()


        pdf_A, pdf_B,pdf_A_and_B = kde(attr1, attr2)
        pdf_A_input, pdf_B_input,pdf_A_and_B_input = kde(attr1_input, attr2_input)

        #import pdb; pdb.set_trace()

        #############2d
        prob_ori2d, prob_input2d = into_grid_prob_2d(pdf_A_and_B, pdf_A_and_B_input,n_bins, pooling)
        prob_ori2d, prob_input2d = prob_ori2d+eps, prob_input2d+eps

        CE_P = -prob_ori2d * torch.log(prob_ori2d)
        CE_Q = -prob_input2d *torch.log(prob_input2d)
        CE_P_Q= -prob_ori2d * torch.log(prob_input2d) 
        CE_Q_P= -prob_input2d * torch.log(prob_ori2d)
        

  
        KL_P_Q =  torch.mean(CE_P_Q-CE_P) #H(P,Q) - H(P)
        if KL_P_Q <0: KL_P_Q=0
        KL_Q_P = torch.mean(CE_Q_P-CE_Q) #H(P,Q) - H(Q)
        if KL_Q_P <0: KL_Q_P=0
        #KL_Q_P =  np.mean(prob_input*np.log(prob_input) - prob_input*np.log(prob_ori))
        JSD = (KL_P_Q+KL_Q_P)/2
        given_box[corr_index]= torch.Tensor([KL_P_Q,JSD])  #KL(P,Q) JSD, MI(P,Q)
        
    return given_box

def kde(attr1, attr2):

    kde_A = gaussian_kde(attr1)
    kde_B = gaussian_kde(attr2)
    temp = np.vstack((attr1,attr2))
    kde_A_and_B = gaussian_kde(temp)

    return kde_A, kde_B, kde_A_and_B

def into_grid_prob_2d(pdf_ori, pdf_input,n_bins, pooling):
    num_boxes = n_bins*n_bins
    x = np.linspace(-35, 35, n_bins+1)
    y = np.linspace(-35, 35,n_bins+1)
    X,Y = np.meshgrid(x,y)

    sliced_array = np.array([X.flatten(), Y.flatten()]) #2,10000 x부터..
    points_ori = pdf_ori(sliced_array)
    reshaped_point_ori = points_ori.reshape(n_bins+1,n_bins+1) #input = 101,101,2 ,x axis to y axis.. #output = 100,100

    points_input = pdf_input(sliced_array)
    reshaped_point_input = points_input.reshape(n_bins+1,n_bins+1)  #input = 101,101,2 ,x axis to y axis.. #output = 100,100
    
    
    reshaped_point_ori_cuda = torch.Tensor(reshaped_point_ori).to('cuda')
    reshaped_point_input_cuda = torch.Tensor(reshaped_point_input).to('cuda')
    #pooling = torch.nn.AvgPool2d(kernel_size=2, stride=1)

    prob_ori = pooling(reshaped_point_ori_cuda.unsqueeze(0)) * ((70/n_bins)*(70/n_bins))
    prob_input = pooling(reshaped_point_input_cuda.unsqueeze(0)) * ((70/n_bins)*(70/n_bins))
    
    return prob_ori.squeeze(), prob_input.squeeze()
