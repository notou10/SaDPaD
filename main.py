from function_interpretable import *
import argparse 
from tqdm import tqdm




parser = argparse.ArgumentParser()
parser.add_argument('--experiment_type', type=str)
parser.add_argument('--real_dir', type=str)
parser.add_argument('--img_dir', type=str)
parser.add_argument('--n_attr', type=int)
parser.add_argument('--n_point', type=int)
parser.add_argument('--load_from_precomputed', type=str2bool)
parser.add_argument('--attr_type', type=str)


args = parser.parse_args()

#text_list = args.text_list
experiment_type = args.experiment_type
img_dir_real = args.real_dir
img_dir_g = args.img_dir
n_attr = args.n_attr
attr_type = args.attr_type
n_point = args.n_point
load_from_precomputed=args.load_from_precomputed


experiment_name= f"{experiment_type}_{n_attr}_{attr_type}"
img_pickle_path = f"pickles/Dclipscore/{experiment_type}/{img_dir_real.split('/')[-1]}/img_mean.npy" 

precomputed_pickle_path = f"pickles/precomputed/{experiment_name}/{img_dir_real.split('/')[-1]}_{img_dir_g.split('/')[-1]}.pkl"

if load_from_precomputed:
    precomputed_dict = pickle.load(open(precomputed_pickle_path, 'rb'))
    CE_differences_1d = precomputed_dict['CE_differences_1d']  
    CE_differences_2d = precomputed_dict['CE_differences_2d']
    text_list = precomputed_dict['text_list']
    temp_plot_CE_difference_1d(CE_differences_1d, text_list,experiment_name, img_dir_real, img_dir_g, n_point, n_attr)
    temp_plot_CE_difference_2d(CE_differences_2d, text_list,experiment_name, img_dir_real, img_dir_g, n_point, n_attr)
    

else:
    text_list, stats = load_candidate(experiment_name=experiment_name, n_attr=n_attr, attr_type=attr_type) 
    text_mean = get_text_mean_rebuttal(text_list)

    img_mean = get_img_mean(img_dir=img_dir_real, img_pickle_path = img_pickle_path)


    DCS_GT_path = f"Dlipscore_npys/Dclipscore_attr_mean/{experiment_name}/{img_dir_real.split('/')[-1]}_{n_attr}.npy"
    DCS_input_path = f"Dlipscore_npys/Dclipscore_attr_mean/{experiment_name}/{img_dir_g.split('/')[-1]}_{n_attr}.npy"
    
    
    DCS_GT_stats, GT_filenames = get_img_stats_attr_mean(img_dir_real, img_mean, text_mean, text_list, DCS_GT_path)
    DCS_input_stats, input_filenames = get_img_stats_attr_mean(img_dir_g, img_mean, text_mean, text_list, DCS_input_path)


    CE_differences_1d = img_stats_into_density_1d(DCS_GT_stats,DCS_input_stats, text_list, n_point) 
    CE_differences_2d = img_stats_into_density_2d(DCS_GT_stats,DCS_input_stats, text_list, n_point) #shape = (nP2, n_point)



    temp_plot_CE_difference_1d(CE_differences_1d, text_list,experiment_name, img_dir_real, img_dir_g, n_point)
    temp_plot_CE_difference_2d(CE_differences_2d, text_list,experiment_name, img_dir_real, img_dir_g, n_point)
    
    
    os.makedirs(f"pickles/precomputed/{experiment_name}/{img_dir_real.split('/')[-1]}_{img_dir_g.split('/')[-1]}", exist_ok=True)
    precomputed_dict = {'CE_differences_1d':CE_differences_1d, 'CE_differences_2d':CE_differences_2d, 'text_list':text_list, \
        'experiment_name':experiment_name, 'img_dir_real':img_dir_real, 'img_dir_g':img_dir_g, 'n_point':n_point, 'n_attr':n_attr}
     
    with open(precomputed_pickle_path, 'wb') as f:
        pickle.dump(precomputed_dict, f)
        
    


