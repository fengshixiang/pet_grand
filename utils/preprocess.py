import shutil

import os
import pickle
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
from PIL import Image
from scipy import ndimage
from batchgenerators.utilities.file_and_folder_operations import load_pickle

from utils import MorphologicalOperation, GetLargestConnectedCompont, GetLargestConnectedCompontBoundingbox

# simense
dose1_2 = '1-2 dose'
dose1_4 = '1-4 dose'
dose1_10 = '1-10 dose'
dose1_20 = '1-20 dose'
dose1_50 = '1-50 dose'
dose1_100 = '1-100 dose'
dose_full = 'Full_dose'

# uexplorer
dose1_22 = 'D2.nii'
dose1_42 = 'D4.nii'
dose1_102 = 'D10.nii'
dose1_202 = 'D20.nii'
dose1_502 = 'D50.nii'
dose1_1002 = 'D100.nii'
dose1_23 = 'DRF_2.nii'
dose1_43 = 'DRF_4.nii'
dose1_103 = 'DRF_10.nii'
dose1_203 = 'DRF_20.nii'
dose1_503 = 'DRF_50.nii'
dose1_1003 = 'DRF_100.nii'
dose_full2 = 'normal.nii'
dose_full3 = 'NORMAL.nii'


def subdirs(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res

def subfiles(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res

def write_pickle(obj, file: str, mode: str = 'wb') -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)

def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a

def normalize_paired_img(input_img, target_img):
    perc995 = np.percentile(input_img, 99.5)
    input_img = np.clip(input_img, 0, perc995) / perc995
    target_img = [np.clip(img, 0, perc995) / perc995 for img in target_img]

    return input_img, target_img


def normalize_img_meanstd(img):
    mean = img.mean()
    std = img.std()
    return ((img - mean) / std).astype(np.float32)


def arr2image(arr, arr_max):
    arr = np.clip(arr, 0, arr_max) / arr_max
    arr = (arr * 255).astype(np.uint8)
    image = Image.fromarray(arr)

    return image


def arr2image_meanstd(arr):
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = (arr * 255).astype(np.uint8)
    image = Image.fromarray(arr)

    return image


def split_nii():
    data_path = '/data/data/PET/nii/uexplorer/'
    patient_n_list = subdirs(data_path, join=True)
    np.random.shuffle(patient_n_list)
    train_patient_list = patient_n_list[:int(len(patient_n_list) * 4 / 5)]
    test_patient_list = patient_n_list[int(len(patient_n_list) * 4 / 5):]
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    for patient_p in train_patient_list:
        shutil.move(patient_p, os.path.join(train_path, patient_p.split('/')[-1]))
    for patient_p in test_patient_list:
        shutil.move(patient_p, os.path.join(test_path, patient_p.split('/')[-1]))


def preprocess_itk_simense():
    data_path = '/home/mnt/data/PET/nii/siemens'
    save_path = '/home/mnt/data/PET/preprocessed/siemens'
    for dose in ['dose_2', 'dose_4', 'dose_10', 'dose_full']:
        os.makedirs(os.path.join(save_path, '{}/train_npy'.format(dose)), exist_ok=True)
        os.makedirs(os.path.join(save_path, '{}/train_png'.format(dose)), exist_ok=True)
        os.makedirs(os.path.join(save_path, '{}/test_npy'.format(dose)), exist_ok=True)
        os.makedirs(os.path.join(save_path, '{}/test_png'.format(dose)), exist_ok=True)

    attr_dict = dict()

    for mode in ['train', 'test']:
        for patient_p in tqdm(subdirs(os.path.join(data_path, mode))):
            patient_dose_full_save_path = os.path.join(save_path, 'dose_full/{}_npy/{}'.format(mode, patient_p.split('/')[-1]))
            patient_dose_2_save_path = os.path.join(save_path, 'dose_2/{}_npy/{}'.format(mode, patient_p.split('/')[-1]))
            patient_dose_4_save_path = os.path.join(save_path, 'dose_4/{}_npy/{}'.format(mode, patient_p.split('/')[-1]))
            patient_dose_10_save_path = os.path.join(save_path, 'dose_10/{}_npy/{}'.format(mode, patient_p.split('/')[-1]))
            os.makedirs(patient_dose_full_save_path, exist_ok=True)
            os.makedirs(patient_dose_2_save_path, exist_ok=True)
            os.makedirs(patient_dose_4_save_path, exist_ok=True)
            os.makedirs(patient_dose_10_save_path, exist_ok=True)
            os.makedirs(patient_dose_full_save_path.replace('_npy', '_png'), exist_ok=True)
            os.makedirs(patient_dose_2_save_path.replace('_npy', '_png'), exist_ok=True)
            os.makedirs(patient_dose_4_save_path.replace('_npy', '_png'), exist_ok=True)
            os.makedirs(patient_dose_10_save_path.replace('_npy', '_png'), exist_ok=True)

            series_n_list = subfiles(patient_p, join=False)
            for i in range(len(series_n_list)):
                if dose1_2 in series_n_list[i]:
                    dose_2_index = i
                    break
            for i in range(len(series_n_list)):
                if dose1_4 in series_n_list[i]:
                    dose_4_index = i
                    break
            for i in range(len(series_n_list)):
                if dose1_10 in series_n_list[i]:
                    dose_10_index = i
                    break
            for i in range(len(series_n_list)):
                if dose_full in series_n_list[i]:
                    dose_full_index = i
                    break
            dose_full_p = os.path.join(patient_p, series_n_list[dose_full_index])
            dose_2_p = os.path.join(patient_p, series_n_list[dose_2_index])
            dose_4_p = os.path.join(patient_p, series_n_list[dose_4_index])
            dose_10_p = os.path.join(patient_p, series_n_list[dose_10_index])

            dose_full_itk, [dose_2_itk, dose_4_itk, dose_10_itk] = get_roi(dose_full_p, [dose_2_p, dose_4_p, dose_10_p])

            dose_full_npy = sitk.GetArrayFromImage(dose_full_itk).astype(np.float32)
            dose_2_npy = sitk.GetArrayFromImage(dose_2_itk).astype(np.float32)
            dose_4_npy = sitk.GetArrayFromImage(dose_4_itk).astype(np.float32)
            dose_10_npy = sitk.GetArrayFromImage(dose_10_itk).astype(np.float32)

            dose_full_dict = {'mean': dose_full_npy.mean(),
                              'var': dose_full_npy.var(),
                              'std': dose_full_npy.std(),
                              'min': dose_full_npy.min(),
                              'max': dose_full_npy.max(),
                              'perc995': np.percentile(dose_full_npy, 99.5),
                              'size': dose_full_itk.GetSize(),
                              'spacing': dose_full_itk.GetSpacing()}
            dose_2_dict = {'mean': dose_2_npy.mean(),
                           'var': dose_2_npy.var(),
                           'std': dose_2_npy.std(),
                           'min': dose_2_npy.min(),
                           'max': dose_2_npy.max(),
                           'perc995': np.percentile(dose_2_npy, 99.5),
                           'size': dose_2_itk.GetSize(),
                           'spacing': dose_2_itk.GetSpacing()}
            dose_4_dict = {'mean': dose_4_npy.mean(),
                           'var': dose_4_npy.var(),
                           'std': dose_4_npy.std(),
                           'min': dose_4_npy.min(),
                           'max': dose_4_npy.max(),
                           'perc995': np.percentile(dose_4_npy, 99.5),
                           'size': dose_4_itk.GetSize(),
                           'spacing': dose_4_itk.GetSpacing()}
            dose_10_dict = {'mean': dose_10_npy.mean(),
                            'var': dose_10_npy.var(),
                            'std': dose_10_npy.std(),
                            'min': dose_10_npy.min(),
                            'max': dose_10_npy.max(),
                            'perc995': np.percentile(dose_10_npy, 99.5),
                            'size': dose_10_itk.GetSize(),
                            'spacing': dose_10_itk.GetSpacing()}
            attr_dict[patient_p.split('/')[-1]] = {
                'dose_full': dose_full_dict,
                'dose_2': dose_2_dict,
                'dose_4': dose_4_dict,
                'dose_10': dose_10_dict,
            }

            for i in range(dose_full_npy.shape[0]):
                dose_full_slice = dose_full_npy[i]
                dose_2_slice = dose_2_npy[i]
                dose_4_slice = dose_4_npy[i]
                dose_10_slice = dose_10_npy[i]
                dose_full_save_p = os.path.join(patient_dose_full_save_path, '{}__{}.npy'.format(patient_p.split('/')[-1], i))
                dose_2_save_p = os.path.join(patient_dose_2_save_path, '{}__{}.npy'.format(patient_p.split('/')[-1], i))
                dose_4_save_p = os.path.join(patient_dose_4_save_path, '{}__{}.npy'.format(patient_p.split('/')[-1], i))
                dose_10_save_p = os.path.join(patient_dose_10_save_path, '{}__{}.npy'.format(patient_p.split('/')[-1], i))
                np.save(dose_full_save_p, dose_full_slice)
                np.save(dose_2_save_p, dose_2_slice)
                np.save(dose_4_save_p, dose_4_slice)
                np.save(dose_10_save_p, dose_10_slice)
                arr_max = np.percentile(dose_full_slice, 99.5)
                dose_full_slice_image = arr2image(dose_full_slice, arr_max)
                dose_2_slice_image = arr2image(dose_2_slice, arr_max)
                dose_4_slice_image = arr2image(dose_4_slice, arr_max)
                dose_10_slice_image = arr2image(dose_10_slice, arr_max)
                dose_full_slice_image.save(dose_full_save_p.replace('_npy', '_png').replace('.npy', '.png'))
                dose_2_slice_image.save(dose_2_save_p.replace('_npy', '_png').replace('.npy', '.png'))
                dose_4_slice_image.save(dose_4_save_p.replace('_npy', '_png').replace('.npy', '.png'))
                dose_10_slice_image.save(dose_10_save_p.replace('_npy', '_png').replace('.npy', '.png'))

    write_pickle(attr_dict, os.path.join(save_path, 'attr.pkl'))


def preprocess_itk_uexplorer():
    data_path = '/home/mnt/data/PET/nii/uexplorer'
    save_path = '/home/mnt/data/PET/preprocessed/uexplorer'
    for dose in ['dose_2', 'dose_4', 'dose_10', 'dose_full']:
        os.makedirs(os.path.join(save_path, '{}/train_npy'.format(dose)), exist_ok=True)
        os.makedirs(os.path.join(save_path, '{}/train_png'.format(dose)), exist_ok=True)
        os.makedirs(os.path.join(save_path, '{}/test_npy'.format(dose)), exist_ok=True)
        os.makedirs(os.path.join(save_path, '{}/test_png'.format(dose)), exist_ok=True)

    attr_dict = dict()

    for mode in ['train', 'test']:
        for patient_p in tqdm(subdirs(os.path.join(data_path, mode))):
            patient_dose_full_save_path = os.path.join(save_path, 'dose_full/{}_npy/{}'.format(mode, patient_p.split('/')[-1]))
            patient_dose_2_save_path = os.path.join(save_path, 'dose_2/{}_npy/{}'.format(mode, patient_p.split('/')[-1]))
            patient_dose_4_save_path = os.path.join(save_path, 'dose_4/{}_npy/{}'.format(mode, patient_p.split('/')[-1]))
            patient_dose_10_save_path = os.path.join(save_path, 'dose_10/{}_npy/{}'.format(mode, patient_p.split('/')[-1]))
            os.makedirs(patient_dose_full_save_path, exist_ok=True)
            os.makedirs(patient_dose_2_save_path, exist_ok=True)
            os.makedirs(patient_dose_4_save_path, exist_ok=True)
            os.makedirs(patient_dose_10_save_path, exist_ok=True)
            os.makedirs(patient_dose_full_save_path.replace('_npy', '_png'), exist_ok=True)
            os.makedirs(patient_dose_2_save_path.replace('_npy', '_png'), exist_ok=True)
            os.makedirs(patient_dose_4_save_path.replace('_npy', '_png'), exist_ok=True)
            os.makedirs(patient_dose_10_save_path.replace('_npy', '_png'), exist_ok=True)

            series_n_list = subfiles(patient_p, join=False)
            dose_2_index = -1
            for i in range(len(series_n_list)):
                if dose1_2 in series_n_list[i] or dose1_22 in series_n_list[i] or dose1_23 in series_n_list[i]:
                    dose_2_index = i
                    break
            for i in range(len(series_n_list)):
                if dose1_4 in series_n_list[i] or dose1_42 in series_n_list[i] or dose1_43 in series_n_list[i]:
                    dose_4_index = i
                    break
            for i in range(len(series_n_list)):
                if dose1_10 in series_n_list[i] or dose1_102 in series_n_list[i] or dose1_103 in series_n_list[i]:
                    dose_10_index = i
                    break
            for i in range(len(series_n_list)):
                if dose_full in series_n_list[i] or dose_full2 in series_n_list[i] or dose_full3 in series_n_list[i]:
                    dose_full_index = i
                    break
            dose_full_p = os.path.join(patient_p, series_n_list[dose_full_index])
            if dose_2_index >= 0:
                dose_2_p = os.path.join(patient_p, series_n_list[dose_2_index])
            dose_4_p = os.path.join(patient_p, series_n_list[dose_4_index])
            dose_10_p = os.path.join(patient_p, series_n_list[dose_10_index])

            if dose_2_index >= 0:
                dose_full_itk, [dose_2_itk, dose_4_itk, dose_10_itk] = get_roi(dose_full_p, [dose_2_p, dose_4_p, dose_10_p])
            else:
                dose_full_itk, [dose_4_itk, dose_10_itk] = get_roi(dose_full_p, [dose_4_p, dose_10_p])

            dose_full_npy = sitk.GetArrayFromImage(dose_full_itk).astype(np.float32)
            if dose_2_index >= 0:
                dose_2_npy = sitk.GetArrayFromImage(dose_2_itk).astype(np.float32)
            dose_4_npy = sitk.GetArrayFromImage(dose_4_itk).astype(np.float32)
            dose_10_npy = sitk.GetArrayFromImage(dose_10_itk).astype(np.float32)

            dose_full_dict = {'mean': dose_full_npy.mean(),
                              'var': dose_full_npy.var(),
                              'std': dose_full_npy.std(),
                              'min': dose_full_npy.min(),
                              'max': dose_full_npy.max(),
                              'perc995': np.percentile(dose_full_npy, 99.5),
                              'size': dose_full_itk.GetSize(),
                              'spacing': dose_full_itk.GetSpacing()}
            if dose_2_index >= 0:
                dose_2_dict = {'mean': dose_2_npy.mean(),
                               'var': dose_2_npy.var(),
                               'std': dose_2_npy.std(),
                               'min': dose_2_npy.min(),
                               'max': dose_2_npy.max(),
                               'perc995': np.percentile(dose_2_npy, 99.5),
                               'size': dose_2_itk.GetSize(),
                               'spacing': dose_2_itk.GetSpacing()}
            dose_4_dict = {'mean': dose_4_npy.mean(),
                           'var': dose_4_npy.var(),
                           'std': dose_4_npy.std(),
                           'min': dose_4_npy.min(),
                           'max': dose_4_npy.max(),
                           'perc995': np.percentile(dose_4_npy, 99.5),
                           'size': dose_4_itk.GetSize(),
                           'spacing': dose_4_itk.GetSpacing()}
            dose_10_dict = {'mean': dose_10_npy.mean(),
                            'var': dose_10_npy.var(),
                            'std': dose_10_npy.std(),
                            'min': dose_10_npy.min(),
                            'max': dose_10_npy.max(),
                            'perc995': np.percentile(dose_10_npy, 99.5),
                            'size': dose_10_itk.GetSize(),
                            'spacing': dose_10_itk.GetSpacing()}
            attr_dict[patient_p.split('/')[-1]] = {
                'dose_full': dose_full_dict,
                'dose_4': dose_4_dict,
                'dose_10': dose_10_dict,
            }
            if dose_2_index >= 0:
                attr_dict[patient_p.split('/')[-1]]['dose_2'] = dose_2_dict
            for i in range(dose_full_npy.shape[0]):
                dose_full_slice = dose_full_npy[i]
                if dose_2_index >= 0:
                    dose_2_slice = dose_2_npy[i]
                dose_4_slice = dose_4_npy[i]
                dose_10_slice = dose_10_npy[i]
                dose_full_save_p = os.path.join(patient_dose_full_save_path, '{}__{}.npy'.format(patient_p.split('/')[-1], i))
                dose_2_save_p = os.path.join(patient_dose_2_save_path, '{}__{}.npy'.format(patient_p.split('/')[-1], i))
                dose_4_save_p = os.path.join(patient_dose_4_save_path, '{}__{}.npy'.format(patient_p.split('/')[-1], i))
                dose_10_save_p = os.path.join(patient_dose_10_save_path, '{}__{}.npy'.format(patient_p.split('/')[-1], i))
                np.save(dose_full_save_p, dose_full_slice)
                if dose_2_index >= 0:
                    np.save(dose_2_save_p, dose_2_slice)
                np.save(dose_4_save_p, dose_4_slice)
                np.save(dose_10_save_p, dose_10_slice)
                arr_max = np.percentile(dose_full_slice, 99.5)
                dose_full_slice_image = arr2image(dose_full_slice, arr_max)
                if dose_2_index >= 0:
                    dose_2_slice_image = arr2image(dose_2_slice, arr_max)
                dose_4_slice_image = arr2image(dose_4_slice, arr_max)
                dose_10_slice_image = arr2image(dose_10_slice, arr_max)
                dose_full_slice_image.save(dose_full_save_p.replace('_npy', '_png').replace('.npy', '.png'))
                if dose_2_index >= 0:
                    dose_2_slice_image.save(dose_2_save_p.replace('_npy', '_png').replace('.npy', '.png'))
                dose_4_slice_image.save(dose_4_save_p.replace('_npy', '_png').replace('.npy', '.png'))
                dose_10_slice_image.save(dose_10_save_p.replace('_npy', '_png').replace('.npy', '.png'))

    write_pickle(attr_dict, os.path.join(save_path, 'attr.pkl'))


def preprocess_itk_all_dose():
    data_path = '/home/mnt/data/AIGC/PET/nii/siemens'
    save_path = '/home/mnt/data/AIGC/PET/preprocessed/siemens'
    for dose in ['dose_2', 'dose_4', 'dose_10', 'dose_20', 'dose_50', 'dose_100', 'dose_full']:
        os.makedirs(os.path.join(save_path, '{}/train_npy'.format(dose)), exist_ok=True)
        os.makedirs(os.path.join(save_path, '{}/test_npy'.format(dose)), exist_ok=True)

    attr_dict = dict()

    for mode in ['train', 'test']:
        for patient_p in tqdm(subdirs(os.path.join(data_path, mode))):
            patient_dose_full_save_path = os.path.join(save_path, 'dose_full/{}_npy/{}'.format(mode, patient_p.split('/')[-1]))
            patient_dose_2_save_path = os.path.join(save_path, 'dose_2/{}_npy/{}'.format(mode, patient_p.split('/')[-1]))
            patient_dose_4_save_path = os.path.join(save_path, 'dose_4/{}_npy/{}'.format(mode, patient_p.split('/')[-1]))
            patient_dose_10_save_path = os.path.join(save_path, 'dose_10/{}_npy/{}'.format(mode, patient_p.split('/')[-1]))
            patient_dose_20_save_path = os.path.join(save_path, 'dose_20/{}_npy/{}'.format(mode, patient_p.split('/')[-1]))
            patient_dose_50_save_path = os.path.join(save_path, 'dose_50/{}_npy/{}'.format(mode, patient_p.split('/')[-1]))
            patient_dose_100_save_path = os.path.join(save_path, 'dose_100/{}_npy/{}'.format(mode, patient_p.split('/')[-1]))
            os.makedirs(patient_dose_full_save_path, exist_ok=True)
            os.makedirs(patient_dose_2_save_path, exist_ok=True)
            os.makedirs(patient_dose_4_save_path, exist_ok=True)
            os.makedirs(patient_dose_10_save_path, exist_ok=True)
            os.makedirs(patient_dose_20_save_path, exist_ok=True)
            os.makedirs(patient_dose_50_save_path, exist_ok=True)
            os.makedirs(patient_dose_100_save_path, exist_ok=True)

            series_n_list = subfiles(patient_p, join=False)
            dose_2_index = -1
            for i in range(len(series_n_list)):
                if dose1_2 in series_n_list[i] or dose1_22 in series_n_list[i] or dose1_23 in series_n_list[i]:
                    dose_2_index = i
                    break
            for i in range(len(series_n_list)):
                if dose1_4 in series_n_list[i] or dose1_42 in series_n_list[i] or dose1_43 in series_n_list[i]:
                    dose_4_index = i
                    break
            for i in range(len(series_n_list)):
                if dose1_10 in series_n_list[i] or dose1_102 in series_n_list[i] or dose1_103 in series_n_list[i]:
                    dose_10_index = i
                    break
            dose_20_index = -1
            for i in range(len(series_n_list)):
                if dose1_20 in series_n_list[i] or dose1_202 in series_n_list[i] or dose1_203 in series_n_list[i]:
                    dose_20_index = i
                    break
            dose_50_index = -1
            for i in range(len(series_n_list)):
                if dose1_50 in series_n_list[i] or dose1_502 in series_n_list[i] or dose1_503 in series_n_list[i]:
                    dose_50_index = i
                    break
            dose_100_index = -1
            for i in range(len(series_n_list)):
                if dose1_100 in series_n_list[i] or dose1_1002 in series_n_list[i] or dose1_1003 in series_n_list[i]:
                    dose_100_index = i
                    break
            for i in range(len(series_n_list)):
                if dose_full in series_n_list[i] or dose_full2 in series_n_list[i] or dose_full3 in series_n_list[i]:
                    dose_full_index = i
                    break
            dose_full_p = os.path.join(patient_p, series_n_list[dose_full_index])
            dose_low_p_list = []
            if dose_2_index >= 0:
                dose_2_p = os.path.join(patient_p, series_n_list[dose_2_index])
                dose_low_p_list.append(dose_2_p)
            dose_4_p = os.path.join(patient_p, series_n_list[dose_4_index])
            dose_low_p_list.append(dose_4_p)
            dose_10_p = os.path.join(patient_p, series_n_list[dose_10_index])
            dose_low_p_list.append(dose_10_p)
            if dose_20_index >= 0:
                dose_20_p = os.path.join(patient_p, series_n_list[dose_20_index])
                dose_low_p_list.append(dose_20_p)
            if dose_50_index >= 0:
                dose_50_p = os.path.join(patient_p, series_n_list[dose_50_index])
                dose_low_p_list.append(dose_50_p)
            if dose_100_index >= 0:
                dose_100_p = os.path.join(patient_p, series_n_list[dose_100_index])
                dose_low_p_list.append(dose_100_p)

            dose_full_itk, dose_low_itk_list = get_roi(dose_full_p, dose_low_p_list)

            dose_full_npy = sitk.GetArrayFromImage(dose_full_itk).astype(np.float32)
            dose_full_dict = {'mean': dose_full_npy.mean(),
                              'var': dose_full_npy.var(),
                              'std': dose_full_npy.std(),
                              'min': dose_full_npy.min(),
                              'max': dose_full_npy.max(),
                              'perc995': np.percentile(dose_full_npy, 99.5),
                              'size': dose_full_itk.GetSize(),
                              'spacing': dose_full_itk.GetSpacing()}
            attr_dict[patient_p.split('/')[-1]] = {'dose_full': dose_full_dict}
            del dose_full_itk
            if dose_2_index >= 0:
                dose_2_itk = dose_low_itk_list.pop(0)
                dose_2_npy = sitk.GetArrayFromImage(dose_2_itk).astype(np.float32)
                dose_2_dict = {'mean': dose_2_npy.mean(),
                               'var': dose_2_npy.var(),
                               'std': dose_2_npy.std(),
                               'min': dose_2_npy.min(),
                               'max': dose_2_npy.max(),
                               'perc995': np.percentile(dose_2_npy, 99.5),
                               'size': dose_2_itk.GetSize(),
                               'spacing': dose_2_itk.GetSpacing()}
                attr_dict[patient_p.split('/')[-1]]['dose_2'] = dose_2_dict
                del dose_2_itk
            dose_4_itk = dose_low_itk_list.pop(0)
            dose_4_npy = sitk.GetArrayFromImage(dose_4_itk).astype(np.float32)
            dose_4_dict = {'mean': dose_4_npy.mean(),
                           'var': dose_4_npy.var(),
                           'std': dose_4_npy.std(),
                           'min': dose_4_npy.min(),
                           'max': dose_4_npy.max(),
                           'perc995': np.percentile(dose_4_npy, 99.5),
                           'size': dose_4_itk.GetSize(),
                           'spacing': dose_4_itk.GetSpacing()}
            attr_dict[patient_p.split('/')[-1]]['dose_4'] = dose_4_dict
            del dose_4_itk
            dose_10_itk = dose_low_itk_list.pop(0)
            dose_10_npy = sitk.GetArrayFromImage(dose_10_itk).astype(np.float32)
            dose_10_dict = {'mean': dose_10_npy.mean(),
                            'var': dose_10_npy.var(),
                            'std': dose_10_npy.std(),
                            'min': dose_10_npy.min(),
                            'max': dose_10_npy.max(),
                            'perc995': np.percentile(dose_10_npy, 99.5),
                            'size': dose_10_itk.GetSize(),
                            'spacing': dose_10_itk.GetSpacing()}
            attr_dict[patient_p.split('/')[-1]]['dose_10'] = dose_10_dict
            del dose_10_itk
            if dose_20_index >= 0:
                dose_20_itk = dose_low_itk_list.pop(0)
                dose_20_npy = sitk.GetArrayFromImage(dose_20_itk).astype(np.float32)
                dose_20_dict = {'mean': dose_20_npy.mean(),
                                'var': dose_20_npy.var(),
                                'std': dose_20_npy.std(),
                                'min': dose_20_npy.min(),
                                'max': dose_20_npy.max(),
                                'perc995': np.percentile(dose_20_npy, 99.5),
                                'size': dose_20_itk.GetSize(),
                                'spacing': dose_20_itk.GetSpacing()}
                attr_dict[patient_p.split('/')[-1]]['dose_20'] = dose_20_dict
                del dose_20_itk
            if dose_50_index >= 0:
                dose_50_itk = dose_low_itk_list.pop(0)
                dose_50_npy = sitk.GetArrayFromImage(dose_50_itk).astype(np.float32)
                dose_50_dict = {'mean': dose_50_npy.mean(),
                                'var': dose_50_npy.var(),
                                'std': dose_50_npy.std(),
                                'min': dose_50_npy.min(),
                                'max': dose_50_npy.max(),
                                'perc995': np.percentile(dose_50_npy, 99.5),
                                'size': dose_50_itk.GetSize(),
                                'spacing': dose_50_itk.GetSpacing()}
                attr_dict[patient_p.split('/')[-1]]['dose_50'] = dose_50_dict
                del dose_50_itk
            if dose_100_index >= 0:
                dose_100_itk = dose_low_itk_list.pop(0)
                dose_100_npy = sitk.GetArrayFromImage(dose_100_itk).astype(np.float32)
                dose_100_dict = {'mean': dose_100_npy.mean(),
                                 'var': dose_100_npy.var(),
                                 'std': dose_100_npy.std(),
                                 'min': dose_100_npy.min(),
                                 'max': dose_100_npy.max(),
                                 'perc995': np.percentile(dose_100_npy, 99.5),
                                 'size': dose_100_itk.GetSize(),
                                 'spacing': dose_100_itk.GetSpacing()}
                attr_dict[patient_p.split('/')[-1]]['dose_100'] = dose_100_dict
                del dose_100_itk

            for i in range(dose_full_npy.shape[0]):
                dose_full_save_p = os.path.join(patient_dose_full_save_path, '{}__{}.npy'.format(patient_p.split('/')[-1], i))
                dose_2_save_p = os.path.join(patient_dose_2_save_path, '{}__{}.npy'.format(patient_p.split('/')[-1], i))
                dose_4_save_p = os.path.join(patient_dose_4_save_path, '{}__{}.npy'.format(patient_p.split('/')[-1], i))
                dose_10_save_p = os.path.join(patient_dose_10_save_path, '{}__{}.npy'.format(patient_p.split('/')[-1], i))
                dose_20_save_p = os.path.join(patient_dose_20_save_path, '{}__{}.npy'.format(patient_p.split('/')[-1], i))
                dose_50_save_p = os.path.join(patient_dose_50_save_path, '{}__{}.npy'.format(patient_p.split('/')[-1], i))
                dose_100_save_p = os.path.join(patient_dose_100_save_path, '{}__{}.npy'.format(patient_p.split('/')[-1], i))
                dose_full_slice = dose_full_npy[i]
                np.save(dose_full_save_p, dose_full_slice)
                if dose_2_index >= 0:
                    dose_2_slice = dose_2_npy[i]
                    np.save(dose_2_save_p, dose_2_slice)
                dose_4_slice = dose_4_npy[i]
                np.save(dose_4_save_p, dose_4_slice)
                dose_10_slice = dose_10_npy[i]
                np.save(dose_10_save_p, dose_10_slice)
                if dose_20_index >= 0:
                    dose_20_slice = dose_20_npy[i]
                    np.save(dose_20_save_p, dose_20_slice)
                if dose_50_index >= 0:
                    dose_50_slice = dose_50_npy[i]
                    np.save(dose_50_save_p, dose_50_slice)
                if dose_100_index >= 0:
                    dose_100_slice = dose_100_npy[i]
                    np.save(dose_100_save_p, dose_100_slice)

    write_pickle(attr_dict, os.path.join(save_path, 'attr_test.pkl'))


def get_histogram():
    data_path = '/home/mnt/data/AIGC/youshang/0827/nii_chosen'
    pkl_path = '/home/mnt/data/AIGC/youshang/0827/preprocessed/hist.pkl'
    hist_dataset = {}
    scale = 10
    dim = 128
    patient_p_list = subdirs(os.path.join(data_path, 'train')) + subdirs(os.path.join(data_path, 'test'))
    for patient_p in patient_p_list:
        for paired_p in subdirs(patient_p):
            series_n_list = subfiles(paired_p, join=False)
            if 'hospital' in data_path:
                for i in range(len(series_n_list)):
                    if 'L__' in series_n_list[i]:
                        L_index = i
                        break
                for i in range(len(series_n_list)):
                    if 'G__' in series_n_list[i] or 'H__' in series_n_list[i]:
                        H_index = i
                        break
            elif 'youshang' in data_path:
                for i in range(len(series_n_list)):
                    if 'RaDyn' in series_n_list[i] or 'SMR' in series_n_list[i]:
                        H_index = i
                        break
                L_index = 1 - H_index
            else:
                raise ValueError
            target_H_p = os.path.join(paired_p, series_n_list[H_index])
            input_p = os.path.join(paired_p, series_n_list[L_index])
            target_H_itk = sitk.ReadImage(target_H_p)
            input_itk = sitk.ReadImage(input_p)
            if target_H_itk.GetSize()[2] != input_itk.GetSize()[2]:
                print(paired_p)
                continue
            input_npy = sitk.GetArrayFromImage(input_itk)
            perc995 = np.percentile(input_npy, 99.5)
            input_npy = (np.clip(input_npy, 0, perc995) / perc995).astype(np.float32)

            # calculate histogram
            histograms, _ = np.histogram(input_npy, bins=dim, range=(0.001, 1))
            normalized_histograms = histograms / histograms.sum(keepdims=True)
            normalized_histograms *= scale

            cum_hist = np.cumsum(normalized_histograms)

            hist_diff = np.diff(normalized_histograms)
            hist_diff = np.insert(hist_diff, 0, hist_diff[0])
            hist_diff *= scale
            combined_histogram = np.concatenate((normalized_histograms, cum_hist, hist_diff)).astype(np.float32)  # num_bins * 3,

            hist_dataset['{}__{}'.format(patient_p.split('/')[-1], paired_p.split('/')[-1])] = combined_histogram

    with open(pkl_path, 'wb') as f:
        pickle.dump(hist_dataset, f)


def get_roi(dose_full_p, dose_low_p_list):
    dose_full_itk = sitk.ReadImage(dose_full_p)
    dose_low_itk_list = []
    for dose_low_p in dose_low_p_list:
        dose_low_itk_list.append(sitk.ReadImage(dose_low_p))

    minimumfilter = sitk.MinimumMaximumImageFilter()
    minimumfilter.Execute(dose_full_itk)

    binary_src = sitk.BinaryThreshold(dose_full_itk, 50, minimumfilter.GetMaximum())
    binary_src = MorphologicalOperation(binary_src, 5)
    binary_src = GetLargestConnectedCompont(binary_src)
    boundingbox = GetLargestConnectedCompontBoundingbox(binary_src)
    # print(boundingbox)  # (x,y,z,xlength,ylength,zlength)
    x1, y1, z1, x2, y2, z2 = boundingbox[0], boundingbox[1], boundingbox[2], boundingbox[0] + boundingbox[3], \
                                                                             boundingbox[1] + boundingbox[4], boundingbox[2] + boundingbox[5]
    dose_full_array = sitk.GetArrayFromImage(dose_full_itk)
    dose_low_array_list = []
    for dose_low_itk in dose_low_itk_list:
        dose_low_array_list.append(sitk.GetArrayFromImage(dose_low_itk))

    # 根据boundingbox得到ROI区域
    roi_dose_full_array = dose_full_array[z1:z2, y1:y2, x1:x2]
    roi_dose_full_image = sitk.GetImageFromArray(roi_dose_full_array)
    roi_dose_full_image.SetSpacing(dose_full_itk.GetSpacing())
    roi_dose_full_image.SetDirection(dose_full_itk.GetDirection())
    roi_dose_full_image.SetOrigin(dose_full_itk.GetOrigin())

    roi_dose_low_itk_list = []
    for i, dose_low_array in enumerate(dose_low_array_list):
        roi_dose_low_array = dose_low_array[z1:z2, y1:y2, x1:x2]
        roi_dose_low_image = sitk.GetImageFromArray(roi_dose_low_array)
        roi_dose_low_image.SetSpacing(dose_low_itk_list[i].GetSpacing())
        roi_dose_low_image.SetDirection(dose_low_itk_list[i].GetDirection())
        roi_dose_low_image.SetOrigin(dose_low_itk_list[i].GetOrigin())
        roi_dose_low_itk_list.append(roi_dose_low_image)

    return roi_dose_full_image, roi_dose_low_itk_list


def check_data():
    attr_p = '/data/data/PET/preprocessed/siemens/attr.pkl'
    attr = load_pickle(attr_p)
    size_list = []
    for patient_name in attr.keys():
        attr_patient = attr[patient_name]
        for series_name in attr_patient.keys():
            size_list.append(attr_patient[series_name]['size'])
            break
    size = np.array(size_list)
    print(size[:, 0].min(), size[:, 0].max(), size[:, 1].min(), size[:, 1].max())
    import matplotlib.pyplot as plt
    plt.hist(size[:, 2])
    plt.savefig('/home/SENSETIME/fengshixiang1/Desktop/size_z.png')


def check_data_nii():
    data_path = '/home/mnt/data/AIGC/PET/nii/uexplorer'
    for mode in ['test']:
        for patient_p in tqdm(subdirs(os.path.join(data_path, mode))):
            series_p_list = subfiles(patient_p)
            size_list = []
            for series_p in series_p_list:
                nii = sitk.ReadImage(series_p)
                size_list.append(nii.GetSize())
            size = size_list.pop(0)
            for s in size_list:
                if s != size:
                    print(patient_p.split('/')[-1])
                    break


def get_test_roi():
    data_path = '/data/data/PET/nii/siemens/test'
    save_path = '/data/data/PET/nii/siemens/test_roi'
    os.makedirs(save_path, exist_ok=True)
    for patient_p in tqdm(subdirs(data_path)):
        series_n_list = subfiles(patient_p, join=False)
        dose_2_index = -1
        for i in range(len(series_n_list)):
            if dose1_2 in series_n_list[i] or dose1_22 in series_n_list[i] or dose1_23 in series_n_list[i]:
                dose_2_index = i
                break
        for i in range(len(series_n_list)):
            if dose1_4 in series_n_list[i] or dose1_42 in series_n_list[i] or dose1_43 in series_n_list[i]:
                dose_4_index = i
                break
        for i in range(len(series_n_list)):
            if dose1_10 in series_n_list[i] or dose1_102 in series_n_list[i] or dose1_103 in series_n_list[i]:
                dose_10_index = i
                break
        for i in range(len(series_n_list)):
            if dose_full in series_n_list[i] or dose_full2 in series_n_list[i] or dose_full3 in series_n_list[i]:
                dose_full_index = i
                break

        dose_full_p = os.path.join(patient_p, series_n_list[dose_full_index])
        if dose_2_index >= 0:
            dose_2_p = os.path.join(patient_p, series_n_list[dose_2_index])
        dose_4_p = os.path.join(patient_p, series_n_list[dose_4_index])
        dose_10_p = os.path.join(patient_p, series_n_list[dose_10_index])

        if dose_2_index >= 0:
            dose_full_itk, [dose_2_itk, dose_4_itk, dose_10_itk] = get_roi(dose_full_p, [dose_2_p, dose_4_p, dose_10_p])
        else:
            dose_full_itk, [dose_4_itk, dose_10_itk] = get_roi(dose_full_p, [dose_4_p, dose_10_p])

        os.makedirs(os.path.join(save_path, dose_full_p.split('/')[-2]), exist_ok=True)
        sitk.WriteImage(dose_full_itk, os.path.join(save_path, dose_full_p.split('/')[-2], dose_full_p.split('/')[-1]))
        sitk.WriteImage(dose_2_itk, os.path.join(save_path, dose_2_p.split('/')[-2], dose_2_p.split('/')[-1]))
        sitk.WriteImage(dose_4_itk, os.path.join(save_path, dose_4_p.split('/')[-2], dose_4_p.split('/')[-1]))
        sitk.WriteImage(dose_10_itk, os.path.join(save_path, dose_10_p.split('/')[-2], dose_10_p.split('/')[-1]))


def merge_attr():
    p1 = '/home/mnt/data/AIGC/PET/preprocessed/uexplorer/attr_train.pkl'
    p2 = '/home/mnt/data/AIGC/PET/preprocessed/uexplorer/attr_test.pkl'
    attr_dict = dict()
    with open(p1, 'rb') as f:
        attr_d = pickle.load(f)
        attr_dict.update(attr_d)
    with open(p2, 'rb') as f:
        attr_d = pickle.load(f)
        attr_dict.update(attr_d)
    write_pickle(attr_dict, '/home/mnt/data/AIGC/PET/preprocessed/uexplorer/attr.pkl')


def move_some_test_to_train():
    nii_path = '/home/mnt/data/AIGC/PET/nii/uexplorer'
    npy_path = '/home/mnt/data/AIGC/PET/preprocessed/uexplorer'
    patient_ids = os.listdir(os.path.join(nii_path, 'test'))
    patient_ids.sort()
    for patient_id in patient_ids[40:]:
        shutil.move(os.path.join(nii_path, 'test', patient_id),
                    os.path.join(nii_path, 'train', patient_id))
        # print(os.path.join(nii_path, 'test', patient_id), os.path.join(nii_path, 'train', patient_id))
        for dose in ['dose_2', 'dose_4', 'dose_10', 'dose_20', 'dose_50', 'dose_100', 'dose_full']:
            shutil.move(os.path.join(npy_path, dose, 'test_npy', patient_id),
                        os.path.join(npy_path, dose, 'train_npy', patient_id))
            # print(os.path.join(npy_path, dose, 'test_npy', patient_id), os.path.join(npy_path, dose, 'train_npy', patient_id))


if __name__ == '__main__':
    # split_nii()
    # preprocess_itk_simense()
    # preprocess_itk_uexplorer()
    # preprocess_itk_all_dose()
    # check_data()
    # check_data_nii()
    # get_test_roi()
    # move_some_test_to_train()
    merge_attr()