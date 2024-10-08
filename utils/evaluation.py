
import os
import pathlib
import glob
import json
import numpy as np
import SimpleITK as sitk 
import nibabel as nib
from skimage.metrics import structural_similarity as compare_ssim

import pandas as pd
pd.options.mode.chained_assignment = None
import radiomics
# radiomics.logger.setLevel('ERROR')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def compute_nrmse(real, pred):
    mse = np.mean(np.square(real - pred))
    nrmse = np.sqrt(mse) / (np.max(real)-np.min(real))
    return nrmse

def compute_mse(real, pred):
    mse = np.mean(np.square(real-pred))
    return mse


def compute_psnr(real, pred):
    PIXEL_MAX = np.max(real)
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(np.mean(np.square(real - pred))))
    return psnr


def compute_ssim(real, pred):
    # ssim = compare_ssim(real/ float(np.max(real)), pred/ float(np.max(pred)))
    ssim = compare_ssim(real/ float(np.max(real)), pred/ float(np.max(pred)), data_range=1)
    return ssim


def compute_mape(real, pred):
    ok_idx = np.where(real!=0)
    mape = np.mean(np.abs((real[ok_idx] - pred[ok_idx]) / real[ok_idx]))
    return mape


def compute_suv(img, mask):
    """Compute metrics of SUV_mean, SUV_max and TotalLesion_Metabolism"""
    volume = np.count_nonzero(mask)
    crop_idx = np.nonzero(mask)
    suv_mean = np.mean(img[crop_idx])
    suv_max = np.max(img[crop_idx])
    tlg = volume * suv_mean
    psnr = compute_psnr(real=mask, pred=img)
    return suv_mean, suv_max, tlg, psnr


def percentage_error(real, pred):
    return abs(pred - real) / abs(real) * 100


class Ultra_low_dose():
    def __init__(self):
        self.real_dir = './ground-truth'
        self.pred_dir = './test'
        self.mask_dir = './mask'
        self.meta_info = './meta_info.csv'
        self.output_path = './output'
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.report_csv = os.path.join(self.output_path, 'report.csv')
        self.selected_feature_list = [
            'firstorder_RootMeanSquared',
            'firstorder_90Percentile',
            'firstorder_Median',
            'glrlm_HighGrayLevelRunEmphasis', 
            'glszm_ZonePercentage',
            'glcm_JointAverage'
        ]
        self.SUV_feature_list = ['SUV_mean', 'SUV_max', 'PSNR', 'TotalLesionMetabolism'] 
        self.organ_list = ['liver', 'heart', 'kidneyRight', 'kidneyLeft']
        self.extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(binCount=100)

    def _load_data(self, subject_file):
        """
        load a subject's full-dose and reconstructed data into array
        :param subject_file: nifti file name of the chosen subject

        :return: array of shape (2, 440, 440, 644) consisting of paired fulldose/generated images
        """
        pred = nib.load(glob.glob(os.path.join(self.pred_dir, subject_file, '*.nii.gz'))[0]).get_fdata()
        real = nib.load(glob.glob(os.path.join(self.real_dir, subject_file, '*.nii.gz'))[0]).get_fdata()
        mask = nib.load(glob.glob(os.path.join(self.mask_dir, subject_file, '*.nii.gz'))[0]).get_fdata()
        return pred, real, mask

    def _get_metainfo(self, subject:str):
        df = pd.read_csv(self.meta_info)
        subject_serie = df[df['PID'] == subject]

        return {
                'weight': subject_serie['weight'].values[0],
                'dose': subject_serie['dose'].values[0],
                'DRF': subject_serie['DRF'].values[0]
                }

    def _get_local_MAPE(self, row): 
        """
        compute weighted MAPE of all local feature evaluation values
        :param row: dataframe row containing all local feature evaluation of a subject

        :return weighted MAPE value
        """
        sum = 0
        feature_ls = self.SUV_feature_list + self.selected_feature_list
        for feature in feature_ls:
            if 'SUV' in feature:
                weight = 0.2 
            elif feature == 'TotalLesionMetabolism' or feature == 'PSNR':
                weight = 0.15
            else:
                weight = 0.05
            sum += weight * row[feature+'_percentage_error']
        return sum / len(feature_ls)

    def _get_subject_SCORE(self, row):
        """
        compute weighted score based on global (NRMSE, PSNR, SSIM) metrics and local MAPE metric
        using positive values of PSNR and SSIM, negative values of NRMSE and MAPE error
        :param row: dataframe row containing all four global and local features

        :return weighted score value
        """
        global_score = 0.4 * row['PSNR'] + 0.2 * row['SSIM'] - 0.4 * row['NRMSE']
        local_score = - row['MAPE']
        return (0.5 * global_score + 0.5 * local_score) * row['weight']

    def _evaluate_single_global(self, subject_file:str):
        """
        compute single subject NRMSE, PSNR and SSIM metrics
        :param subject_file: subject nifti filename

        :return dataframe evaluting global nrmse, psnr, ssim metrics of one subject
        """
        sub_pred, sub_real, _ = self._load_data(subject_file)
        return pd.DataFrame({
            'PID': [subject_file.split(os.extsep)[0]],
            'NRMSE': [compute_nrmse(real=sub_real, pred=sub_pred)], 
            'PSNR': [compute_psnr(real=sub_real, pred=sub_pred)], 
            'SSIM': [compute_ssim(real=sub_real, pred=sub_pred)]
        })

    def _add_drf_weight(self, subject_file):
        weight_dic = {'4': 0.05, '10': 0.15, '20':0.2, '50':0.25, '100':0.35}
        meta = self._get_metainfo(subject_file.split(os.extsep)[0])

        return weight_dic[str(meta['DRF'])]

    def _evaluate_single_local(self, subject_file:str):
        """
        compute single subject SUV, selected local feature metrics
        :param subject_file: subject nifti filename

        :return dataframw evaluating local SUV and selected features
        """
        # Find the subject's meta information
        subject = subject_file.split(os.extsep)[0]
        meta = self._get_metainfo(subject)
        SUV_ratio = meta['weight'] * 1000 / meta['dose']

        gen, original, sphere = self._load_data(subject_file)

        # Generate fulldose, prediction array
        original_ary = original * SUV_ratio
        gen_ary = gen * SUV_ratio
        original_sitk = sitk.GetImageFromArray(original_ary)
        gen_sitk = sitk.GetImageFromArray(gen_ary)

        # Iterate through the organ list
        eval_df = pd.DataFrame()
        for label, organ in enumerate(self.organ_list, start=1):
            df = pd.DataFrame({'PID': [subject], 'Organ': [organ]})
            # print('{}-{}---start'.format(subject, organ))
            organ_mask = np.where(sphere==label, sphere, 0)
            organ_mask = np.where(organ_mask!=label, organ_mask, 1)
            mask_sitk = sitk.GetImageFromArray(organ_mask)

            # Comopute selected features
            original_feature_vector = self.extractor.execute(original_sitk, mask_sitk)
            gen_feature_vector = self.extractor.execute(gen_sitk, mask_sitk)
            for feature in self.selected_feature_list:
                real, pred = original_feature_vector['original_'+feature], gen_feature_vector['original_'+feature]
                feature_df = pd.DataFrame({
                    '{}_real'.format(feature): real,
                    '{}_pred'.format(feature): pred,
                    '{}_percentage_error'.format(feature): [percentage_error(real=real, pred=pred)]
                })
                df = pd.concat([df, feature_df], axis=1)

            # Compute SUV features
            for i, suv_feature in enumerate(self.SUV_feature_list):
                real, pred = compute_suv(original_ary, organ_mask)[i], compute_suv(gen_ary, organ_mask)[i]
                feature_df = pd.DataFrame({
                    '{}_real'.format(suv_feature): real,
                    '{}_pred'.format(suv_feature): pred,
                    '{}_percentage_error'.format(suv_feature): [percentage_error(real=real, pred=pred)]
                })
                df = pd.concat([df, feature_df], axis=1)
            eval_df = eval_df.append(df)
        return eval_df

    def compute_global_metrics(self):
        """
        evaluation global NRMSE, PSNR, SSIM metrics for all predicted (model generated) subjects

        :return: dataframe containing NRMSE, PSNR, SSIM metric evaluation results
        """
        assert len(os.listdir(self.real_dir)) == len(os.listdir(self.pred_dir)) == self.num_subject, print("Mismatch subject amount")

        global_df = pd.DataFrame()
        for subject_file in os.listdir(self.pred_dir):
            eval_df = self._evaluate_single_global(subject_file)
            global_df = global_df.append(eval_df)
        global_df = global_df[['PID', 'NRMSE', 'PSNR', 'SSIM']]
        return global_df

    def compute_local_metrics(self, verbose=False, organ_verbose=False):
        """
        evaluate local SUV, selected metrics for all subjects
        :param verbose: boolean value specifying whether return all local features
        :param organ_verbose: boolean value specifying whether return local percentage error averaged across all organs

        :return: dataframe containing all SUV, selected feature metric evaluation results
        """
        local_df = pd.DataFrame()
        for subject_file in os.listdir(self.pred_dir):
            eval_df = self._evaluate_single_local(subject_file)
            local_df = local_df.append(eval_df)
        if verbose:
            return local_df
        elif organ_verbose:
            local_df = local_df[[col for col in local_df.columns if 'percentage_error' in col or col=='Organ' or col=='PID']]
            return local_df.groupby('PID').agg('mean')
        else:
            local_df = local_df[[col for col in local_df.columns if 'percentage_error' in col or col=='Organ' or col=='PID']]
            local_df['MAPE'] = local_df.apply(lambda row: self._get_local_MAPE(row), axis=1)
            return local_df.groupby('PID').agg('mean')[['MAPE']]
        

    def evaluate_all(self):
        """
        evaluate global and local metrics of all subjects
        : param save_to_file: boolean value specifying whether save evaluation report to csv

        :return: dataframe of evaluation result containing [global_NRMSE, global_PSNR, global_SSIM, local_MAPE]
        """
        all_df = pd.DataFrame()
        for subject_file in [x for x in os.listdir(self.pred_dir) if not x.startswith('.')]:
            print('Evaluating subject ' + subject_file)
            global_df = self._evaluate_single_global(subject_file) 
            local_df = self._evaluate_single_local(subject_file)
            local_df = local_df[[col for col in local_df.columns if 'percentage_error' in col or col=='Organ' or col=='PID']]
            local_df['MAPE'] = local_df.apply(lambda row: self._get_local_MAPE(row), axis=1)
            local_df = local_df.groupby('PID').agg('mean')[['MAPE']]

            eval_df = pd.merge(global_df, local_df, on=['PID'], how='inner')
            eval_df['weight'] = self._add_drf_weight(subject_file)
            all_df = all_df.append(eval_df)

            all_df['SCORE'] = all_df.apply(lambda row: self._get_subject_SCORE(row), axis=1)
            all_df = all_df[['PID', 'NRMSE', 'PSNR', 'SSIM', 'MAPE', 'SCORE']]
            all_df.to_csv(self.report_csv, index=False)

if __name__ == "__main__":
    eva = Ultra_low_dose().evaluate_all()
