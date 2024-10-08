import pandas as pd


def process_log(log_p):
    metric_list = [[], [], []]
    with open(log_p, 'r') as f:
        i = 0
        for line in f.readlines():
            if 'var' in line:
                line_split = line.split('var')
                psnr = float(line_split[1].split(',')[0])
                ssim = float(line_split[2].split(',')[0])
                nmse = float(line_split[3].split(',')[0])
                if i % 2 == 0:
                    metric_list[0].append(round(psnr, 2))
                    metric_list[1].append(round(ssim, 3))
                    metric_list[2].append(round(nmse, 4))
                i += 1
    df = pd.DataFrame({'psnr': metric_list[0], 'ssim': metric_list[1], 'nmse': metric_list[2]})
    df = df.transpose()
    df.to_excel(log_p.replace('.log', '.xlsx'))


if __name__ == '__main__':
    # process_log('/data/Projects/bbdm_pet/output/bbdm_dose4-10_context_normtype3/train.log')
    # process_log('/data/Projects/bbdm_pet/output/bbdm_dose4_context_normtype3/train.log')
    # process_log('/data/Projects/bbdm_pet/output/bbdm_multidose_context_normtype3/train.log')
    process_log('/data/Projects/bbdm_pet/output/bbdm_multidose_context_normtype3_2/train.log')
    # process_log('/data/Projects/bbdm_pet/output/bbdm_dose4_context_reg_normtype3/train.log')