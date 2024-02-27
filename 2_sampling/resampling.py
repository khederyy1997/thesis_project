from imblearn.under_sampling import RandomUnderSampler, NearMiss, ClusterCentroids
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import os
from pathlib import Path


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def sampling_strategy(X,y,n_samples, t='majority'):
    target_classes = ''
    if t == 'majority':
        target_classes = y.value_counts() > n_samples
    elif t == 'minority':
        target_classes = y.value_counts() < n_samples
    tc = target_classes[target_classes == True].index
    #target_classes_all = y.value_counts().index
    sampling_strategy = {}
    for target in tc:
        sampling_strategy[target] = n_samples
    return sampling_strategy

def resample_to_csv(groundtruth_list, undersample_techniques):

    print("\n# - # UNDER & OVER SAMPLING # - #")
    print(f"Resampling dei file: {groundtruth_list}\n")

    total = len(undersample_techniques)

    for file in groundtruth_list:

        file_name = file.split(".")[0]

        # Reading the csv file
        df = pd.read_csv(f'groundtruth/sequence/{file_name}.csv')
        df.drop(["inizio", "fine"], axis=1, inplace=True)

        features = []
        for feature in df.columns:
            if feature not in ["id", "classe"]:
                features.append(feature)
        X = df[features]
        y = df[["id", 'classe']]

        count = y.value_counts()
        new_y = y["id"].astype(str) + "_" + y["classe"].astype(str)
        n_samples = count.mean().astype(np.int64)

        printProgressBar(0, total, prefix = file_name)

        for i, model in enumerate(undersample_techniques):

            # undersampling
            name_undersampling = str(model).split(".")[-1][:-2]

            under_sampler = model(sampling_strategy=sampling_strategy(X, new_y, n_samples, t="majority"))
            X_under, y_under = under_sampler.fit_resample(X.copy(), new_y.copy())

            # oversampling
            over_sampler = SMOTE(sampling_strategy=sampling_strategy(X_under, y_under,n_samples, t='minority'),k_neighbors=2)
            X_bal, y_bal = over_sampler.fit_resample(X_under, y_under)


            # rimetto le colonne come prima
            new_y_bal = y_bal.str.split('_')
            df3 = pd.DataFrame(new_y_bal.to_list(), columns=['id','classe'])

            # ricreo il df iniziale
            new_df = pd.concat([X_bal, df3], axis=1)

            # salvo il file 
            new_df.to_csv(f"groundtruth/sequence/resampled/{file_name}_{name_undersampling}.csv", index = False)

            printProgressBar(i+1, total, prefix = file_name)

os.chdir("../0_data")
Path("groundtruth/sequence/resampled").mkdir(parents=True, exist_ok=True)
resample_to_csv(["10fps_normalized.csv"], [RandomUnderSampler, NearMiss, ClusterCentroids])