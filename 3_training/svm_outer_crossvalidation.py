import pandas as pd
import sklearn
from numpy import shape
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler, NearMiss, ClusterCentroids
from imblearn.over_sampling import SMOTE
from sampling_one_out import sampling_strategy
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
# from matplotlib.pyplot import savefig
from statistics import mean
from my_functions import *
import json
import numpy as np
import seaborn as sn
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from pathlib import Path
import os
import pickle


os.chdir('../0_data')

def recall1(label4, confusion_matrix2):
    row = confusion_matrix2[label4, :]
    return confusion_matrix2[label4, label4] / row.sum()

def precision1(label5, confusion_matrix3):
    col = confusion_matrix3[:, label5]
    return confusion_matrix3[label5, label5] / col.sum()

ACTION_COL = ["AU01_r", 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r',
              'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
STATS = []
R_FEATURES = [x + "_" + y for y in ACTION_COL for x in STATS]


def reports_to_txt(file_name, models_list, toScale=False, resampled=False, resampleTechnique=0):

    if resampled:
        printres = str(resampleTechnique)
    else:
        printres = "no"
    print(f'\n\n\n\n{file_name} - resample: {printres}')
    listona = ["precision", "recall", "f1-score"]

    if toScale:
        df = pd.read_csv(f"groundtruth/sequence/{file_name}_normalized.csv")
    else:
        df = pd.read_csv(f"groundtruth/sequence/{file_name}.csv")

    #ids_list = df["id"].unique()
    labels = df["classe"].unique()

    print("\n" + file_name)

    for model in models_list:

        # model name to pretty print
        model_name = str(model).split(".")[-1][:-2]
        print("--- " + model_name)
        lista = []
        tabellone = {}

        #MI SERVE PER FARE OUTER CROSSVALIDATION
        if not resampled:
            X = np.array(df.drop(["classe", "id", "inizio", "fine"], axis=1))
        else:
            X = np.array(df.drop(["classe", "id"], axis=1))

        X = np.array(df.drop(["classe", "id", "inizio", "fine"], axis=1))
        y = df["classe"]

        #SE VOGLIO TOGLIERE TUTTE LE MINs
        #X = np.array(df.drop(["classe", "id","min_AU01_r", "min_AU02_r", "min_AU04_r", "min_AU05_r", "min_AU06_r", "min_AU07_r", "min_AU09_r", "min_AU10_r", "min_AU12_r", "min_AU14_r", "min_AU15_r", "min_AU17_r", "min_AU20_r", "min_AU23_r", "min_AU25_r", "min_AU26_r",
        #"min_AU45_r"], axis=1))

        #faccio outer crossvalidation
        kf = sklearn.model_selection.StratifiedKFold(5, shuffle=True)

        out = 0

        #questo for e per crossvalidation,
        for train_index, test_index in kf.split(X,y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if resampled:
                if resampleTechnique == 0:
                    undersample = ClusterCentroids
                elif resampleTechnique == 1:
                    undersample = NearMiss
                else:
                    undersample = RandomUnderSampler

                count = y_train.value_counts()
                n_samples = count.mean().astype(np.int64)

                under_sampler = undersample(
                    sampling_strategy=sampling_strategy(X_train, y_train, n_samples, t="majority"))
                X_under, y_under = under_sampler.fit_resample(X_train.copy(), y_train.copy())

                # oversampling
                over_sampler = SMOTE(sampling_strategy=sampling_strategy(X_under, y_under, n_samples, t='minority'),
                                     k_neighbors=2)
                X_train, y_train = over_sampler.fit_resample(X_under, y_under)
                count = y_train.value_counts()

            print(f'******************* TURN OUT: {out} ********************')

            tabella = {"precision": None, "recall": None, "f1-score": None, "accuracy": None}

            # leave one out
            #X_train, X_test, y_train, y_test = train_test_one_subject_out(df, out, isScale = toScale)

            # classic train test
            #if not resampled:
            #    X_train, X_test, y_train, y_test = train_test_split(df.drop(R_FEATURES + ["classe", "id", "inizio", "fine"], axis=1), df["classe"], test_size=0.20)
            #else:
            #     X_train, X_test, y_train, y_test = train_test_split(df.drop(R_FEATURES + ["classe", "id"], axis=1), df["classe"], test_size=0.20)

            # Creo e alleno il modello
            # clf = model(random_state=42)

            parameters = {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100, 1000, 10000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1]}

            # time window 25 - accuracy 0.65
            # parameters = {'kernel': ['rbf'], 'C': [10000], 'gamma': [0.00001]}

            # time window 20 - accuracy 0.64
            # parameters = {'kernel': ['rbf'], 'C': [100], 'gamma': [0.001]}

            # time window 25 - accuracy 0.71
            #parameters = {'kernel': ['rbf'], 'C': [10], 'gamma': [0.01]}

            # parameters = {'C': [10], 'gamma': [0.01], 'kernel': ['rbf']}
            # parameters = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}

            #use scoring="f1_weighted" quando set è inbalanced
            clf = GridSearchCV(model, parameters, verbose=0, cv=5, scoring="f1_macro")

            # passare tutto il dataset (senza le colonne id, classe, inizio e fine da X)
            clf.fit(X_train, y_train)

            pickle.dump(clf, open(f"../trained_model_{file_name}-{toScale}.sav", 'wb'))
            # exit(0)

            # Predict sul modello allenato
            y_true, y_pred = y_test, clf.predict(X_test)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_["mean_test_score"]
            stds = clf.cv_results_["std_test_score"]
            for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()

            report = classification_report(y_true, y_pred, output_dict=True)
            print(report)
            print()

            # Salvo i risultati in un file txt
            if out // 10 == 0:
                out_str = f"0{out}"
            else:
                out_str = f'{out}'

            # Confusion Matrix
            array = confusion_matrix(y_true, y_pred)
            lista.append(array)
            print(array)
            # plot_confusion_matrix(clf, X_test, y_test)
            # savefig(f"confusion_matrix/{file_name}/ID{out}_{model_name}_{str(toScale)}_scale.png")

            tabella["accuracy"] = report["accuracy"]

            print("accuracy: ", report["accuracy"])

            for measure in listona:
                tabella[measure] = ([report[str(cls)][measure] for cls in range(1, 5)], report["macro avg"][measure],
                                    report["weighted avg"][measure])

                tabellone[out_str] = tabella

        sum_array = ([[sum(matrix[i][j] for matrix in lista) for j in range(len(array[i]))] for i in range(len(array))])
        print(sum_array)
        df_cm = pd.DataFrame(sum_array)

        print("QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ")

        mean_of_conf_matrix_arrays = np.sum(lista, axis=0)

        print("Cofusion marix")
        print(mean_of_conf_matrix_arrays)

        print("Accuracy")
        diagonal_sum = mean_of_conf_matrix_arrays.trace()
        sum_of_all_elements = mean_of_conf_matrix_arrays.sum()
        print(diagonal_sum / sum_of_all_elements)

        print("label precision recall")
        # print(f"{precision('1', mean_of_conf_matrix_arrays):9.3f} {recall('1', mean_of_conf_matrix_arrays):6.3f}")

        print(recall1(0, mean_of_conf_matrix_arrays))
        print(recall1(1, mean_of_conf_matrix_arrays))
        print(recall1(2, mean_of_conf_matrix_arrays))
        print(recall1(3, mean_of_conf_matrix_arrays))
        print(precision1(0, mean_of_conf_matrix_arrays))
        print(precision1(1, mean_of_conf_matrix_arrays))
        print(precision1(2, mean_of_conf_matrix_arrays))
        print(precision1(3, mean_of_conf_matrix_arrays))

        sumf1 = 0
        sumf2 = 0
        sumf3 = 0
        sumf4 = 0
        sumpr = 0
        sumrc = 0

        print("f1")
        print((2.0 * precision1(0, mean_of_conf_matrix_arrays) * recall1(0, mean_of_conf_matrix_arrays)) / (
                precision1(0, mean_of_conf_matrix_arrays) + recall1(0, mean_of_conf_matrix_arrays)))

        sumf1 = (2.0 * precision1(0, mean_of_conf_matrix_arrays) * recall1(0, mean_of_conf_matrix_arrays)) / (
                precision1(0, mean_of_conf_matrix_arrays) + recall1(0, mean_of_conf_matrix_arrays))

        sumpr = precision1(0, mean_of_conf_matrix_arrays)
        sumrc = recall1(0, mean_of_conf_matrix_arrays)

        print((2.0 * precision1(1, mean_of_conf_matrix_arrays) * recall1(1, mean_of_conf_matrix_arrays)) / (
                precision1(1, mean_of_conf_matrix_arrays) + recall1(1, mean_of_conf_matrix_arrays)))

        sumf2 = (2.0 * precision1(1, mean_of_conf_matrix_arrays) * recall1(1, mean_of_conf_matrix_arrays)) / (
                precision1(1, mean_of_conf_matrix_arrays) + recall1(1, mean_of_conf_matrix_arrays))

        sumpr += precision1(1, mean_of_conf_matrix_arrays)
        sumrc += recall1(1, mean_of_conf_matrix_arrays)

        print((2.0 * precision1(2, mean_of_conf_matrix_arrays) * recall1(2, mean_of_conf_matrix_arrays)) / (
                precision1(2, mean_of_conf_matrix_arrays) + recall1(2, mean_of_conf_matrix_arrays)))

        sumf3 = (2.0 * precision1(2, mean_of_conf_matrix_arrays) * recall1(2, mean_of_conf_matrix_arrays)) / (
                precision1(2, mean_of_conf_matrix_arrays) + recall1(2, mean_of_conf_matrix_arrays))

        sumpr += precision1(2, mean_of_conf_matrix_arrays)
        sumrc += recall1(2, mean_of_conf_matrix_arrays)

        print((2.0 * precision1(3, mean_of_conf_matrix_arrays) * recall1(3, mean_of_conf_matrix_arrays)) / (
                precision1(3, mean_of_conf_matrix_arrays) + recall1(3, mean_of_conf_matrix_arrays)))

        sumf4 = (2.0 * precision1(3, mean_of_conf_matrix_arrays) * recall1(3, mean_of_conf_matrix_arrays)) / (
                precision1(3, mean_of_conf_matrix_arrays) + recall1(3, mean_of_conf_matrix_arrays))

        sumpr += precision1(3, mean_of_conf_matrix_arrays)
        sumrc += recall1(3, mean_of_conf_matrix_arrays)

        print("macro Fscore")
        print((sumf1 + sumf2 + sumf3 + sumf4) / 4)

        print("macro Precision")
        print(sumpr / 4)

        print("macro Recall")
        print(sumrc / 4)

        row1 = 0
        row2 = 0
        row3 = 0
        row4 = 0

        row1 = mean_of_conf_matrix_arrays[0, :]
        row2 = mean_of_conf_matrix_arrays[1, :]
        row3 = mean_of_conf_matrix_arrays[2, :]
        row4 = mean_of_conf_matrix_arrays[3, :]

        print("weighted Fscore")
        print((sumf1 * row1.sum() + sumf2 * row2.sum() + sumf3 * row3.sum() + sumf4 * row4.sum()) / (
                    row1.sum() + row2.sum() + row3.sum() + row4.sum()))

Path("json/").mkdir(parents=True, exist_ok=True)

# reports_to_txt("50fps", [svm.SVC()], toScale=False, resampled=True, resampleTechnique=1)
# reports_to_txt("50fps", [svm.SVC()], toScale=False, resampled=True, resampleTechnique=2)
# reports_to_txt("25fps", [svm.SVC()], toScale=False, resampled=True, resampleTechnique=1)
# reports_to_txt("25fps", [svm.SVC()], toScale=False, resampled=True, resampleTechnique=2)
# reports_to_txt("10fps", [svm.SVC()], toScale=False, resampled=True, resampleTechnique=1)
# reports_to_txt("10fps", [svm.SVC()], toScale=False, resampled=True, resampleTechnique=2)
# reports_to_txt("50fps", [svm.SVC()], toScale=False, resampled=False)
# reports_to_txt("25fps", [svm.SVC()], toScale=True, resampled=False)
# reports_to_txt("10fps", [svm.SVC()], toScale=True, resampled=False)
# reports_to_txt("25fps", [svm.SVC()], toScale=False, resampled=False)
# reports_to_txt("10fps", [svm.SVC()], toScale=False, resampled=False)

# reports_to_txt("50fps", [svm.SVC()], toScale=False, resampled=False, resampleTechnique=2)
reports_to_txt("50fps", [svm.SVC()], toScale=True, resampled=False)