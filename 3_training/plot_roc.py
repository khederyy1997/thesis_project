from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from my_functions import *
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def plot_roc(files_list, models_list, toScale = False):

    for file_name in files_list:

        df = pd.read_csv(f"groundtruth/frame/{file_name}.csv")
        ids_list = df["id"].unique()
        print("\n" + file_name)

        for model in models_list:

            # model name to pretty print
            model_name = str(model).split(".")[-1][:-2]
            print("--- " + model_name)

            for out in ids_list:

                print(f"------- {out}")

                # Divido in train (all subjects - 1) e test (1 subject)
                X_train, X_test, y_train, y_test = train_test_one_subject_out(df, out, isScale = toScale)               

                # Binarize the output
                y_train = label_binarize(y_train, classes = [0,1,2,3,4])
                y_test = label_binarize(y_test, classes = [0,1,2,3,4])
                n_classes = y_test.shape[1]

                # Learn to predict each class against the other
                clf = OneVsRestClassifier(model(random_state = 42))
                clf.fit(X_train, y_train)

                if model == SVC:
                    y_score = clf.decision_function(X_test)
                else:
                    y_score = clf.predict_proba(X_test)

                # Compute ROC curve and ROC area for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])

                # Compute micro-average ROC curve and ROC area
                fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

                # Plot ROC curve
                plt.figure()
                plt.plot(fpr["micro"], tpr["micro"],
                        label='micro-average ROC curve (area = {0:0.2f})'
                            ''.format(roc_auc["micro"]))
                for i in range(n_classes):
                    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                                ''.format(i, roc_auc[i]))

                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'Testing subject {out}')
                plt.legend(loc="lower right")

                plt.savefig(f"roc/frame/ID{out}_{model_name}_{str(toScale)}_scale.png")

for bool in [True, False]:
    plot_roc(   ["all"],
                [DecisionTreeClassifier],
                toScale=bool)