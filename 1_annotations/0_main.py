import os, json
import pickle

from globals import printProgressBar, annotationCompleted, flatten
from frame import all_annotation_to_groundtruth
from sequence import make_csv_of_sequences
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

def absolute_maximum_scale(series):
    return series / series.abs().max()

os.chdir('../')

print("\n# - # UPDATE CSV ANNOTAZIONI # - #\n")

Path("0_data/groundtruth/frame/").mkdir(parents=True, exist_ok=True)
Path("0_data/groundtruth/sequence/").mkdir(parents=True, exist_ok=True)

with open("1_annotations/dictCol.json", "r") as f:
    json_file = f.read()
    DICT_COL = json.loads(json_file)

COLUMNS = ['GAZE', 'POSE', 'LANDMARK2D', 'LANDMARK3D', 'RIGID_SHAPE', 'ACTION_UNITS']
ALL = flatten([DICT_COL[x] for x in COLUMNS])
SOME = ['pose_Rx', 'pose_Ry', 'pose_Rz', 'X_0', 'X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9', 'X_10', 'X_11', 'X_12', 'X_13', 'X_14', 'X_15', 'X_16', 'X_17', 'X_18', 'X_19', 'X_20', 'X_21', 'X_22', 'X_23', 'X_24', 'X_25', 'X_26', 'X_27', 'X_28', 'X_29', 'X_30', 'X_31', 'X_32', 'X_33', 'X_34', 'X_35', 'X_36', 'X_37', 'X_38', 'X_39', 'X_40', 'X_41', 'X_42', 'X_43', 'X_44', 'X_45', 'X_46', 'X_47', 'X_48', 'X_49', 'X_50', 'X_51', 'X_52', 'X_53', 'X_54', 'X_55', 'X_56', 'X_57', 'X_58', 'X_59', 'X_60', 'X_61', 'X_62', 'X_63', 'X_64', 'X_65', 'X_66', 'X_67', 'Y_0', 'Y_1', 'Y_2', 'Y_3', 'Y_4', 'Y_5', 'Y_6', 'Y_7', 'Y_8', 'Y_9', 'Y_10', 'Y_11', 'Y_12', 'Y_13', 'Y_14', 'Y_15', 'Y_16', 'Y_17', 'Y_18', 'Y_19', 'Y_20', 'Y_21', 'Y_22', 'Y_23', 'Y_24', 'Y_25', 'Y_26', 'Y_27', 'Y_28', 'Y_29', 'Y_30', 'Y_31', 'Y_32', 'Y_33', 'Y_34', 'Y_35', 'Y_36', 'Y_37', 'Y_38', 'Y_39', 'Y_40', 'Y_41', 'Y_42', 'Y_43', 'Y_44', 'Y_45', 'Y_46', 'Y_47', 'Y_48', 'Y_49', 'Y_50', 'Y_51', 'Y_52', 'Y_53', 'Y_54', 'Y_55', 'Y_56', 'Y_57', 'Y_58', 'Y_59', 'Y_60', 'Y_61', 'Y_62', 'Y_63', 'Y_64', 'Y_65', 'Y_66', 'Y_67', 'Z_0', 'Z_1', 'Z_2', 'Z_3', 'Z_4', 'Z_5', 'Z_6', 'Z_7', 'Z_8', 'Z_9', 'Z_10', 'Z_11', 'Z_12', 'Z_13', 'Z_14', 'Z_15', 'Z_16', 'Z_17', 'Z_18', 'Z_19', 'Z_20', 'Z_21', 'Z_22', 'Z_23', 'Z_24', 'Z_25', 'Z_26', 'Z_27', 'Z_28', 'Z_29', 'Z_30', 'Z_31', 'Z_32', 'Z_33', 'Z_34', 'Z_35', 'Z_36', 'Z_37', 'Z_38', 'Z_39', 'Z_40', 'Z_41', 'Z_42', 'Z_43', 'Z_44', 'Z_45', 'Z_46', 'Z_47', 'Z_48', 'Z_49', 'Z_50', 'Z_51', 'Z_52', 'Z_53', 'Z_54', 'Z_55', 'Z_56', 'Z_57', 'Z_58', 'Z_59', 'Z_60', 'Z_61', 'Z_62', 'Z_63', 'Z_64', 'Z_65', 'Z_66', 'Z_67', 'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
POSE_T = ['pose_Tx', 'pose_Ty', 'pose_Tz']
LANDMARK_COL = flatten([DICT_COL["LANDMARK3D"]])
ACTION_COL = ["AU01_r", 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
BEST_AU = ['AU04_r', 'AU09_r', 'AU12_r', 'AU15_r', 'AU20_r', 'AU25_r']
EXCLUDED = ["0_test", "repository"]
ATTRIBUTI = ["id", "classe", "frame", "end_seq"] + ACTION_COL

# annotationCompleted("0_data/annotations/csv", EXCLUDED)
# annotationCompleted("0_data/annotations/csv_from_elan", EXCLUDED)

# Creo il dataframe e lo salvo in formato csv (all.csv)
ans = "n"
if os.path.isfile(f"0_data/groundtruth/frame/all.csv"):
    ans = input(f"file 0_data/groundtruth/frame/all.csv exists, do you want to overwrite it (y/N)? ")
if ans == "y" or not os.path.isfile(f"0_data/groundtruth/frame/all.csv"):
    dataset = pd.DataFrame(flatten(all_annotation_to_groundtruth(ACTION_COL, "0_data")), columns = ATTRIBUTI)
    dataset.to_csv(f"0_data/groundtruth/frame/all.csv", index = False)



# range_FPS = [10,25,50]
range_FPS = [50]


print(f"\nGenerando i file csv [sequence] dai file csv [frame]: \n")

# Creo altri csv che tengono conto delle sequenze e non dei frame
for i, FPS in enumerate(range_FPS):
    ans = "n"
    if os.path.isfile(f"0_data/groundtruth/sequence/{FPS}fps.csv"):
        ans = input(f"0_data/groundtruth/sequence/{FPS}fps.csv exists, do you want to overwrite it (y/N)? ")
    if ans == "y" or not os.path.isfile(f"0_data/groundtruth/sequence/{FPS}fps.csv"):
        DF = make_csv_of_sequences(FPS, "0_data/groundtruth/frame/all.csv")
        DF.to_csv(f"0_data/groundtruth/sequence/{FPS}fps.csv", index = False)
    else:
        DF = pd.read_csv(f"0_data/groundtruth/sequence/{FPS}fps.csv", index_col=False)

    object = StandardScaler()

    BASIC = ["id", "classe", "inizio", "fine"]
    STATS = ["mean", "stdev", "mad", "iqr", "en", "rms"]
    FEATURES = [x + "_" + y for y in ACTION_COL for x in STATS]
    COLUMNS = BASIC + FEATURES  # colonne del nuovo csv

    # standardization
    DF_std = pd.DataFrame(object.fit_transform(DF[FEATURES]), columns=FEATURES)
    DF_std = pd.concat([DF[BASIC], DF_std], axis=1)

    pickle.dump(object, open("./StandardScaler.sav", "wb"))


    # for col in DF.columns:
    #     if col in ACTION_COL:
    #         DF[col] = absolute_maximum_scale(DF[col])
    #         DF[col] = list(map(lambda x: round(x, ndigits=4), DF[col]))

    DF_std.to_csv(f"0_data/groundtruth/sequence/{FPS}fps_normalized.csv", index=False)

print("finished generating dataset")