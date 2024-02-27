venv/Scripts/pyinstaller .\main_realtimeclassification.py --add-data "trained_model.sav;./" --add-data "config.csv;./" --add-data "StandardScaler.sav;./" --collect-submodules "sklearn"
