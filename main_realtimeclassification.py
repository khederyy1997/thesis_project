import pickle
import time
from statistics import mean, stdev
from scipy.stats import skew, kurtosis
import pandas as pd
from ZMQ_Consumer import ZMQ_Consumer
from PML_generator import PML_generator
import socket
import csv
from sklearn.preprocessing import StandardScaler

def sendPML(pml, ip, port):
    msg = bytes("PML:" + pml.toString(), 'utf-8')
    # print(f'Sending {msg} to {ip}:{port}')
    sock = socket.socket(socket.AF_INET,
                         socket.SOCK_DGRAM)
    sock.sendto(msg, (ip, int(port)))


if __name__ == "__main__":

    BASIC = ["id", "inizio", "fine"]
    STATS = ["min", "max", "mean", "stdev", "skew", "kurt"]
    AU_names = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26", "AU45"]
    FEATURES = [x + "_" + y + "_r" for y in AU_names for x in STATS]
    COLUMNS = FEATURES

    time_window = 25
    frame_counter = 0
    AU_dict_zero = {x:[] for x in AU_names}
    AU_dict = AU_dict_zero
    id = -1
    classe = -1
    start = 0

    with open("config.csv", "r", encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            if row[0] == "zmq_connectstring":
                zmq_connectstring = row[1].strip()
            if row[0] == "zmq_hands_connectstring":
                zmq_hands_connectstring = row[1].strip()
            if row[0] == "whiteboard_ip":
                whiteboard_ip = row[1].strip()
            if row[0] == "whiteboard_port":
                whiteboard_port = row[1].strip()
            if row[0] == "modelfile":
                modelfile = row[1].strip()
            if row[0] == "camera_id":
                camera_id = int(row[1].strip())

    clf = pickle.load(open(modelfile, 'rb'))

    scaler = pickle.load(open("StandardScaler.sav", 'rb'))

    ZMQC = ZMQ_Consumer(zmq_connectstring)

    ZMQC_hands = ZMQ_Consumer(zmq_hands_connectstring)

    pml = PML_generator()

    udp_ip = whiteboard_ip
    udp_port = whiteboard_port

    while True:
        OF_text = ZMQC.consume_multipart()
        hands_data = ZMQC_hands.consume_multipart()

        if hands_data is not None:
            if hands_data[0:3] == "HA:":
                hands = [float(x) for x in hands_data[4:].split(" ")]
                pml.SetHandsPosition(hands[0], hands[1], hands[2], hands[3], hands[4], hands[5], hands[6], hands[7])
                sendPML(pml, udp_ip, udp_port)
                # print(pml.toString(pretty=True))

        if OF_text is not None:

            if OF_text[0:3] == "AU:":

                frame_counter += 1
                AUs = OF_text[5:-1].split("][")
                for AU in AUs:
                    sp = AU.split(",")
                    AUname = sp[0]
                    AUvalue = float(sp[1])
                    if AUvalue < 0:
                        AUvalue = 0
                    if AUvalue > 5:
                        AUvalue = 5
                    AU_dict[AUname].append(AUvalue)
                if frame_counter % time_window == 0:
                    ls_seq = []

                    for feature in AU_names:  # per ogni feature
                        arr_seq = AU_dict[feature]  # sequenza della singola feature

                        stats = [
                            min(arr_seq),
                            max(arr_seq),
                            mean(arr_seq),
                            stdev(arr_seq),
                            skew(arr_seq),
                            kurtosis(arr_seq)
                        ]

                        stats = [round(x, 4) for x in stats]

                        ls_seq += stats

                    df = pd.DataFrame(data=[ls_seq], columns=COLUMNS, index=[0])
                    df = scaler.transform(df)
                    df = pd.DataFrame(data=df, columns=COLUMNS, index=[0])
                    y_pred = clf.predict(df)
                    print(y_pred)

                    AU_dict = AU_dict = AU_dict_zero

            if OF_text[0:3] == "HD:":
                # print(OF_text)
                headData = [float(x) for x in OF_text[4:].split(" ")]
                pml.SetHeadPosition(headData[0], headData[1], headData[2])
                pml.SetHeadRotation(headData[3], headData[4], headData[5])
                pml.SetHeadConfidence(headData[6])
                sendPML(pml, udp_ip, udp_port)

            if OF_text[0:3] == "GZ:":
                # print(OF_text)
                gaze = [float(x) for x in OF_text[4:].split(" ")]
                pml.SetGaze(gaze[0], gaze[1])
                sendPML(pml, udp_ip, udp_port)
                # time.sleep(0.01)