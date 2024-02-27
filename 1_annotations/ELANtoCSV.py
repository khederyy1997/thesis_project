from xml.dom import minidom
import os
from pathlib import Path
import pandas as pd
from globals import GROUP_IDs

os.chdir("../")

# 0: bocca chiusa
# 1: parla
# 2: porta la forchetta alla bocca
# 3: mastica
# 4: sorride / ride
# 5: mastica e parla

COLUMNS = ["id", "classe", "inizio", "fine"]

def getTimeSlot(slots, id):
    for s in slots:
        if s.getAttribute("TIME_SLOT_ID") == id:
            return s.getAttribute("TIME_VALUE")
    return None

def removeOverlappingSignals(signals):
    result = [signals[0]]
    for i in range(1, len(signals) - 1):
        signal = signals[i].copy()
        prev = signals[i - 1]
        next = signals[i + 1]
        if signal[2] > prev[3] and signal[3] < next[2]:
            result.append(signal)
        else:
            if signal[2] < prev[3] and signal[3] > prev[3]:
                signal[2] = prev[3]
            if signal[3] > next[2] and signal[2] < next[2]:
                signal[3] = next[2]
            result.append(signal)
    return result


if __name__ == "__main__":
    Path("0_data/annotations/csv_from_elan").mkdir(parents=True, exist_ok=True)
    files = os.scandir("0_data/annotations/elan")
    for file in files:
        if os.path.isfile(file):
            print(f'converting file {os.path.join("0_data/annotations/elan", file.name)}')
            result = {"Left": [], "Right": []}
            tree = minidom.parse(os.path.join("0_data/annotations/elan", file.name))
            video_id = file.name[file.name.find("_") + 1:-4]
            # openposefile = os.path.join("0_data/csv", "out" + participant_id + ".csv")

            slots = tree.getElementsByTagName("TIME_SLOT")
            tiers = tree.getElementsByTagName("TIER")

            for tier in tiers:

                behavior_class = None
                if tier.getAttribute("LINGUISTIC_TYPE_REF") == "Speaking":
                    behavior_class = 1
                elif tier.getAttribute("LINGUISTIC_TYPE_REF") == "Eating":
                    behavior_class = 3
                elif tier.getAttribute("LINGUISTIC_TYPE_REF") == "Smiling":
                    behavior_class = 4
                elif tier.getAttribute("LINGUISTIC_TYPE_REF") == "Gaze":
                    behavior_class = None
                elif tier.getAttribute("LINGUISTIC_TYPE_REF") == "Actions":
                    behavior_class = 2
                elif tier.getAttribute("LINGUISTIC_TYPE_REF") == "Errors":
                    behavior_class = None
                if behavior_class is not None:
                    side = tier.getAttribute("PARTICIPANT")
                    participant_id = GROUP_IDs[int(video_id)][side]
                    signals = tier.getElementsByTagName("ALIGNABLE_ANNOTATION")
                    for signal in signals:
                        start = getTimeSlot(slots, signal.getAttribute("TIME_SLOT_REF1"))
                        end = getTimeSlot(slots, signal.getAttribute("TIME_SLOT_REF2"))
                        if start is not None and end is not None:
                            start = int(int(start) / 1000 * 25)
                            end = int(int(end) / 1000 * 25)
                            result[side].append([participant_id, behavior_class, start, end])
        result["Left"].sort(key=lambda x: x[2])
        result["Left"] = removeOverlappingSignals(result["Left"])
        pd.DataFrame(result["Left"], columns=COLUMNS).to_csv(os.path.join(f'0_data/annotations/csv_from_elan/{GROUP_IDs[int(video_id)]["Left"]}.csv'), index=False)
        result["Right"].sort(key=lambda x: x[2])
        result["Right"] = removeOverlappingSignals(result["Right"])
        pd.DataFrame(result["Right"], columns=COLUMNS).to_csv(os.path.join(f'0_data/annotations/csv_from_elan/{GROUP_IDs[int(video_id)]["Right"]}.csv'), index=False)