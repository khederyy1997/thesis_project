import cv2
import zmq
import mediapipe as mp
import time
import csv
import matplotlib.pyplot as plt

def zmq_init(zmq_connectstring):
    global context, zmq_sender
    context = zmq.Context()
    zmq_sender = context.socket(zmq.PUSH)
    print(f"connecting to zmq... [{zmq_connectstring}]", end="")
    zmq_sender.bind(zmq_connectstring)
    time.sleep(0.5)
    print("done")


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

zmq_init(zmq_hands_connectstring)

# utils for drawing on image
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# mediapipe pose model
mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5)

# define a video capture object
vid = cv2.VideoCapture(camera_id)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    image = frame
    # convert image to RGB (just for input to model)
    image_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # get results using mediapipe
    results = mp_pose.process(image_input)

    time.sleep(0.01)

    if not results.pose_landmarks:
        print("no results found")
    else:
        left = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        right = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        # print(results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST])
        zmq_sender.send(b"HA: %f %f %f %f %f %f %f %f" % (left.x, left.y, left.z, left.visibility, right.x, right.y, right.z, right.visibility))
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            None,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
