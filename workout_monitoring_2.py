import cv2

import solutions2

from ai_gym3 import AIGym


cap = cv2.VideoCapture("20180615_World Record Bench Press with 142.5 kg by Jennifer Thompson USA in 63 kg class.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

gym = AIGym(
    model="yolo11n-pose.pt",
    show=True,
    kpts=[5, 7, 9],
    exercise="bench"
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = gym.monitor(im0)

cv2.destroyAllWindows()