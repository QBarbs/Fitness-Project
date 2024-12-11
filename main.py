import cv2

import qb_solutions

from qb_ai_gym import AIGym
# Use line 9 if you want to use a video or use line 10 to use your device's camera for form tracking
# I recommend mp4 as it was the only one I used for this project, but
# other file formats might be fine
# cap = cv2.VideoCapture("<Insert video file path>) 
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


vid_stride = 24
frame_count = 0

# Uncomment lines 19-20, lines 48-49, and line 54 if you want to save a video with CV annotations
#From ultralytics
# video_writer = cv2.VideoWriter("workouttest.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
#End

gym = AIGym(
    model="yolo11n-pose.pt",
    show=True,
    kpts=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    kpts_angle=[11, 13, 15],
    exercise="squat",
    up_angle=170.0,
    down_angle=95.0
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # if frame_count % vid_stride == 0:
    #     im0 = gym.monitor_squat(im0)


    im0 = gym.monitor_squat(im0)
    # frame_count += 1
    # if frame_count == (24 * 5):


    # From Ultalytics
    # video_writer.write(im0)
    # End


cv2.destroyAllWindows()
# From Ultralytics
# video_writer.release()
# End
