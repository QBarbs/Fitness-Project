# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.solutions.solutions import BaseSolution  # Import a parent class
from ultralytics.utils.plotting import Annotator
import body_mappings as body
import numpy as np


class AIGym(BaseSolution):
    """A class to manage the gym steps of people in a real-time video stream based on their poses."""

    def __init__(self, **kwargs):
        """Initialization function for AiGYM class, a child class of BaseSolution class, can be used for workouts
        monitoring.
        """
        # Check if the model name ends with '-pose'
        if "model" in kwargs and "-pose" not in kwargs["model"]:
            kwargs["model"] = "yolo11n-pose.pt"
        elif "model" not in kwargs:
            kwargs["model"] = "yolo11n-pose.pt"

        super().__init__(**kwargs)
        self.count = []  # List for counts, necessary where there are multiple objects in frame
        self.angle = []  # List for angle, necessary where there are multiple objects in frame
        self.left_angle = []
        self.right_angle = []
        self.stage = []  # List for stage, necessary where there are multiple objects in frame

        # Extract details from CFG single time for usage later
        self.initial_stage = None
        self.up_angle = float(self.CFG["up_angle"])  # Pose up predefined angle to consider up pose
        self.down_angle = float(self.CFG["down_angle"])  # Pose down predefined angle to consider down pose
        self.kpts = self.CFG["kpts"]  # User selected kpts of workouts storage for further usage
        self.kpts_angle = self.CFG["kpts_angle"]
        self.lw = self.CFG["line_width"]  # Store line_width for usage
        self.exercise = "squat" 
        self.feedback = ""
        self.prev_feedback = ""
        self.squat_stance = "" # Not used but was considering having functionality for adjusting the qb_ai_gym class to track based on user's preferred squat stance
        
    
    def monitor_squat(self, im0):

        """
        Monitor the workouts using Ultralytics YOLOv8 Pose Model: https://docs.ultralytics.com/tasks/pose/.

        Args:
            im0 (ndarray): The input image that will be used for processing
        Returns
            im0 (ndarray): The processed image for more usage
        """
        # Extract tracks
        tracks = self.model.track(source=im0, persist=True, classes=self.CFG["classes"])[0]


        self.up_angle = 170.0
        self.down_angle = 95.0
        
        if tracks.boxes.id is not None:
            # Extract and check keypoints
            if len(tracks) > len(self.count):
                new_human = len(tracks) - len(self.count)
                # self.angle += [0] * new_human
                self.left_angle += [0] * new_human
                self.right_angle += [0] * new_human
                self.count += [0] * new_human
                self.stage += ["-"] * new_human

            # Initialize annotator
            self.annotator = Annotator(im0, line_width=self.lw)

            # Enumerate over keypoints
            for ind, k in enumerate(reversed(tracks.keypoints.data)):
                # Get keypoints and estimate the angle
                kpts = [k[int(self.kpts[i])].cpu() for i in range(len(self.kpts))]
                left_kpts_angle = [k[int(self.kpts[body.left_hip])].cpu(), k[int(self.kpts[body.left_knee])].cpu(), k[int(self.kpts[body.left_leg])].cpu()]
                right_kpts_angle = [k[int(self.kpts[body.right_hip])].cpu(), k[int(self.kpts[body.right_knee])].cpu(), k[int(self.kpts[body.right_leg])].cpu()]

                self.left_angle[ind] = self.annotator.estimate_pose_angle(*left_kpts_angle)
                self.right_angle[ind] = self.annotator.estimate_pose_angle(*right_kpts_angle)

                im0 = self.annotator.draw_specific_points(k, [11, 12, 13, 14, 15, 16], radius=self.lw * 3)
                # Determine stage and count logic based on angle thresholds
                if self.left_angle[ind] < self.down_angle and self.right_angle[ind] < self.down_angle:
                    self.feedback = self.check_squat_form(im0, k=kpts, phase="down")
                    if self.stage[ind] == "Going down":
                        self.stage[ind] = "Down"
                elif self.left_angle[ind] > self.up_angle and self.right_angle[ind] > self.up_angle:
                    if self.stage[ind] == "Going up":
                        self.count[ind] += 1
                    self.stage[ind] = "Up"

                    self.feedback = self.check_squat_form(im0, k=kpts, phase="up")
                elif self.down_angle <= self.left_angle[ind] <= self.up_angle and self.down_angle <= self.right_angle[ind] <= self.up_angle:
                
                    if self.stage[ind] == "Up":
                        self.stage[ind] = "Going down"
                    elif self.stage[ind] == "Down":
                        self.stage[ind] = "Going up"

                # Display angle, count, stage text, and feedback
                self.annotator.plot_angle_and_count_and_stage(
                    angle_text=self.left_angle[ind],  # angle text for display
                    count_text=self.count[ind],  # count text for workouts
                    stage_text=self.stage[ind],  # stage position text
                    center_kpt=k[int(self.kpts[body.left_knee])],  # center keypoint for display
                )

                # if self.feedback != self.prev_feedback:
                # self.annotator.plot_workout_information(self.feedback, position=(self.kpts[0], self.kpts[1]))
                self.annotator.plot_workout_information(self.feedback, position=(0, 60))
                
                
        self.display_output(im0)  # Display output image, if environment support display
        return im0  # return an image for writing or further usage
    
    def check_squat_form(self, im0, k, phase):
        self.feedback = ""
        tolerance = 169.0
        left_mid_x = (k[body.left_shoulder][0] + k[body.left_hip][0]) / 2
        left_mid_y = (k[body.left_shoulder][1] + k[body.left_hip][1]) / 2
        left_mid = (left_mid_x, left_mid_y)
        left_mid_kpts = [k[int(self.kpts[body.left_shoulder])].cpu(), left_mid, k[int(self.kpts[body.left_hip])].cpu()]
        left_mid_angle = self.annotator.estimate_pose_angle(*left_mid_kpts)

        right_mid_x = (k[body.right_shoulder][0] + k[body.right_hip][0]) / 2
        right_mid_y = (k[body.right_shoulder][1] + k[body.right_hip][1]) / 2
        right_mid = (right_mid_x, right_mid_y)
        right_mid_kpts = [k[int(self.kpts[body.right_shoulder])].cpu(), right_mid, k[int(self.kpts[body.right_hip])].cpu()]
        right_mid_angle = self.annotator.estimate_pose_angle(*right_mid_kpts)

        if left_mid_angle < tolerance or right_mid_angle < tolerance:
            self.feedback = self.feedback + "Align back to neutral position. "
        # Check user's form when in "up" part of squat
        if phase == "up":
            self.nose_kpt = [k[int(self.kpts[body.nose])].cpu().numpy()]
            self.eyes_kpts = np.array([k[int(self.kpts[body.left_eye])].cpu().numpy(), k[int(self.kpts[body.right_eye])].cpu().numpy()])
            self.hips_kpts = [k[int(self.kpts[body.left_hip])].cpu().numpy(), k[int(self.kpts[body.right_hip])].cpu().numpy()]
            self.shoulders_kpts = np.array([k[int(self.kpts[body.left_shoulder])].cpu().numpy(), k[int(self.kpts[body.right_shoulder])].cpu().numpy()])
            self.knee_kpts = np.array([k[int(self.kpts[body.left_knee])].cpu().numpy(), k[int(self.kpts[body.right_knee])].cpu().numpy()])
            tolerance = 50
            
            # Checks head alignment first but removed due to not working properly
            # if abs(self.eyes_kpts[0][0] - self.eyes_kpts[1][0]) > tolerance:
            #     self.feedback = self.feedback + "Align head to neutral position (Horizontally). "

            
            # if abs(self.eyes_kpts[0][1] - self.shoulders_kpts[0][1]) > tolerance or abs(self.eyes_kpts[1][1] - self.shoulders_kpts[1][1]) > tolerance:
            #     self.feedback = self.feedback + "Align head to neutral position (Vertically). "

            # Check hip alignment
            if abs(self.hips_kpts[0][0] - self.shoulders_kpts[0][0]) > tolerance or abs(self.hips_kpts[1][0] - self.shoulders_kpts[1][0]) > tolerance:
                self.feedback = self.feedback + "Align hips with shoulders. "

            
            if abs(self.knee_kpts[0][0]-self.hips_kpts[0][0]) > tolerance or abs(self.knee_kpts[1][0]-self.hips_kpts[1][0]) > tolerance:
                self.feedback = self.feedback + "Align knees with hips. "

        # CHeck user's form when in "down" part of squat
        else:
            self.nose_kpt = [k[int(self.kpts[body.nose])].cpu().numpy()]
            self.eyes_kpts = np.array([k[int(self.kpts[body.left_eye])].cpu().numpy(), k[int(self.kpts[body.right_eye])].cpu().numpy()])
            self.hips_kpts = [k[int(self.kpts[body.left_hip])].cpu().numpy(), k[int(self.kpts[body.right_hip])].cpu().numpy()]
            self.shoulders_kpts = np.array([k[int(self.kpts[body.left_shoulder])].cpu().numpy(), k[int(self.kpts[body.right_shoulder])].cpu().numpy()])
            self.knee_kpts = np.array([k[int(self.kpts[body.left_knee])].cpu().numpy(), k[int(self.kpts[body.right_knee])].cpu().numpy()])

            
            if abs(self.hips_kpts[0][1]-self.knee_kpts[0][1]) > tolerance or abs(self.hips_kpts[1][1]-self.knee_kpts[1][1]) > tolerance:
                self.feedback = self.feedback + "Squat at or below hip level. "

            if self.check_distance(self.shoulders_kpts[0][0], self.hips_kpts[0][0], 50, "<") is False or self.check_distance(self.shoulders_kpts[1][0], self.hips_kpts[1][0], 50, "<") is False:
                self.feedback = self.feedback + "Align shoulders and hips. Keep the spine neutral without excessive rounding. " 

        # If no feedback is given (no problems are present in user's form), the feedback will just
        # contain a string saying that the user's form has no current issues.
        if self.feedback == "":
            self.feedback = "No form issues currently present."
        return self.feedback    


    """""
    Checks the distance between two numbers, and requires a tolerance and sign (greater than, less than or equal to, etc.) to compare
    the two numbers.
    """""
    def check_distance(self, num_1, num_2, tolerance, sign):
        match sign:
            case "<":
                if abs(num_1-num_2) < tolerance:
                    return True
                else:
                    return False

            case ">":
                if abs(num_1-num_2) > tolerance:
                    return True
                else:
                    return False
                
            case "=":
                if abs(num_1-num_2) == tolerance:
                    return True
                else:
                    return False

            case "<=":
                if abs(num_1-num_2) <= tolerance:
                    return True
                else:
                    return False
                
            case ">=":
                if abs(num_1-num_2) <= tolerance:
                    return True
                else:
                    return False
            
            case _:
                return None


