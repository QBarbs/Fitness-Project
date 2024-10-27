# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.solutions.solutions import BaseSolution  # Import a parent class
from ultralytics.utils.plotting import Annotator
import body_mappings as body
import math
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
        self.stage = []  # List for stage, necessary where there are multiple objects in frame

        # Extract details from CFG single time for usage later
        self.initial_stage = None
        self.up_angle = float(self.CFG["up_angle"])  # Pose up predefined angle to consider up pose
        self.down_angle = float(self.CFG["down_angle"])  # Pose down predefined angle to consider down pose
        self.kpts = self.CFG["kpts"]  # User selected kpts of workouts storage for further usage
        self.lw = self.CFG["line_width"]  # Store line_width for usage
        self.exercise = "squat"  # squat, bench, deadlift

    def monitor(self, im0):
        """
        Monitor the workouts using Ultralytics YOLOv8 Pose Model: https://docs.ultralytics.com/tasks/pose/.

        Args:
            im0 (ndarray): The input image that will be used for processing
        Returns
            im0 (ndarray): The processed image for more usage
        """
        # Extract tracks
        tracks = self.model.track(source=im0, persist=True, classes=self.CFG["classes"])[0]

    

        # # Setup keypoints to be stored
        # squat = {{"primary_left": [left_hip, left_knee, left_leg]}, 
        #          {"primary_right": [right_hip, right_knee, right_leg]}
        #          }


        if tracks.boxes.id is not None:
            # Extract and check keypoints
            if len(tracks) > len(self.count):
                new_human = len(tracks) - len(self.count)
                self.angle += [0] * new_human
                self.count += [0] * new_human
                self.stage += ["-"] * new_human

            # Initialize annotator
            self.annotator = Annotator(im0, line_width=self.lw)

            # Enumerate over keypoints
            for ind, k in enumerate(reversed(tracks.keypoints.data)):
                # Get keypoints and estimate the angle
                kpts = [k[int(self.kpts[i])].cpu() for i in range(3)]
                self.angle[ind] = self.annotator.estimate_pose_angle(*kpts)
                im0 = self.annotator.draw_specific_points(k, self.kpts, radius=self.lw * 3)

                # Determine stage and count logic based on angle thresholds
                if self.angle[ind] < self.down_angle:
                    if self.stage[ind] == "up":
                        self.count[ind] += 1
                    self.stage[ind] = "down"
                elif self.angle[ind] > self.up_angle:
                    self.stage[ind] = "up"

                # Display angle, count, and stage text
                self.annotator.plot_angle_and_count_and_stage(
                    angle_text=self.angle[ind],  # angle text for display
                    count_text=self.count[ind],  # count text for workouts
                    stage_text=self.stage[ind],  # stage position text
                    center_kpt=k[int(self.kpts[1])],  # center keypoint for display
                )

        self.display_output(im0)  # Display output image, if environment support display
        return im0  # return an image for writing or further usage
    
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

        # # Setup keypoints to be stored
        # squat = {{"primary_left": [left_hip, left_knee, left_leg]}, 
        #          {"primary_right": [right_hip, right_knee, right_leg]}
        #          }
        self.up_angle = 170.0
        self.down_angle = 95.0
        if tracks.boxes.id is not None:
            # Extract and check keypoints
            if len(tracks) > len(self.count):
                new_human = len(tracks) - len(self.count)
                self.angle += [0] * new_human
                self.count += [0] * new_human
                self.stage += ["-"] * new_human

            # Initialize annotator
            self.annotator = Annotator(im0, line_width=self.lw)

            # Enumerate over keypoints
            for ind, k in enumerate(reversed(tracks.keypoints.data)):
                # Get keypoints and estimate the angle
                kpts = [k[int(self.kpts[i])].cpu() for i in range(3)]
                self.angle[ind] = self.annotator.estimate_pose_angle(*kpts)

                im0 = self.annotator.draw_specific_points(k, self.kpts, radius=self.lw * 3)
                # Determine stage and count logic based on angle thresholds
                if self.angle[ind] < self.down_angle:
                    self.feedback = self.check_squat_form(self, im0, k, "up")
                    if self.stage[ind] == "up":
                        self.count[ind] += 1
                    self.stage[ind] = "down"
                elif self.angle[ind] > self.up_angle:
                    self.stage[ind] = "up"

                # Display angle, count, and stage text
                self.annotator.plot_angle_and_count_and_stage(
                    angle_text=self.angle[ind],  # angle text for display
                    count_text=self.count[ind],  # count text for workouts
                    stage_text=self.stage[ind],  # stage position text
                    center_kpt=k[int(self.kpts[1])],  # center keypoint for display
                )

        self.display_output(im0)  # Display output image, if environment support display
        return im0  # return an image for writing or further usage
    
    def check_squat_form(self, im0, k, phase):
        self.feedback = ""
        self.left_torso_kpts = [k[int(self.kpts[body.left_shoulder, body.left_hip, body.left_knee])].cpu()]
        self.right_torso_kpts = [k[int(self.kpts[body.right_shoulder, body.right_hip, body.right_knee])].cpu()]
        self.left_torso_angle = self.annotator.estimate_pose_angle(*self.left_torso_kpts)
        self.right_torso_angle = self.annotator.estimate_pose_angle(*self.right_torso_angle)

        left_parallel_coordinates = [k[int(body.left_hip, body.left_knee, body.left_leg)].cpu().numpy()]
        right_parallel_coordinates = [k[int(body.right_hip, body.right_knee, body.right_leg)].cpu().numpy()]
        tolerance = 0
        # Check user's form when in "up" part of squat
        if phase == "up":
            self.nose_kpt = [k[int(self.kpts[body.nose])].cpu().numpy()]
            self.eyes_kpts = np.array([k[int(self.kpts[body.left_eye])].cpu().numpy(), k[int(self.kpts[body.right_eye])].cpu().numpy()])
            self.hips_kpts = [k[int(self.kpts[body.left_hip])].cpu().numpy(), k[int(self.kpts[body.right_hip])].cpu().numpy()]
            self.shoulders_kpts = np.array([k[int(self.kpts[body.left_shoulder])].cpu().numpy(), k[int(self.kpts[body.right_shoulder])].cpu().numpy()])
            self.knee_kpts = np.array([k[int(self.kpts[body.left_knee])].cpu().numpy(), k[int(self.kpts[body.right_knee])].cpu().numpy()])
            tolerance = 5
            # Checks head alignment first
            if abs(self.eyes_kpts[0][0] - self.eyes_kpts[1][0]) > tolerance:
                self.feedback = self.feedback + "Align head to neutral position (Horizontally)."
            #Plagarism?

            tolerance = 15
            if abs(self.eyes_kpts[0][1] - self.shoulders_kpts[0][1]) < tolerance or abs(self.eyes_kpts[1][1] - self.shoulders_kpts[1][1]):
                self.feedback = self.feedback + "Align head to neutral position (Vertically)."

            # Hip checking
            tolerance = 5
            if abs(self.hips_kpts[0][0] - self.shoulders_kpts[0][0]) > tolerance or abs(self.hips_kpts[1][0] - self.shoulders_kpts[1][0]) > tolerance:
                self.feedback = self.feedback + "Align hips with shoulders."

            tolerance = 5
            if abs(self.knee_kpts[0][0]-self.hips_kpts[0][0]) > tolerance or abs(self.knee_kpts[1][0]-self.hips_kpts[1][0]) > tolerance:
                self.feedback = self.feedback + "Align knees with hips."

        # CHeck user's form when in "down" part of squat
        else:
            self.nose_kpt = [k[int(self.kpts[body.nose])].cpu().numpy()]
            self.eyes_kpts = np.array([k[int(self.kpts[body.left_eye])].cpu().numpy(), k[int(self.kpts[body.right_eye])].cpu().numpy()])
            self.hips_kpts = [k[int(self.kpts[body.left_hip])].cpu().numpy(), k[int(self.kpts[body.right_hip])].cpu().numpy()]
            self.shoulders_kpts = np.array([k[int(self.kpts[body.left_shoulder])].cpu().numpy(), k[int(self.kpts[body.right_shoulder])].cpu().numpy()])
            self.knee_kpts = np.array([k[int(self.kpts[body.left_knee])].cpu().numpy(), k[int(self.kpts[body.right_knee])].cpu().numpy()])

            tolerance = 5
            if abs(self.hips_kpts[0][1]-self.knee_kpts[0][1]) > tolerance or abs(self.hips_kpts[1][1]-self.knee_kpts[1][1]) > tolerance:
                self.feedback = self.feedback + "Squat at or below hip level."

            if self.check_distance(self.shoulders_kpts[0][0], self.hips_kpts[0][0], 5, "<") is False or self.check_distance(self.shoulders_kpts[1][0], self.hips_kpts[1][0], 5, "<") is False:
                self.feedback = self.feedback + "Align shoulders and hips. Keep the spine neutral without excessive rounding." 

        # If no feedback is given (no problems are present in user's form), the feedback will just
        # contain a string saying that the user's form has no current issues.
        if self.feedback == "":
            self.feedback = "No form issues currently present."
        
        return self.feedback
            

    def monitor_bench(self, im0):
        """
        Monitor the workouts using Ultralytics YOLOv8 Pose Model: https://docs.ultralytics.com/tasks/pose/.

        Args:
            im0 (ndarray): The input image that will be used for processing
        Returns
            im0 (ndarray): The processed image for more usage
        """
        bench = {{"primary_left": [body.left_shoulder, body.left_elbow, body.left_wrist]}, 
                 {"primary_right": [body.right_shoulder, body.right_elbow, body.right_wrist]},
                 {"secondary": []}}

        # Extract tracks
        tracks = self.model.track(source=im0, persist=True, classes=self.CFG["classes"])[0]
        if tracks.boxes.id is not None:
            # Extract and check keypoints
            if len(tracks) > len(self.count):
                new_human = len(tracks) - len(self.count)
                self.angle += [0] * new_human
                self.count += [0] * new_human
                self.stage += ["-"] * new_human
            

            # Initialize annotator
            self.annotator = Annotator(im0, line_width=self.lw)

            # Enumerate over keypoints
            for ind, k in enumerate(reversed(tracks.keypoints.data)):
                # Get keypoints and estimate the angle
                kpts = [k[int(self.kpts[i])].cpu() for i in range(3)]
                self.angle[ind] = self.annotator.estimate_pose_angle(*kpts)
                im0 = self.annotator.draw_specific_points(k, self.kpts, radius=self.lw * 3)

                # Determine stage and count logic based on angle thresholds
                if self.angle[ind] < self.down_angle:
                    if self.stage[ind] == "up":
                        self.count[ind] += 1
                    self.stage[ind] = "down"
                elif self.angle[ind] > self.up_angle:
                    self.stage[ind] = "up"

                # Display angle, count, and stage text
                self.annotator.plot_angle_and_count_and_stage(
                    angle_text=self.angle[ind],  # angle text for display
                    count_text=self.count[ind],  # count text for workouts
                    stage_text=self.stage[ind],  # stage position text
                    center_kpt=k[int(self.kpts[1])],  # center keypoint for display
                )

        self.display_output(im0)  # Display output image, if environment support display
        return im0  # return an image for writing or further usage    
    

    # def check_form(angle):
    """""
    Checks the distance between two numbers, and requires a tolerance and sign (greater than, less than or equal to, etc.) to compare
    the two numbers.

    """""
    def check_distance(num_1, num_2, tolerance, sign):
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
            
            case other:
                return None


