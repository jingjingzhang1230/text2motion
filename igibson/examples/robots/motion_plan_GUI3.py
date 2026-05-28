import argparse
import logging
import os
import sys
from sys import platform
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import webbrowser
import textwrap

import numpy as np
import yaml

import igibson
from igibson import object_states
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.constants import ViewerMode
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper

from igibson.objects.visual_marker import VisualMarker
import pybullet as p

from openai import OpenAI
import base64
import re

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# --- GLOBAL EVENTS & SHARED DATA ---
stop_event = threading.Event()
start_planning_event = threading.Event()  
user_instruction_shared = ""              

# --- HELPER FUNCTIONS ---
def get_color(i):
    if i < 0 or i > 4: return [1, 1, 1, 0.5]
    colors = [[0, 0, 1, 0.5], [0, 1, 0, 0.5], [1, 0, 0, 0.5], [0, 0, 0, 0.5], [1, 1, 1, 0.5]]
    return colors[i]

def encode_image(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"

def get_vlm_response(client, user_instruction, image_input_path):
    text_prompt = "This picture shows a robot in a simulated home. Each trajectory of dots is a sequence of waypoints that guide the robot to move along. Please rate each of the trajectory out of 100 how well they match the description of user instruction by analyzing if they space relation between the trajectory and objects mentioned, such as left or right, far or close. User instruction: "+ user_instruction + " Give the score of how much the trajectory matches the user instruction out of 100 in the end, with the score inside brackets[]. "
    try:
        image_data_url = encode_image(image_input_path)
        response = client.chat.completions.create(
            # model="qwen2.5-vl-72b-instruct",  # Aliyun
            model= "Qwen/Qwen3-VL-8B-Instruct",#"Qwen/Qwen2.5-VL-7B-Instruct", # ModleScope Model-Id  # modelScope
            messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_data_url}}, {"type": "text", "text": text_prompt}]}],
            stream=False
        )
        match = re.search(r"\[(\d+)\]", response.choices[0].message.content)
        print( response.choices[0].message.content )
        return float(match.group(1)) if match else 0.0
    except Exception as e:
        print(f"VLM Error: {e}")
        return 0.0

def check_path_dist(path1, path2):
    length = min(len(path1), len(path2))
    max_dist = 0
    for i in range(length):
        max_dist = max(np.linalg.norm(path2[i][:2] - path1[i][:2]), max_dist)
    return max_dist < 0.20

def check_path_dist_with_multiple_paths(current_path, cluster_paths):
    for stored_path in cluster_paths:
        if check_path_dist(current_path, stored_path): return True, 0
    return False, -1

# --- SEPARATED SCREENSHOT LOGIC ---
def take_screenshot_only(path, query_number, env, motion_planner):
    """Helper to take a screenshot without querying the VLM."""
    env.simulator.viewer.customized_record4vlm = True
    motion_planner.sample_dry_run_base_plan(path)
    image_path = env.simulator.viewer.video_folder + "/{:05d}.png".format(query_number)
    return image_path

def take_screenshot_and_query_VLM(path, all_scores, iter_scores, query_number, env, motion_planner, client, query_repeat_num, marker_idx, markers, user_instruct):
    # Capture the image using our new helper
    image_path = take_screenshot_only(path, query_number, env, motion_planner)
    
    # move markers away
    for k in range(marker_idx): markers[k].set_position([0, 0, 3.0])
    
    vlm_scores = []
    print(f"Querying VLM for instruction: '{user_instruct}'...")
    for j in range(query_repeat_num):
        vlm_scores.append(get_vlm_response(client, user_instruct, image_path))
    
    vlm_scores = np.array(vlm_scores, dtype=float)
    average_score = np.nanmean(vlm_scores)
    iter_scores.append(vlm_scores)
    all_scores.append(average_score)
    print(f"VLM scores: {vlm_scores}")
    print(f"Average VLM score over {query_repeat_num} queries: {average_score}")
    return all_scores, iter_scores, average_score, image_path

# --- SIMULATION WORKER ---
def run_simulation_logic(config_data, gui_update_callback, gui_status_callback, gui_initial_image_callback):
    global user_instruction_shared
    try:
        gui_status_callback("Loading Environment...")
        
        config_data["optimized_renderer"] = False
        if platform == "darwin": config_data["texture_scale"] = 0.5
        config_data["hide_robot"] = False
        
        env = iGibsonEnv(config_file=config_data, mode="gui_interactive", action_timestep=1.0 / 120.0, physics_timestep=1.0 / 120.0)
        motion_planner = MotionPlanningWrapper(env, optimize_iter=2, full_observability_2d_planning=True, collision_with_pb_2d_planning=False, visualize_2d_planning=False, visualize_2d_result=False)
        env.reset()

        for obj in env.scene.get_objects():
            if obj.category == "bottom_cabinet": obj.states[object_states.Open].set_value(True)

        client = OpenAI(
        api_key="ms-7989bfe4-c06f-4eb8-a81d-50ad7d5b49c8", # use ModelScope SDK Token
        base_url="https://api-inference.modelscope.cn/v1"

        # api_key="sk-d9383d00eaed46a9b42e4c7995e27c68", # use AliYun SDK Token
        # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )


        env.simulator.viewer.initial_pos = config_data['viewer_initial_pos']
        env.simulator.viewer.initial_view_direction = config_data['viewer_initial_view_direction']
        env.simulator.viewer.reset_viewer()
        
        env.land(env.robots[0], config_data['start_position'], config_data['start_rotation'])
        env.robots[0].tuck()
        env.simulator.sync()

        # --- MODIFIED: TAKE INITIAL SCREENSHOT BEFORE WAITING ---
        gui_status_callback("Capturing initial view...")
        try:
            final_goal = config_data['final_goal'] 
            # visualize goal with marker
            goal_marker = VisualMarker(visual_shape=p.GEOM_SPHERE, radius=0.2, rgba_color=get_color(1))
            env.simulator.import_object(goal_marker)
            goal_marker.set_position(final_goal)
            print(f"Final goal set at: {final_goal}")
            # Create a fake path that stays exactly at the start position
            dummy_path = [config_data['start_position']] * 5 
            # Use query_number 99999 so it doesn't overlap with real iteration images (00000.png, 00001.png...)
            initial_img_path = take_screenshot_only(dummy_path, 0, env, motion_planner)
            gui_initial_image_callback(initial_img_path)
        except Exception as e:
            print(f"Error capturing initial screenshot: {e}")

        gui_status_callback("Ready. Waiting for user instruction...")

        

        start_planning_event.wait() 
        
        if stop_event.is_set():
            env.close()
            return
            
        gui_status_callback("Running Simulation...")
        user_instruction = user_instruction_shared

        
        success_attempts = 0
        iter_scores = []
        all_scores = []
        prev_paths = []
        best_global_score = -1


    
        while success_attempts < 20 and not stop_event.is_set():
            env.land(env.robots[0], config_data['start_position'], config_data['start_rotation'])
            path = motion_planner.plan_base_motion(final_goal)

            if path is None:
                continue

            marker_idx = 0
            markers = []
            marker_color = get_color(0)
            
            for j in range(0, len(path), 5):
                way_point = path[j]
                markers.append(VisualMarker(visual_shape=p.GEOM_SPHERE, radius=0.06, rgba_color=marker_color))
                env.simulator.import_object(markers[marker_idx])
                markers[marker_idx].set_position([way_point[0], way_point[1], 0])
                marker_idx += 1
            env.simulator.sync()

            path_array = np.array(path)
            
            if len(prev_paths) == 0:
                prev_paths.append([path_array])
                for k in range(marker_idx): markers[k].set_position([0,0,3])
                continue

            flag_mergeable, index_cluster = motion_planner.check_path_mergeable_with_multiple_paths(final_goal, path_array, prev_paths)
            
            process_path = False
            if flag_mergeable:
                flag_small_dist, _ = check_path_dist_with_multiple_paths(path_array, prev_paths[index_cluster])
                if not flag_small_dist:
                    process_path = True
                else:
                    for k in range(marker_idx): markers[k].set_position([0,0,3])
            else:
                process_path = True
                prev_paths.append([path_array])

            if process_path:
                all_scores, iter_scores, current_score, img_path = take_screenshot_and_query_VLM(
                    path, all_scores, iter_scores, success_attempts+1, env, motion_planner, 
                    client, 5, marker_idx, markers, user_instruction
                )
                
                success_attempts += 1
                
                is_new_best = False
                if current_score > best_global_score:
                    best_global_score = current_score
                    is_new_best = True
                    
                gui_update_callback(success_attempts, best_global_score, img_path if is_new_best else None)
            
        gui_status_callback("Stopped / Finished")
        env.close()
        
    except Exception as e:
        print(f"Simulation crashed: {e}")
        import traceback
        traceback.print_exc()

# --- GUI CLASS ---
class MotionPlanningGUI:
    def __init__(self, root, config_filename):
        self.root = root
        self.root.title("Anytime Motion Planning Interface")
        self.root.geometry("1200x700") 
        
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 12))
        style.configure("TLabel", font=("Helvetica", 11))

        # --- Input & Controls ---
        top_frame = ttk.Frame(root, padding="10")
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="User Instruction:").pack(anchor=tk.W)
        self.instruct_entry = ttk.Entry(top_frame, width=80)
        self.instruct_entry.pack(fill=tk.X, pady=5)
        self.instruct_entry.insert(0, "Move to the window from the right hand side of the room as much as possible.")

        btn_frame = ttk.Frame(top_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        self.start_btn = ttk.Button(btn_frame, text="Start Planning", command=self.start_planning)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="Stop Generating", command=self.stop_generation, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_lbl = ttk.Label(btn_frame, text="Status: Starting UI...", foreground="blue")
        self.status_lbl.pack(side=tk.LEFT, padx=20)

        # It starts with empty text ("") so it is invisible at first
        self.survey_lbl = tk.Label(btn_frame, text="", fg="blue", cursor="hand2", font=("Helvetica", 11, "underline"))
        self.survey_lbl.pack(side=tk.LEFT, padx=20)
        
        # Bind a mouse click (<Button-1>) to open the URL in the default web browser
        self.survey_url = "https://qualtrics.kcl.ac.uk/jfe/preview/previewId/52258505-ed68-4d3e-8a9b-06f275efc3c7/SV_cU4VdJ9fa242Ix8?Q_CHL=preview&Q_SurveyVersionID=current"  # Replace with your actual link
        self.survey_lbl.bind("<Button-1>", lambda e: webbrowser.open_new(self.survey_url))

        # --- Main Content Area ---
        content_frame = ttk.Frame(root, padding="10")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # LEFT SIDE: Image
        img_frame = ttk.Frame(content_frame)
        img_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        ttk.Label(img_frame, text="Current / Best path view:", font=("Helvetica", 14, "bold")).pack(pady=5)
        self.score_lbl = ttk.Label(img_frame, text="Waiting for environment...")
        self.score_lbl.pack()
        
        self.image_label = ttk.Label(img_frame)
        self.image_label.pack(pady=10)

        # RIGHT SIDE: Plot
        plot_frame = ttk.Frame(content_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        ttk.Label(plot_frame, text="Max Score over Iterations", font=("Helvetica", 14, "bold")).pack(pady=5)
        
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Max Score")
        self.ax.set_ylim([0, 100])
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.iterations = []
        self.max_scores = []

        # Start thread with the new update_initial_image callback
        try:
            config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
            t = threading.Thread(
                target=run_simulation_logic, 
                args=(config_data, self.update_gui_data, self.update_status, self.update_initial_image)
            )
            t.daemon = True
            t.start()
        except Exception as e:
            messagebox.showerror("Error", f"Could not load config: {e}")

    def start_planning(self):
        global user_instruction_shared
        instruction = self.instruct_entry.get()
        if not instruction.strip(): return
        user_instruction_shared = instruction
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        start_planning_event.set()


    def stop_generation(self):
        stop_event.set()
        start_planning_event.set() 
        self.status_lbl.config(text="Status: Stopping... (Wait for current cycle)")
        self.stop_btn.config(state=tk.DISABLED)
        
        # --- Show the survey link text ---
        self.survey_lbl.config(text="Thank you! Please click here to take our survey.")
        # ---------------------------------------------

    def update_status(self, text):
        self.root.after(0, lambda: self.status_lbl.config(text=f"Status: {text}"))

    # --- MODIFIED: Refactored image updating to be reusable ---
    def update_initial_image(self, image_path):
        raw_text = (
            "This is a home environment where the robot has to move to the goal position. "
            "The picture below shows the current location of the robot. The goal position is "
            "circled in the picture. Please write your preference on how the robot should "
            "navigate in the room towards the goal in the box above."
        )
        
        # --- Break the text automatically after every 80 characters ---
        instruction_text = textwrap.fill(raw_text, width=80)
        
        # Schedule the UI updates on the main thread
        self.root.after(0, lambda: self._update_image_display(image_path, instruction_text))
        # self.root.after(0, lambda: self._update_image_display(image_path, "Score: N/A (Initial State - Ready to Plan)"))

    def update_gui_data(self, iteration, max_score, image_path):
        self.root.after(0, lambda: self._update_on_main_thread(iteration, max_score, image_path))

    def _update_on_main_thread(self, iteration, max_score, image_path):
        # Update Plot
        self.iterations.append(iteration)
        self.max_scores.append(max_score)
        
        self.ax.clear()
        self.ax.plot(self.iterations, self.max_scores, marker='o', linestyle='-', color='b')
        self.ax.set_title("Max VLM Score vs Iteration")
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Max Score")
        self.ax.set_ylim([0, 100])
        self.ax.grid(True)
        self.canvas.draw()

        # Update Image (if new best)
        if image_path:
            self._update_image_display(image_path, f"Score: {max_score:.2f}")

    def _update_image_display(self, image_path, score_text):
        """Helper function to load and display an image reliably on the GUI."""
        if not image_path: return
        try:
            pil_img = Image.open(image_path)
            base_width = 500
            w_percent = (base_width / float(pil_img.size[0]))
            h_size = int((float(pil_img.size[1]) * float(w_percent)))
            pil_img = pil_img.resize((base_width, h_size), Image.Resampling.LANCZOS)
            
            tk_img = ImageTk.PhotoImage(pil_img)
            self.image_label.config(image=tk_img)
            self.image_label.image = tk_img 
            self.score_lbl.config(text=score_text)
        except Exception as e:
            print(f"Error updating image display: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default=os.path.join(igibson.configs_path, "my_fetch_motion_planning.yaml"))
    args = parser.parse_args()

    root = tk.Tk()
    app = MotionPlanningGUI(root, args.config)
    
    def on_closing():
        stop_event.set()
        start_planning_event.set()
        root.destroy()
        sys.exit(0)
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()