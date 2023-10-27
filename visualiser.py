import os
import argparse
import tkinter as tk

import cv2
import numpy as np
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import matplotlib.pyplot as plt

from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk) 

from Logreader import LogReader, log_bag_dir

class Visualiser:
    def __init__(self,exp,image_topic,w=None,h=None,display_ids=True):
        self.display_ids = display_ids
        self.image_topic = image_topic
        self.bridge = CvBridge()

        self.root = tk.Tk()

        # Read bag
        self.init_bag(exp)

        # Setup GUI
        buffer_w = 10
        buffer_h = 100
        if w is None:
            w = self.img_width + buffer_w
        if h is None:
            h = self.img_height + buffer_h
        graph_width = 500
    
        self.header_canvas = tk.Canvas(self.root,width=w, height=50)
        self.header_canvas.grid(row=0,column=0)
        self.video_canvas = tk.Canvas(self.root, width=w, height=h)
        self.video_canvas.grid(row=1,column=0)
        self.graph_canvas = tk.Canvas(self.root,width=graph_width,height=h)
        self.graph_canvas.grid(row=1,column=1)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        # Header GUI
        self.create_header(exp)

        # Graph
        self.create_graph()

        # Video player
        self.update_frame(0)
        self.create_video_slider()

    def close(self):
        self.root.quit()
        self.root.destroy()
        quit()

    def clear_canvases(self):
        self.video_canvas.pack_forget()
        self.graph_canvas.pack_forget()
        self.figure_canvas.get_tk_widget().destroy()
        self.graph_dropdown.destroy()

        plt.close(self.graph_fig)

    def init_bag(self,exp):
        self.exp = exp
        self.lr = LogReader(exp)
        self.bag = rosbag.Bag(self.lr.bagfile, "r")
        self.load_video()
        self.root.title(exp)

    def create_header(self,exp):
        # Create list
        onlyfiles = [f for f in os.listdir(log_bag_dir) if os.path.isfile(os.path.join(log_bag_dir, f))]
        logbags = sorted([x[:-4] for x in onlyfiles if x.endswith(".bag")])
        print("{} logbags available".format(len(logbags)))
        if exp not in logbags:
            error_message = "{} is not a valid rosbag file detected in {}".format(exp,log_bag_dir)
            raise Exception(error_message)
        # Create dropdown
        logbag_choice = tk.StringVar(self.header_canvas,name="logbag_choice")
        logbag_choice.set(exp)
        logbag_choice.trace("w", self.update_logbag)
        self.logbag_dropdown = tk.OptionMenu(self.header_canvas, logbag_choice, *logbags)
        self.logbag_dropdown.pack()

    def update_logbag(self,n,m,x):
        exp = self.root.getvar(n)
        self.init_bag(exp)
        self.clear_canvases()
        self.create_graph()
        self.update_frame(0)
        self.create_video_slider()

    def create_video_slider(self):
        self.playing = False
        self.curr_frame = 0
        button_width = 50
        self.play_button = tk.Button(self.video_canvas,text="Play",command=self.play_pause)
        self.play_button.place(x=0, y=self.img_height,width=button_width,height=50)

        frame_index = tk.IntVar()
        self.img_scale = tk.Scale(self.video_canvas, from_=0, to=self.num_frames-1, orient=tk.HORIZONTAL, variable=frame_index, length=self.img_width-button_width-1, command=self.update_frame)
        self.img_scale.place(x=button_width+1, y=self.img_height)

    def update_frame(self,frame_index):
        self.curr_frame = int(frame_index)
        cv_img = self.bridge.imgmsg_to_cv2(self.msgs[self.curr_frame], desired_encoding="passthrough")
        cv_img = self.process_image(cv_img)
        photo = self.photo_image(cv_img)
        self.video_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.video_canvas.image = photo

        # Update graph
        curr_time = self.times[self.curr_frame].secs + self.times[self.curr_frame].nsecs/1000000000
        for ax in self.graph_axes:
            ax.get_lines().pop().remove()
            ax.axvline(x=curr_time,color='r')
        self.figure_canvas.draw()

    def process_image(self,img):
        if self.display_ids:
            label_positions = self.lr.get_label_positions(self.times[self.curr_frame],self.stamps[self.curr_frame])
            for id in label_positions:
                if label_positions[id] is not None:
                    org = (int(img.shape[1]*label_positions[id][0]),int(img.shape[0]*label_positions[id][1]))
                    colour = list(self.lr.bodies[id].colour_rgb)
                    col = colour.copy()
                    col[0] = colour[2]*255
                    col[1] = colour[1]*255
                    col[2] = colour[0]*255
                    
                    img = cv2.putText(img, id, org, cv2.FONT_HERSHEY_SIMPLEX ,  1, col, 2, cv2.LINE_AA) 

        return img

    def load_video(self):
        self.msgs = []
        self.times = []
        self.stamps = []
        for _,msg,t in self.bag.read_messages(topics=[self.image_topic]):
            self.msgs.append(msg)
            self.times.append(t)
            self.stamps.append(msg.header.stamp) # Use stamp time as that is what is used to synch everything
        self.img_height = self.msgs[0].height
        self.img_width = self.msgs[0].width
        self.num_frames = len(self.times)

    def play_pause(self):
        if self.playing:
            # Pause
            self.playing = False
            self.play_button.config(text="Play")
        else:
            # Play
            self.playing = True
            self.play_button.config(text="Pause")
            if self.curr_frame == self.num_frames:
                # Restart video if at the end
                self.curr_frame = 0
            self.update()


    def update(self):
        if self.playing:
            if self.curr_frame < self.num_frames:
                self.update_frame(self.curr_frame)
                self.img_scale.set(self.curr_frame)

                self.curr_frame += 1
                self.root.after(1, self.update)
            else:
                # Toggle back to pause after video ends
                self.play_pause()

    def photo_image(self,img):
        h, w = img.shape[:2]
        data = f'P6 {w} {h} 255 '.encode() + img[..., ::-1].tobytes()
        return tk.PhotoImage(width=w, height=h, data=data, format='PPM')
    
    def create_graph(self,default_graph="engagement_value_with_robot"):
        # Get dropdown list
        graph_list = self.lr.log_columns
        # Prune skeleton, pose and header columns
        self.graph_list = sorted([ x for x in graph_list if ("pose_" not in x and "skeleton_" not in x and "header" not in x and x!="Time" and x!="ids" and x!="group_id")])
        self.graph_list += ["Special: groups","Special: engagement comparisons"]

        self.graph_fig,self.graph_axes = self.lr.plot_comparison(self.exp,default_graph,show=False,save_csvs=False,save_fig=False)
        curr_time = self.times[0].secs + self.times[0].nsecs/1000000000
        self.graph_axes[0].axvline(x=curr_time)

        
        self.figure_canvas = FigureCanvasTkAgg(self.graph_fig, master=self.graph_canvas)

        # Create dropdown widget
        graph_choice = tk.StringVar(self.graph_canvas,name="graph_choice")
        graph_choice.set(default_graph)
        graph_choice.trace("w", self.update_graph)
        self.graph_dropdown = tk.OptionMenu(self.graph_canvas, graph_choice, *self.graph_list)
        self.graph_dropdown.pack()

        self.figure_canvas.draw()
        self.figure_canvas.get_tk_widget().pack()


        

    def update_graph(self,n,m,x):
        
        graph_choice = self.root.getvar(n)
        if graph_choice == "Special: groups":
            # Groups
            self.graph_fig,self.graph_axes = self.lr.plot_groups(self.exp,show=False,save_fig=False)
        elif graph_choice == "Special: engagement comparisons":
            self.graph_fig,self.graph_axes = self.lr.plot_enagegement_comparisons(self.exp,show=False,save_fig=False)
        else:
            # Ordinary column-based graph
            self.graph_fig,self.graph_axes = self.lr.plot_comparison(self.exp,graph_choice,show=False,save_csvs=False,save_fig=False)
        self.figure_canvas.get_tk_widget().destroy()
        self.figure_canvas = FigureCanvasTkAgg(self.graph_fig, master=self.graph_canvas)
        curr_time = self.times[self.curr_frame].secs + self.times[self.curr_frame].nsecs/1000000000
        for ax in self.graph_axes:
            ax.axvline(x=curr_time,color='r')
        self.figure_canvas.draw()
        self.figure_canvas.get_tk_widget().pack(expand=True)
        

def main():
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("--exp", help="Name of rosbag (without .bag)", default="FrontTurn360_0")
    parser.add_argument("--image_topic", help="Image topic.", default="/opendr/image_pose_annotated")

    args = parser.parse_args()

    vis = Visualiser(args.exp,args.image_topic)
    #vis.update()
    vis.root.mainloop()

if __name__ == '__main__':
    main()