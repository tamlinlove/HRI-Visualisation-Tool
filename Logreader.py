import pandas as pd
from bagpy import bagreader
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import os
from itertools import product
import seaborn as sns
import numpy as np
import argparse

log_bag_dir = "rosbags/"

joints = [
    'nose',
    'neck', 
    'r_sho',
    'r_elb',
    'r_wri',
    'l_sho',
    'l_elb',
    'l_wri',
    'r_hip',
    'r_knee',
    'r_ank',
    'l_hip',
    'l_knee',
    'l_ank',
    'r_eye',
    'l_eye',
    'r_ear',
    'l_ear'
    ]


CMAP = plt.get_cmap("tab10")

class BodyLog:
    def __init__(self,id,colour_index=0):
        self.id = id

        self.pose_df = None
        self.pose_df_expanded = None
        self.body_norm_df = None
        self.face_norm_df = None
        self.engagement_status_df = None
        self.velocity_df = None
        self.skeleton_df = None
        self.skeleton_df_expanded = None
        self.activity_df = None

        self.colour = CMAP(colour_index)
        self.colour_rgb = CMAP.colors[colour_index]

    def plot_engagement(self):
        edf = self.engagement_status_df.copy()
        # Convert to easier values for visualisation
        edf.level.replace([1,2,3,4], [-2,1,2,-1], inplace=True)
        edf.plot(x="Time",y="level")
        plt.show()

class LogReader:
    def __init__(self,name,csv_dir="csvs/",fig_dir=None):
        self.bagfile = log_bag_dir+name+".bag"
        self.bagreader = bagreader(self.bagfile)
        
        self.tracked_table = self.get_topic_df("/humans/bodies/tracked")
        self.engagement_table = None
        self.group_table = None
        self.bodies = {}
        self.colours = {}
        self.groups = []

        self.csv_dir = csv_dir
        if not os.path.exists(csv_dir):
            os.mkdir(csv_dir)

        if fig_dir is None:
            fig_dir = "figs/{}/".format(name)
        self.fig_dir = fig_dir
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)

        # Read in all the bodies
        all_topics = self.bagreader.topic_table["Topics"].to_list()
        colour_index = 0
        for topic in all_topics:
            t_split = topic.split("/")
            if t_split[1] == "humans" and t_split[2] == "bodies" and t_split[3] != "tracked":
                if t_split[3] not in self.bodies:
                    self.bodies[t_split[3]] = BodyLog(t_split[3],colour_index=colour_index)
                    self.colours[t_split[3]] = self.bodies[t_split[3]].colour
                    colour_index += 1
                if t_split[4] == "poses":
                    self.bodies[t_split[3]].pose_df = self.get_topic_df(topic)
                elif t_split[4] == "body_orientation":
                    self.bodies[t_split[3]].body_norm_df = self.get_topic_df(topic)
                elif t_split[4] == "face_orientation":
                    self.bodies[t_split[3]].face_norm_df = self.get_topic_df(topic)
                elif t_split[4] == "velocity":
                    self.bodies[t_split[3]].velocity_df = self.get_topic_df(topic)
                elif t_split[4] == "engagement_status":
                    self.bodies[t_split[3]].engagement_status_df = self.get_topic_df(topic)
                elif t_split[4] == "skeleton2d":
                    self.bodies[t_split[3]].skeleton_df = self.get_topic_df(topic)
                elif t_split[4] == "activity":
                    self.bodies[t_split[3]].activity_df = self.get_topic_df(topic)
                else:
                    print("Unrecognised topic: {}".format(topic))
            elif t_split[1] == "humans" and t_split[2] == "interactions":
                if t_split[3] == "engagements":
                    self.engagement_table = self.get_topic_df(topic)
                elif t_split[3] == "groups":
                    self.group_table = self.get_topic_df(topic)
                    self.group_table = self.group_table.drop_duplicates(subset=["header.stamp.secs","header.stamp.nsecs","group_id","members"],keep="last")
                    self.groups = list(self.group_table["group_id"].unique())
                else:
                    print("Unrecognised topic: {}".format(topic))

        self.log_columns = self.get_log_columns()

    def get_log_columns(self):
        cols = []
        for id in self.bodies:
            log = self.compile_log(self.bodies[id])
            cols += list(log.columns)
        return list(set(cols))


    def get_topic_df(self,topic):
        data = self.bagreader.message_by_topic(topic)
        return pd.read_csv(data)
    
    def compile_log(self,body,time=None):
        index_cols = ["header.stamp.secs","header.stamp.nsecs"]
        track_cols = ["Time"]+index_cols+["ids"]
        log = self.tracked_table.copy()[track_cols]

        if time is not None:
            time_condition = (log["header.stamp.secs"]==time.secs) & (log["header.stamp.nsecs"] == time.nsecs)
            log = log.loc[time_condition]

        empty_cols_to_add = []

        # Add engagement status
        if body.engagement_status_df is not None:
            edf = body.engagement_status_df.copy()[index_cols+["level"]]
            log = log.merge(edf,on=index_cols,how="left")
            log = log.drop_duplicates(subset=index_cols,keep="last")
        else:
            empty_cols_to_add.append("level")
        #log["level"].fillna(0, inplace=True)

        # Add activity
        if body.activity_df is not None:
            adf = body.activity_df.copy()[index_cols+["activity"]]
            log = log.merge(adf,on=index_cols,how="left")
        else:
            empty_cols_to_add.append("activity")

        # Add pose
        if body.pose_df is not None:
            if body.pose_df_expanded is None:
                pdf = body.pose_df.copy()[index_cols+["poses"]]
                pdf,joint_cols = self.process_pose_df(pdf)
                pdf = pdf[index_cols+joint_cols]
                body.pose_df_expanded = pdf
            else:
                pdf = body.pose_df_expanded.copy()
            log = log.merge(pdf,on=index_cols,how="left")
        else:
            for joint in joints:
                for i in ["x","y","z"]:
                    joint_col_name = "pose_"+joint+"."+i
                    empty_cols_to_add.append(joint_col_name)

        # Add Skeleton
        if body.skeleton_df is not None:
            if body.skeleton_df_expanded is None:
                skdf = body.skeleton_df.copy()[index_cols+["skeleton"]]
                skdf, skeleton_cols = self.process_skeleton_df(skdf)
                skdf = skdf[index_cols+skeleton_cols]
                body.skeleton_df_expanded = skdf
            else:
                skdf = body.skeleton_df_expanded.copy()
            log = log.merge(skdf,on=index_cols,how="left")
        else:
            for joint in joints:
                for i in ["x","y","c"]:
                    joint_col_name = "skeleton_"+joint+"."+i
                    empty_cols_to_add.append(joint_col_name)

        vector_cols = ["vector.x","vector.y","vector.z"]
        # Add body norm
        if body.body_norm_df is not None:
            bndf = body.body_norm_df.copy()[index_cols+vector_cols]
            log = log.merge(bndf,on=index_cols,how="left")
            rename_mapping = {col: "bn_"+col for col in vector_cols}
            log.rename(columns=rename_mapping,inplace=True)
        else:
            for v_col in vector_cols:
                empty_cols_to_add.append("bn_"+v_col)

        # Add face norm
        if body.face_norm_df is not None:
            fndf = body.face_norm_df.copy()[index_cols+vector_cols]
            log = log.merge(fndf,on=index_cols,how="left")
            rename_mapping = {col: "fn_"+col for col in vector_cols}
            log.rename(columns=rename_mapping,inplace=True)
        else:
            for v_col in vector_cols:
                empty_cols_to_add.append("fn_"+v_col)

        # Add velocity
        vel_cols = ["twist.linear.x","twist.linear.y","twist.linear.z"]
        if body.velocity_df is not None:
            vdf = body.velocity_df.copy()[index_cols+vel_cols]
            log = log.merge(vdf,on=index_cols,how="left")
            rename_mapping = {col: "vel_"+col for col in vel_cols}
            log.rename(columns=rename_mapping,inplace=True)
            vel_mag = pd.Series(np.linalg.norm(log[list(rename_mapping.values())].copy().values,axis=1),name="vel_magnitude")
            log = pd.concat([log,vel_mag],axis=1)
        else:
            for v_col in vel_cols:
                empty_cols_to_add.append("vel_"+v_col)
            empty_cols_to_add.append("vel_magnitude")

        # Add engagement values
        # Start with robot engagement
        rob_condition = ((self.engagement_table['person_a'] == body.id) & (self.engagement_table['person_b'].isnull())) | ((self.engagement_table['person_b'] == body.id) & (self.engagement_table['person_a'].isnull()))
        evdf = self.engagement_table.loc[rob_condition][index_cols+["distance","mutual_gaze","engagement"]]
        evdf = evdf.drop_duplicates(subset=index_cols,keep="last")
        log = log.merge(evdf,on=index_cols,how="left")
        rename_mapping = {"distance":"distance_to_robot","mutual_gaze":"mutual_gaze_with_robot","engagement":"engagement_value_with_robot"}
        log.rename(columns=rename_mapping,inplace=True)

        # Human-Human engagements that involve this body
        ab_condition = ((self.engagement_table['person_a'] == body.id) & (self.engagement_table['person_b'].notnull())) | ((self.engagement_table['person_b'] == body.id) & (self.engagement_table['person_a'].notnull()))
        abevdf = self.engagement_table.loc[ab_condition][index_cols+["person_a","person_b","distance","mutual_gaze","engagement"]]
        abevdf = abevdf.drop_duplicates(subset=index_cols+["person_a","person_b"],keep="last")
        unique_ids = list(pd.unique(abevdf[['person_a', 'person_b']].values.ravel('K')))
        if len(unique_ids)>1:
            unique_ids.remove(body.id)
            eng_cols = ["distance","mutual_gaze","engagement"]
            for other_id in unique_ids:
                this_condition = ((self.engagement_table['person_a'] == body.id) & (self.engagement_table['person_b'] == other_id)) | ((self.engagement_table['person_b'] == body.id) & (self.engagement_table['person_a'] == other_id))
                obdf = abevdf.loc[this_condition][index_cols+eng_cols]
                log = log.merge(obdf,on=index_cols,how="left")
                rename_mapping = {col: other_id+"_"+col for col in eng_cols}
                log.rename(columns=rename_mapping,inplace=True)

        # Groups that include this body
        if self.group_table is not None:
            gdf = self.group_table.copy()
            gdf['members'] = gdf['members'].str.strip('[]').str.split(',')
            gdf = gdf.explode("members")
            gdf['members'] = gdf['members'].str.strip(" ").str.strip("''")
            gdf = gdf.loc[gdf['members'] == body.id][index_cols+["group_id"]]
            log = log.merge(gdf,on=index_cols,how="left")


        # Add empty columns where neccesary
        if len(empty_cols_to_add) != 0:
            log.reindex(columns=log.columns.tolist()+empty_cols_to_add, fill_value=None)

        return log
    
    def get_label_positions(self,time,stamp):
        label_pos = {}
        row = self.get_row_at_time(stamp)
        if not row.empty:
            curr_bodies = row["ids"].strip("[]").split(",")
            curr_bodies = [x.strip().strip("'") for x in curr_bodies]
            if curr_bodies != [""]:
                x_offset = 0.1
                y_offset = -0.2
                
                # Get 2D positions
                for id in curr_bodies:
                    skel = self.get_skeleton_at_time(stamp,id)
                    if skel is None or pd.isna(skel["skeleton_neck.x"]):
                        label_pos[id] = None
                    else:
                        x = float(skel["skeleton_neck.x"]) + x_offset
                        y = float(skel["skeleton_neck.y"]) + y_offset
                        label_pos[id] = (x,y)
        return label_pos
            

    def get_pose_at_time(self,time,id):
        
        if self.bodies[id].pose_df_expanded is None:
            time_condition = (self.bodies[id].pose_df["header.stamp.secs"]==time.secs) & (self.bodies[id].pose_df["header.stamp.nsecs"] == time.nsecs)
            pdf = self.bodies[id].pose_df.copy().loc[time_condition]
            if pdf.empty:
                return None
            pdf,_ = self.process_pose_df(pdf)
        else:
            pdf = self.bodies[id].pose_df_expanded.copy()
            time_condition = (pdf["header.stamp.secs"]==time.secs) & (pdf["header.stamp.nsecs"] == time.nsecs)
            pdf = pdf.loc[time_condition]
        return None if pdf.empty else pdf.iloc[0,:]
    
    def get_skeleton_at_time(self,time,id):
        approximation = 100000
        if self.bodies[id].skeleton_df is None:
            return None
        if self.bodies[id].skeleton_df_expanded is None:
            skdf = self.bodies[id].skeleton_df.copy()
            skdf["nsec_apprx"] = skdf["header.stamp.nsecs"]//approximation
            time_condition = (self.bodies[id].skeleton_df["header.stamp.secs"]==time.secs) & (self.bodies[id].skeleton_df["nsec_apprx"] == time.nsecs//approximation)
            skdf = skdf.loc[time_condition]
            if skdf.empty:
                return None
            skdf,_ = self.process_skeleton_df(skdf)
        else:
            skdf = self.bodies[id].skeleton_df_expanded.copy()
            skdf["nsec_apprx"] = skdf["header.stamp.nsecs"]//approximation
            time_condition = (skdf["header.stamp.secs"]==time.secs) & (skdf["nsec_apprx"] == time.nsecs//approximation)
            skdf = skdf.loc[time_condition]
        return None if skdf.empty else skdf.iloc[0,:]

    def get_row_at_time(self,time):
        time_condition = (self.tracked_table["header.stamp.secs"]==time.secs) & (self.tracked_table["header.stamp.nsecs"] == time.nsecs)
        time_df = self.tracked_table.copy().loc[time_condition]
        time_df = time_df.drop_duplicates(subset=["header.stamp.secs","header.stamp.nsecs"],keep="last")
        return pd.Series() if time_df.empty else time_df.iloc[0, :]
    
    def process_pose_df(self,pdf):
        pdf['poses'] = pdf['poses'].str.strip('[]').str.split(",")
        pdf[joints] = pd.DataFrame(pdf.poses.tolist(), index= pdf.index)
        joint_cols = []
        for joint in joints:
            for i,j in zip(["x","y","z"],[1,2,3]):
                joint_col_name = "pose_"+joint+"."+i
                joint_cols.append(joint_col_name)
                pdf[joint_col_name] = pdf[joint].str.strip().str.split("\n").str[j].str.split(":")
                pdf[["joint_helper",joint_col_name]] = pd.DataFrame(pdf[joint_col_name].tolist(), index= pdf.index)
                pdf[joint_col_name] = pdf[joint_col_name].str.strip().astype(float)
        pdf = pdf.drop_duplicates(subset=["header.stamp.secs","header.stamp.nsecs"],keep="last")
        return pdf,joint_cols
    
    def process_skeleton_df(self,skdf):
        skdf["skeleton"] = skdf["skeleton"].str.strip("[]").str.split(",")
        skdf[joints] = pd.DataFrame(skdf["skeleton"].tolist(), index= skdf.index)
        joint_vars = list(product(joints,[0,1,2]))
        skeleton_cols = []
        for i in range(len(joint_vars)):
            joint_col_name = "skeleton_"+joint_vars[i][0]+"."+"xyc"[joint_vars[i][1]]
            skeleton_cols.append(joint_col_name)
            skdf[joint_col_name] = skdf[joint_vars[i][0]].str.strip().str.split("\n").str[joint_vars[i][1]].str.split(":")
            skdf[["joint_helper",joint_col_name]] = pd.DataFrame(skdf[joint_col_name].tolist(), index= skdf.index)
            skdf[joint_col_name] = skdf[joint_col_name].str.strip().astype(float)
        skdf = skdf.drop_duplicates(subset=["header.stamp.secs","header.stamp.nsecs"],keep="last")
        return skdf,skeleton_cols

    
    def plot(self,log,label=None,x="Time",y="distance_to_robot",show=True,ax=None,set_xlim=True,colour=None,**kwargs):
        if label is None:
            label = y
        if colour is None:
            log.plot(x=x,y=y,label=label,ax=ax,**kwargs)
        else:
            log.plot(x=x,y=y,label=label,ax=ax,color=colour,**kwargs)
        if set_xlim:
            if x is not None:
                plt.xlim(log[x].astype(float).min(axis=0),log[x].astype(float).max(axis=0))
            else:
                xmin = min(log.index)
                xmax = max(log.index)
                plt.xlim(xmin,xmax)
        if show:
            plt.show()

    def plot_comparison(self,exp_name,y,x="Time",bodies=None,save_csvs=True,show=True,save_fig=True):
        fig, ax = plt.subplots()
                    
        ax.set_title(exp_name)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if bodies is None:
            bodies = self.bodies
        for id in bodies:
            log = self.compile_log(self.bodies[id])
            if y in log:
                self.plot(log,label=id,x=x,y=y,ax=ax,show=False,colour=self.bodies[id].colour)
                if save_csvs:
                    log.to_csv(self.csv_dir+exp_name+"_"+id+".csv")
            else:
                print("No column {} for body {}".format(y,id))
        if show:
            plt.show()

        if save_fig:
            plt.savefig(self.fig_dir+x+"__"+y+".png")

        return fig,[ax]

    def plot_groups(self,exp_name,x="Time",show=True,save_fig=True):
        bodies = list(self.bodies.keys())+["ROBOT"]
        palette = sns.color_palette(None, len(self.groups))

        fig, ax = plt.subplots()
        ax.set_title(exp_name)
      
        if self.group_table is not None:
            group_df = self.group_table.copy()[["Time","header.stamp.secs","header.stamp.nsecs","group_id","members"]]
            for group in self.groups:
                this_gdf = group_df.loc[group_df["group_id"]==group].copy()
                this_gdf['members'] = this_gdf['members'].str.strip('[]').str.split(',')
                this_gdf = this_gdf.explode("members")
                this_gdf['members'] = this_gdf['members'].str.strip(" ").str.strip("''")
                for bid in bodies:
                    body_df = this_gdf.copy().loc[this_gdf["members"]==bid]
                    body_df["b_index"] = bodies.index(bid)
                    body_df.plot(x=x,y="b_index",ax=ax,color=palette[self.groups.index(group)],linewidth=7)

        
        legend = ax.get_legend()
        if legend is not None:
            ax.get_legend().remove()
        ax.set_yticks(list(range(len(bodies))),bodies,rotation="horizontal")
        ax.set_xlabel(x)
        ax.set_ylabel("Body")

        plt.xlim(self.tracked_table[x].astype(float).min(axis=0),self.tracked_table[x].astype(float).max(axis=0))
        if show:
            plt.show()

        if save_fig:
            plt.savefig(self.fig_dir+x+"__Groups.png")

        return fig,[ax]

    def plot_enagegement_comparisons(self,exp_name,x="Time",include_robot=True,show=True,save_fig=True):
        bodies = list(self.bodies.keys())
        if include_robot:
            bodies.append("ROBOT")

        scale = 6
        fig_height = scale
        fig_width = scale
        fig, axs = plt.subplots(3,1,figsize=(fig_width,fig_height))
        metrics = ["engagement","mutual_gaze","distance"]
        rob_metric_map = {"distance":"distance_to_robot","mutual_gaze":"mutual_gaze_with_robot","engagement":"engagement_value_with_robot"}
        st = fig.suptitle(exp_name, fontsize=14)
        for i in range(len(metrics)):
            axs[i].set_title(metrics[i])

        num_pairs = 0
        for i in range(len(bodies)-1):
            if bodies[i] != "ROBOT":
                log_i = self.compile_log(self.bodies[bodies[i]])
            for j in range(i+1,len(bodies)):
                num_pairs += 1
                if bodies[j] == "ROBOT":
                    for k in range(len(metrics)):
                        self.plot(log_i,label="{}_ROBOT".format(bodies[i]),x=x,y=rob_metric_map[metrics[k]],ax=axs[k],show=False,set_xlim=False)
                else:
                    for k in range(len(metrics)):
                        y = "{}_{}".format(bodies[j],metrics[k])
                        if y in log_i:
                            self.plot(log_i,label="{}_{}".format(bodies[i],bodies[j]),x=x,y=y,ax=axs[k],show=False,set_xlim=False)

        # Presentation
        fig.tight_layout()
        st.set_y(0.95)
        fig.subplots_adjust(top=0.85)

        max_cols = 4
        if num_pairs <= max_cols:
            ncol = num_pairs
            nrow = 1
        else:
            ncol = max_cols
            nrow = num_pairs // ncol
        
        lines_labels = [axs[0].get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        leg = fig.legend(lines, labels,loc = 'upper center',ncol=ncol)
        [ax.get_legend().remove() for ax in axs]

        '''
        offset = -0.5
        bb = leg.get_bbox_to_anchor().transformed(axs[-1].transAxes.inverted())
        bb.y0 += offset
        bb.y1 += offset
        leg.set_bbox_to_anchor(bb, transform = axs[-1].transAxes)
        '''

        if show:
            plt.show()

        if save_fig:
            plt.savefig(self.fig_dir+x+"__EngagementComparison.png", bbox_extra_artists=(leg,st), bbox_inches='tight')   

        return fig,axs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default="MultiBodyTest")
    args = parser.parse_args()


    bodies = None
    show = False

    lr = LogReader(args.exp)
    print("Bodies: {}".format(list(lr.bodies.keys())))

    lr.plot_comparison(args.exp,y="level",bodies=bodies,show=show)
    lr.plot_comparison(args.exp,y="engagement_value_with_robot",bodies=bodies,show=show)
    lr.plot_comparison(args.exp,y="distance_to_robot",bodies=bodies,show=show)
    lr.plot_comparison(args.exp,y="mutual_gaze_with_robot",bodies=bodies,show=show)
    lr.plot_comparison(args.exp,y="activity",bodies=bodies,show=show)
    lr.plot_comparison(args.exp,y="vel_magnitude",bodies=bodies,show=show)
    lr.plot_groups(args.exp,show=show)
    lr.plot_enagegement_comparisons(args.exp,show=show)

    


