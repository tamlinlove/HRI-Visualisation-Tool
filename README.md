# HRI-Visualisation-Tool
A simple tool for visualising rosbagged logs of an HRI application

## Installation
Ideally, you should create a virtual environment, for example:
```
python -m venv venv
source venv/bin/activate
```
From there, you can install all requirements via:
```
pip install -r requirements.txt
```

## Running the code
To run the visualisation tool, simply run:
```
python visualiser.py --exp NAME_OF_ROSBAG
```
e.g.
```
python visualiser.py --exp Include2Exclude1_0
```
Note the lack of .bag extension. Leaving the exp flag empty will default to *FrontTurn360_0*

If your rosbags directory is empty, or the bags do not contain the expected topic structure, things will probably break.

## Rosbag Topic Structure
The following topics are expected in each rosbag:
*   /humans/bodies/{BODY_ID}/activity   hriri/Activity
*   /humans/bodies/{BODY_ID}/body_orientation   geometry_msgs/Vector3Stamped
*   /humans/bodies/{BODY_ID}/engagement_status  hri_msgs/EngagementLevel
*   /humans/bodies/{BODY_ID}/face_orientation   geometry_msgs/Vector3Stamped
*   /humans/bodies/{BODY_ID}/poses  geometry_msgs/PoseArray
*   /humans/bodies/{BODY_ID}/skeleton2d hri_msgs/Skeleton2D
*   /humans/bodies/{BODY_ID}/velocity   geometry_msgs/TwistStamped
*   /humans/bodies/tracked  hri_msgs/IdsList
*   /humans/interactions/engagements    hriri/EngagementValue
*   /humans/interactions/groups hri_msgs/Group
*   /opendr/image_pose_annotated    sensor_msgs/Image
*   /opendr/poses   opendr_bridge/OpenDRPose2D
