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
Note the lack of .bag extension. Leaving the exp flag empty will default to FrontTurn360_0

If your rosbags directory is empty, or the bags do not contain the expected topic structure, things will probably break.