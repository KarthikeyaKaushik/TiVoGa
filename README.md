#SNN Controller for Bio-Snake Robot 
Made in TUM for the course 'Development and Control of a Bio-Snake Robot'
Works with ROS Kinetic on Ubuntu 16.04

Installation
======
1. Install [Gazebo](http://gazebosim.org/download) and [ROS](http://wiki.ros.org/kinetic/Installation/Ubuntu)
2. Clone the [snake simulation](https://github.com/alexansari101/snake_ws):
```bash
$ git clone https://github.com/alexansari101/snake_ws
```
3. Move to the downloaded workspace and build
```bash
$ cd snake_ws
$ catkin_make
```
4. Go to the parent directory and clone this repo
```bash
$ git clone https://github.com/KarthikeyaKaushik/TiVoGa 
```

Running
======
1. Go to the directory containing both repos

2. Source 
```bash 
$ cd snake_ws
$ source devel/setup.bash
```

3. Start the simulation
```bash
$ roslaunch snake_control gazebo.launch gait:=false paused:=false
```

4. In a new tab start the positioning script
```bash
$ cd ..
$ cd TiVoGa
$ ./get_snake_pos.py
```

5. In a new tab start the gaits script
```bash
$ ./gaits.py 1
```
6. Add a target in Gazebo

7. Start recording in the positioning script





