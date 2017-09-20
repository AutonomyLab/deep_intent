### Introduction

This page provides supplementary materials for the following paper, presented in [2016 IEEE/RSJ International Conference on Intelligent Robots and Systems](http://www.iros2016.org/) (IROS 2016).

**"UAV, Come To Me: End-to-End, Multi-Scale Situated HRI with an Uninstrumented Human and a Distant UAV"**<br />
[Mani Monajjemi](https://mani.im), [Sepehr Mohaimenianpour](http://sepehr.im) and [Richard Vaughan](https://www.cs.sfu.ca/~vaughan/)<br />
_[AutonomyLab](http://autonomylab.org/), [Simon Fraser University](http://sfu.ca)_

<i class="fa fa-file-text-o"></i>&nbsp;<a href="#">BIB</a>&nbsp;
<i class="fa fa-file-pdf-o"></i>&nbsp;<a href="https://autonomylab.github.io/doc/monajjemi_iros16.pdf">PDF</a>
<i class="fa fa-github"></i>&nbsp;<a href="#source-code">Code</a>

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/6kKuGH0B8XY?rel=0" frameborder="0" allowfullscreen></iframe>

#### Changelog

- Oct 12, 2016: Updated webpage to include the paper and all source repositories
- Mar 14, 2016: Initial webpage and source code release

#### Documentation

TBA

#### Source Code

##### "bebop_hri": The behavior generator code, configurations and launch files for the paper (ROS package)

- <i class="fa fa-github"></i>&nbsp;[Git Repository](https://github.com/AutonomyLab/bebop_hri)<a href=""></a>
- IROS'16 commit hash: `2ea50ade4e88`

##### "obzerver": Periodic Salient Object Detector

- <i class="fa fa-github"></i>&nbsp;[Git repository](https://github.com/AutonomyLab/obzerver)
- Branch/version: `opencv-3.0`
- Note: Requires OpenCV 3.x

##### "obzerver_ros": The ROS binding for "obzerver"

- <i class="fa fa-github"></i>&nbsp;[Git repository](https://github.com/AutonomyLab/obzerver_ros)
- IROS'16 commit hash: `23e7d29fa8b3`
- Note: Requires a version of ROS [vision_opencv](https://github.com/ros-perception/vision_opencv) package that is compiled against OpenCV 3.x (e.g [our fork](https://github.com/AutonomyLab/vision_opencv))

##### "bebop_autonomy": ROS driver for Parrot Bebop Drone

- <i class="fa fa-github"></i>&nbsp;[Git repository](https://github.com/AutonomyLab/bebop_autonomy)
- IROS'16 version: `0.4.1`

##### "bebop_vel_ctrl": Velocity controller for Parrot Bebop Drone (ROS package)

- <i class="fa fa-github"></i>&nbsp;[Git repository](https://github.com/AutonomyLab/bebop_vel_ctrl)
- IROS'16 commit hash: `4962f392b33c`

##### "bebop_vservo": Visual servo controller for Parrot Bebop Drone (ROS package)

- <i class="fa fa-github"></i>&nbsp;[Git repository](https://github.com/AutonomyLab/bebop_vservo)
- IROS'16 commit hash: `f25bfaaf74d0`

##### "autonomy_leds": Firmware and animation engine for DotStar LED strips (ROS package)

- <i class="fa fa-github"></i>&nbsp; [Git repository](https://github.com/AutonomyLab/autonomy_leds)
- Branch/version (firmware): `cjmcu_beetle`
- Branch/version (engine): `dev`

##### "cftld_ros": ROS wrapper for CFTld, a long-term visual tracker

- <i class="fa fa-github"></i>&nbsp;[Git repository](https://github.com/AutonomyLab/cftld_ros)
- IROS'16 commit hash: `378029e2b0eb0c61c555710838844cd13e026aab`
- Based on [CFTld tracker](https://github.com/klahaag/CFtld)

##### "autonomy_human": Short-range Face Engagement and Optical Flow Based Gesture Detection

- [Git repository](https://github.com/AutonomyLab/autonomy_hri/tree/dev/autonomy_human)
- Branch/version: `dev`
- Note: Requires a version of ROS [vision_opencv](https://github.com/ros-perception/vision_opencv) package that is compiled against OpenCV 3.x (e.g. [our fork](https://github.com/AutonomyLab/vision_opencv))
