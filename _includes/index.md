### Abstract

Drivers and pedestrians engage in non-verbal and social cues to signal their intent, which is crucial to their interactions in traffic scenarios. We propose to learn such cues and model a pedestrian’s intent. The learnt model then predicts actions likely to be performed 400 − 600ms in the future. Responding to adverse actions in advance, we tread towards full autonomy.
<i class="fa fa-github"></i>&nbsp;<a href="https://github.com/AutonomyLab/deep_intent">Code</a>

##### Network Structure

<img src="./public/network.png" width="100%">

<style>
table, th, td {
    border: 0px solid black;
}
</style>

##### Examples from the JAAD Dataset
<table>
  <tr>
    <td align="center"><img src="./public/1_orig.gif"></td>
    <td aligh="center"><img src="./public/1_pred.gif"></td>
    <td align="center"><img src="./public/1_truth.gif"></td>
  </tr>
  tr>
    <td align="center"><img src="./public/2_orig.gif"></td>
    <td aligh="center"><img src="./public/2_pred.gif"></td>
    <td align="center"><img src="./public/2_truth.gif"></td>
  </tr>
  tr>
    <td align="center"><img src="./public/3_orig.gif"></td>
    <td aligh="center"><img src="./public/3_pred.gif"></td>
    <td align="center"><img src="./public/3_truth.gif"></td>
  </tr>
  tr>
    <td align="center"><img src="./public/4_orig.gif"></td>
    <td aligh="center"><img src="./public/4_pred.gif"></td>
    <td align="center"><img src="./public/4_truth.gif"></td>
  </tr>
  tr>
    <td align="center"><img src="./public/5_orig.gif"></td>
    <td aligh="center"><img src="./public/5_pred.gif"></td>
    <td align="center"><img src="./public/5_truth.gif"></td>
  </tr>
  tr>
    <td align="center"><img src="./public/6_orig.gif"></td>
    <td aligh="center"><img src="./public/6_pred.gif"></td>
    <td align="center"><img src="./public/6_truth.gif"></td>
  </tr>
</table>

##### Test results for samples from the KITTI dataset (The network is not trained on this data)

<table>
  <tr>
    <td align="center"><img src="./public/kitti_highway_pred.gif"></td>
    <td aligh="center"><img src="./public/kitti_highway_truth.gif"></td>
    <td align="center"><img src="./public/kitti_people_pred.gif"></td>
    <td aligh="center"><img src="./public/kitti_people_truth.gif"></td>
  </tr>
</table>


##### Insufficiently accurate predictions for samples from the KITTI dataset

<table>
  <tr>
    <td align="center"><img align="right" src="./public/kitti_1_pred.gif"></td>
    <td aligh="center"><img align="left" src="./public/kitti_1_truth.gif"></td>
    <td align="center"><img align="right" src="./public/kitti_2_pred.gif"></td>
    <td aligh="center"><img align="left" src="./public/kitti_2_truth.gif"></td>
  </tr>
   <tr>
    <td align="center"><img align="right" src="./public/kitti_3_pred.gif"></td>
    <td aligh="center"><img align="left" src="./public/kitti_3_truth.gif"></td>
    <td align="center"><img align="right" src="./public/kitti_4_pred.gif"></td>
    <td aligh="center"><img align="left" src="./public/kitti_4_truth.gif"></td>
  </tr>
</table>

##### 10-frame predictions 
Input | Predictions | Ground Truth

<img src="./public/1_orig.png" width="100%"> 
<img src="./public/1_pred.png" width="100%">
<img src="./public/1_truth.png" width="100%">

----

<img src="./public/2_orig.png" width="100%"> 
<img src="./public/2_pred.png" width="100%">
<img src="./public/2_truth.png" width="100%">

----
<img src="./public/3_orig.png" width="100%"> 
<img src="./public/3_pred.png" width="100%">
<img src="./public/3_truth.png" width="100%">

----

<img src="./public/4_orig.png" width="100%"> 
<img src="./public/4_pred.png" width="100%">
<img src="./public/4_truth.png" width="100%">

----

<img src="./public/5_orig.png" width="100%"> 
<img src="./public/5_pred.png" width="100%">
<img src="./public/5_truth.png" width="100%">

----





