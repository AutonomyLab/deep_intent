### Abstract

Humans are constantly engaged in making predictions about our surroundings in order to safely navigate them. These predictions depend on our models of the objects in the environment and the relationships between them. This work is motivated to provide an autonomous car with a predictive intelligence that enhances it role in human interaction scenarios. One commonly encountered case is that of a pedestrian crossing a road. We model the visual structure of the traffic scene by learning to generate a sequence of frames that depict its future. Our spatio-temporal network architecture considers the context of all objects in motion against traditional approaches of segmenting pedestrian motion out for action recognition. We further analyze the predicted video using a deep neural network to identify pedestrians' actions. This work is focused on learning representations that account for motion and complex relationships of objects in a traffic scene in order to anticipate adverse pedestrian behaviour 10-20 frames in the future or 400-600 ms prior to its occurrence. We also show that the predictions are extensible to difficult road conditions like snowy weather or sun-glare on camera lens. Recognizing adverse pedestrian actions in advance, we can add precious time to supplement a driver's response, potentially saving human lives.
<br />
[Pratik Gujjar](https://www.sfu.ca/~pgujjar/) <br />
<i class="fa fa-github"></i>&nbsp;<a href="https://github.com/AutonomyLab/deep_intent">Code</a>
<a href="https://docs.google.com/presentation/d/1iRiFNqW0-Q5b8KaCYM9JfhME_rDH-ZGX-Ka-yxU9bb4/edit?usp=sharing">Slides</a>

##### Network Structure
Encoder and decoder are both convolutional neural networks. Decoder employs transposed 3D convolutions to upsample learnt representation.

<img src="./public/network.png" width="100%">

<style>
table, th, td {
    border: 0px solid black;
}
</style>

##### Examples from the JAAD Dataset 
10 frames input -> 10 frames (1/3 seconds) future prediction

<table>
  <tr>
    <td align="center">Input Frames</td>
    <td align="center">Predicted Future</td>
    <td align="center">Ground Truth</td>
    <td align="center">Input Frames</td>
    <td align="center">Predicted Future</td>
    <td align="center">Ground Truth</td>
  </tr>
  <tr>
    <td align="center"><img src="./public/1_orig.gif"></td>
    <td aligh="center"><img src="./public/1_pred.gif"></td>
    <td align="center"><img align="left" src="./public/1_truth.gif"></td>
    <td align="center"><img src="./public/2_orig.gif"></td>
    <td aligh="center"><img src="./public/2_pred.gif"></td>
    <td align="center"><img align="left" src="./public/2_truth.gif"></td>
  </tr>
  <tr>
    <td align="center"><img src="./public/3_orig.gif"></td>
    <td aligh="center"><img src="./public/3_pred.gif"></td>
    <td align="center"><img align="left" src="./public/3_truth.gif"></td>
    <td align="center"><img src="./public/4_orig.gif"></td>
    <td aligh="center"><img src="./public/4_pred.gif"></td>
    <td align="center"><img align="left" src="./public/4_truth.gif"></td>
  </tr>
  <tr>
    <td align="center"><img src="./public/5_orig.gif"></td>
    <td aligh="center"><img src="./public/5_pred.gif"></td>
    <td align="center"><img align="left" src="./public/5_truth.gif"></td>
    <td align="center"><img src="./public/6_orig.gif"></td>
    <td aligh="center"><img src="./public/6_pred.gif"></td>
    <td align="center"><img align="left" src="./public/6_truth.gif"></td>
  </tr>
</table>

##### Examples from the JAAD Dataset 
10 frames input -> 20 frames (2/3 seconds) of future predictions

<table>
  <tr>
    <td align="center">Input</td>
    <td align="center">Predictions</td>
    <td align="center">Truth</td>
    <td align="center">Input</td>
    <td align="center">Predictions</td>
    <td align="center">Truth</td>
  </tr>
  <tr>
    <td align="center"><img src="./public/20_frame_1_orig.gif"></td>
    <td aligh="center"><img src="./public/20_frame_1_pred.gif"></td>
    <td align="center"><img align="left" src="./public/20_frame_1_truth.gif"></td>
    <td align="center"><img src="./public/20_frame_2_orig.gif"></td>
    <td aligh="center"><img src="./public/20_frame_2_pred.gif"></td>
    <td align="center"><img align="left" src="./public/20_frame_2_truth.gif"></td>
  </tr>
  <tr>
    <td align="center"><img src="./public/20_frame_3_orig.gif"></td>
    <td aligh="center"><img src="./public/20_frame_3_pred.gif"></td>
    <td align="center"><img align="left" src="./public/20_frame_3_truth.gif"></td>
    <td align="center"><img src="./public/20_frame_4_orig.gif"></td>
    <td aligh="center"><img src="./public/20_frame_4_pred.gif"></td>
    <td align="center"><img align="left" src="./public/20_frame_4_truth.gif"></td>
  </tr>

</table>
<!---
##### 20 frame predictions compared across two models over two epochs of training. Model 1 upsamples 10 input frames to 20 output frames at early stages of the encoder. Model 2 does this in the final stage. 

<table>
  <tr>
    <td align="center" colspan="2">Epoch 1</td>
    <td align="center" colspan="2">Epoch 7</td>
    <td align="center" rowspan="2">Truth</td>
  </tr>
  <tr>
    <td align="center">Model 1</td>
    <td align="center">Model 2</td>
    <td align="center">Model 1</td>
    <td align="center">Model 2</td>
  </tr>
  <tr>
    <td align="center"><img src="./public/model_1_epoch_0_1.gif"></td>
    <td aligh="center"><img src="./public/model_2_epoch_0_1.gif"></td>
    <td align="center"><img src="./public/model_1_epoch_6_1.gif"></td>
    <td aligh="center"><img src="./public/model_2_epoch_6_1.gif"></td>
    <td aligh="center"><img src="./public/20_truth_1.gif"></td>
  </tr>
  <tr>
    <td align="center"><img src="./public/model_1_epoch_0_2.gif"></td>
    <td aligh="center"><img src="./public/model_2_epoch_0_2.gif"></td>
    <td align="center"><img src="./public/model_1_epoch_6_2.gif"></td>
    <td aligh="center"><img src="./public/model_2_epoch_6_2.gif"></td>
    <td aligh="center"><img src="./public/20_truth_2.gif"></td>
  </tr>
</table>
--->

##### Asymmetric encoder-decoder against symmetric encoder-decoder architetures
We observed that an asymmetric structure, particularly so with a "stronger" decoder, performs better than using symmetric models
derived extended from convolutional classifiers.

<table>
  <tr>
    <td align="center">Symmetric</td>
    <td align="center">Asymmetric</td>
    <td align="center">Symmetric</td>
    <td align="center">Asymmetric</td>
  </tr>
  <tr>
    <td align="center"><img src="./public/symmetric_1.gif"></td>
    <td aligh="center"><img src="./public/asymmetric_1.gif"></td>
    <td align="center"><img src="./public/symmetric_2.gif"></td>
    <td aligh="center"><img src="./public/asymmetric_2.gif"></td>
  </tr>
</table>

##### Test results for samples from the KITTI dataset (The network is not trained on this data)
Though the network has never seen these samples, predictions are coherent and plausible.

<table>
  <tr>
    <td align="center">Predictions</td>
    <td align="center">Truth</td>
    <td align="center">Predictions</td>
    <td align="center">Truth</td>
  </tr>
  <tr>
    <td align="center"><img src="./public/kitti_highway_pred.gif"></td>
    <td aligh="center"><img src="./public/kitti_highway_truth.gif"></td>
    <td align="center"><img src="./public/kitti_people_pred.gif"></td>
    <td aligh="center"><img src="./public/kitti_people_truth.gif"></td>
  </tr>
</table>


##### Insufficiently accurate predictions for samples from the KITTI dataset
We deduce that the inaccuracies stem from the fast movement of the scene in samples from the KITTI dataset. Since the model is only trained on JAAD samples, object translations between frames for KITTI data are not without flaws.

<table>
  <tr>
    <td align="center">Predictions</td>
    <td align="center">Truth</td>
    <td align="center">Predictions</td>
    <td align="center">Truth</td>
  </tr>
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

##### 10-frame predictions unrolled
Unrolled frames to analyze coherence in predictions; both in object distinctions and overall quality.

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





