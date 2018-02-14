### Abstract

Humans are constantly engaged in making predictions about our surroundings in order to safely navigate them. These predictions depend on our models of the objects in the environment and the relationships between them. This work is motivated to provide an autonomous car with a predictive intelligence that enhances it role in human interaction scenarios. One commonly encountered case is that of a pedestrian crossing a road. We model the visual structure of the traffic scene by learning to generate a sequence of frames that depict its future. Our spatio-temporal network architecture considers the context of all objects in motion against traditional approaches of segmenting pedestrian motion out for action recognition. We further analyze the predicted video using a deep neural network to identify pedestrians' actions. This work is focused on learning representations that account for motion and complex relationships of objects in a traffic scene in order to anticipate adverse pedestrian behaviour 10-20 frames in the future or 400-600 ms prior to its occurrence. We also show that the predictions are extensible to difficult road conditions like snowy weather or sun-glare on camera lens. Recognizing adverse pedestrian actions in advance, we can add precious time to supplement a driver's response, potentially saving human lives.
<br />
[Pratik Gujjar](https://www.sfu.ca/~pgujjar/) <br /> [Richard Vaughan](http://rtv.github.io/) <br />
<i class="fa fa-github"></i>&nbsp;<a href="https://github.com/AutonomyLab/deep_intent">Code</a>
<a href="https://docs.google.com/presentation/d/1iRiFNqW0-Q5b8KaCYM9JfhME_rDH-ZGX-Ka-yxU9bb4/edit?usp=sharing">Slides</a>

##### Network Architecture
The encoder is a spatio-temporal neural network composed of three 3D convolutional layers.
The decoder is composed of ConvLSTM layers. Unlike the encoder, the decoder layers up-sample steadily to facilitate fluid transforms.

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





