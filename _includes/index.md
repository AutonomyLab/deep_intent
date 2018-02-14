### Abstract

Humans are constantly engaged in making predictions about our surroundings in order to safely navigate them. These predictions depend on our models of the objects in the environment and the relationships between them. This work is motivated to provide an autonomous car with a predictive intelligence that enhances it role in human interaction scenarios. One commonly encountered case is that of a pedestrian crossing a road. We model the visual structure of the traffic scene by learning to generate a sequence of frames that depict its future. Our spatio-temporal network architecture considers the context of all objects in motion against traditional approaches of segmenting pedestrian motion out for action recognition. We further analyze the predicted video using a deep neural network to identify pedestrians' actions. This work is focused on learning representations that account for motion and complex relationships of objects in a traffic scene in order to anticipate adverse pedestrian behaviour 10-20 frames in the future or 400-600 ms prior to its occurrence. We also show that the predictions are extensible to difficult road conditions like snowy weather or sun-glare on camera lens. Recognizing adverse pedestrian actions in advance, we can add precious time to supplement a driver's response, potentially saving human lives.
<br />
[Pratik Gujjar](https://www.sfu.ca/~pgujjar/) and [Richard Vaughan](http://rtv.github.io/) <br />
<i class="fa fa-github"></i>&nbsp;<a href="https://github.com/AutonomyLab/deep_intent">Code</a>
<a href="https://docs.google.com/presentation/d/1iRiFNqW0-Q5b8KaCYM9JfhME_rDH-ZGX-Ka-yxU9bb4/edit?usp=sharing">Slides</a>

##### Future Frames Prediction
The encoder is a spatio-temporal neural network composed of three 3D convolutional layers.
The decoder is composed of ConvLSTM layers. Unlike the encoder, the decoder layers up-sample steadily to facilitate fluid transforms.

<p style="text-align:center;"><img src="./public/abstract_net.png" align="center" width="60%"></p>
<img src="./public/network.png" width="150%">

<style>
table, th, td {
    border: 0px solid black;
}
</style>


##### Pedestrian Action Recognition
Fine-tuned a 3D Convolutional model C3D1, pretrained on Sports 1M dataset (487 classes). Baseline performance scores by learning to recognize actions in the original 16 frame sequences. Subsampling employed to battle skewed data distribution.


<p style="text-align:center;"><img src="./public/abstract_net_cla.png" align="center" width="80%"></p>
<img src="./public/ped_action_set.png" width="70%">

<style>
table, th, td {
    border: 0px solid black;
}
</style>

##### Examples from the JAAD Dataset 
10 frames input -> 10 frames (1/3 seconds) future prediction

<table>
  <tr>
    <td align="center">Past + Predicted Frames</td>
    <td align="center">Past + Ground Truth</td>
  </tr>
  <tr>
    <td align="center"><img src="./public/gifs/pred/vid_23.gif" width="65%"></td>
    <td aligh="center"><img src="./public/gifs/truth/vid_23.gif" width="78.5%"></td>
  </tr>
  <tr>
    <td align="center"><img src="./public/gifs/pred/vid_91.gif" width="60%"></td>
    <td aligh="center"><img src="./public/gifs/truth/vid_91.gif" width="80%"></td>
  </tr>
  <tr>
    <td align="center"><img src="./public/gifs/pred/vid_165.gif" width="60%"></td>
    <td aligh="center"><img src="./public/gifs/truth/vid_165.gif" width="80%"></td>
   </tr>
   <tr>
   <td align="center"><img src="./public/gifs/pred/vid_232.gif" width="60%"></td>
    <td aligh="center"><img src="./public/gifs/truth/vid_232.gif" width="80%"></td>
  </tr>
  <tr>
    <td align="center"><img src="./public/gifs/pred/vid_306.gif" width="60%"></td>
    <td aligh="center"><img src="./public/gifs/truth/vid_306.gif" width="80%"></td>
  </tr>
  <tr>
    <td align="center"><img src="./public/gifs/pred/vid_350.gif" width="60%"></td>
    <td aligh="center"><img src="./public/gifs/truth/vid_350.gif" width="80%"></td>
  </tr>
  <tr>
    <td align="center"><img src="./public/gifs/pred/vid_417.gif" width="60%"></td>
    <td aligh="center"><img src="./public/gifs/truth/vid_417.gif" width="80%"></td>
  </tr>
  <tr>
    <td align="center"><img src="./public/gifs/pred/vid_471.gif" width="60%"></td>
    <td aligh="center"><img src="./public/gifs/truth/vid_471.gif" width="80%"></td>
  </tr>
</table>





