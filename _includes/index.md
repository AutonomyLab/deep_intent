### Abstract 
We explore prediction of urban pedestrian actions by generating a video future of the traffic scene, and show promising results in classifying pedestrian behaviour before it is observed. We compare several encoder-decoder network models that predict 16 frames (400-600 milliseconds of video) from the preceding 16 frames. Our main contribution is a method for learning a sequence of representations to iteratively transform features learnt from the input to the future. Then we use a binary action classifier network for determining a pedestrian’s crossing intent from predicted video. Our results show an average precision of 81%, significantly higher than previous methods. The model with best classification performance runs for 117 ms on commodity GPU, giving an effective look-ahead of 416 ms.
<br />

[Pratik Gujjar](https://www.sfu.ca/~pgujjar/) and [Richard Vaughan](http://rtv.github.io/) <br />
<i class="fa fa-github"></i>&nbsp;<a href="https://github.com/AutonomyLab/deep_intent">Code</a>

##### Future Prediction 
Our objective is to predict the future positions of salient objects like vehicles and pedestrians by learning their motion. Functionally, an encoder reads a sequence of frames ![xT]<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\large&space;\boldsymbol{x}&space;=&space;\{x_{T},...,&space;x_{1}\}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\large&space;\boldsymbol{x}&space;=&space;\{x_{T},...,&space;x_{1}\}" title="\large \boldsymbol{x} = \{x_{T},..., x_{1}\}" /></a>to yield dense representations z = {z 1 , ..., z T }. Conditioned on z, a decoder will then auto-regressively predict an image sequence y 0 = {y T 0 +1 , ..., y 2T} by minimizing a pixel-wise loss between y 0 and ground truth frames y = {y T +1 , ..., y 2T }. Each generated frame is of the same resolution as the input. We reverse the temporal ordering of input data to condition the latent space with spatial information from the latest frame. The most recent frame carries forward the closest contextual resemblance. Recursively learning representations from each input frame, we expect to first learn a temporal regularity in the early representations and parametrize a temporal variance in the later representations.

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
    <td aligh="center"><img src="./public/gifs/truth/vid_23.gif" width="77.5%"></td>
  </tr>
  <tr>
    <td align="center"><img src="./public/gifs/pred/vid_91.gif" width="65%"></td>
    <td aligh="center"><img src="./public/gifs/truth/vid_91.gif" width="77.5%"></td>
  </tr>
  <tr>
    <td align="center"><img src="./public/gifs/pred/vid_165.gif" width="65%"></td>
    <td aligh="center"><img src="./public/gifs/truth/vid_165.gif" width="77.5%"></td>
   </tr>
   <tr>
   <td align="center"><img src="./public/gifs/pred/vid_232.gif" width="65%"></td>
    <td aligh="center"><img src="./public/gifs/truth/vid_232.gif" width="77.5%"></td>
  </tr>
  <tr>
    <td align="center"><img src="./public/gifs/pred/vid_306.gif" width="65%"></td>
    <td aligh="center"><img src="./public/gifs/truth/vid_306.gif" width="77.5%"></td>
  </tr>
  <tr>
    <td align="center"><img src="./public/gifs/pred/vid_350.gif" width="65%"></td>
    <td aligh="center"><img src="./public/gifs/truth/vid_350.gif" width="77.5%"></td>
  </tr>
  <tr>
    <td align="center"><img src="./public/gifs/pred/vid_417.gif" width="65%"></td>
    <td aligh="center"><img src="./public/gifs/truth/vid_417.gif" width="77.5%"></td>
  </tr>
  <tr>
    <td align="center"><img src="./public/gifs/pred/vid_471.gif" width="65%"></td>
    <td aligh="center"><img src="./public/gifs/truth/vid_471.gif" width="77.5%"></td>
  </tr>
</table>





