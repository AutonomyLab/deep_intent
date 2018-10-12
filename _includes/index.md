### Abstract 
We explore prediction of urban pedestrian actions by generating a video future of the traffic scene, and show promising results in classifying pedestrian behaviour before it is observed. We compare several encoder-decoder network models that predict 16 frames (400-600 milliseconds of video) from the preceding 16 frames. Our main contribution is a method for learning a sequence of representations to iteratively transform features learnt from the input to the future. Then we use a binary action classifier network for determining a pedestrian’s crossing intent from predicted video. Our results show an average precision of 81%, significantly higher than previous methods. The model with best classification performance runs for 117 ms on commodity GPU, giving an effective look-ahead of 416 ms.
<br />

[Pratik Gujjar](https://www.sfu.ca/~pgujjar/) and [Richard Vaughan](http://rtv.github.io/) <br />

<a href="https://github.com/AutonomyLab/deep_intent">[code]</a>

<p align="center">
<a href="http://www.youtube.com/watch?feature=player_embedded&v=UHMBtu43Gi4
" target="_blank"><img src="./public/video_img.png" 
alt="DeepIntent" width="637" height="358" border="0" /></a>
</p>

### Predicting a Future
Typically, three-quarters of a second are needed to see a hazard and to decide to stop. Three-quarters more of a second are needed to actuate the brakes to stop a vehicle. An early prediction of a potentially hazardous action, could
add precious time before one decides to act.

<p align="center">
<img src="./public/see-think-do.png" border="0" />
</p>

Commonly expected crossing behaviour *Standing-Looking-Crossing* and *Crossing-Looking* only account for half the situations observed. In more than 90% of the [JAAD](http://data.nvision2.eecs.yorku.ca/JAAD_dataset/) dataset, pedestrians are observed to use some-form of non-verbal communication in their crossing behaviour. The most prominent signal is to look at oncoming traffic. 

<table>
  <tr>
    <td align="center"><img src="./public/look.png" width="50%"></td>
    <td align="center"><img src="./public/step-forward.png" width="50%"></td>
  </tr>
</table>

Without contextual knowledge, the pedestrian in the example below is likely to be predicted as continuing to walk across. The knowledge of the stationary car adds the information that the pedestrian is very likely to stop before it.

<p align="center">
<img src="./public/context.gif" border="0" />
</p>

<!--### Technical Description
Our objective is to predict the future positions of salient objects like vehicles and pedestrians by learning their motion. Functionally, an encoder reads a sequence of frames __x__ to yield dense representations __z__. Conditioned on __z__, a decoder will then auto-regressively predict an image sequence __y'__ by minimizing a pixel-wise loss between __y'__ and ground truth frames __y__. Each generated frame is of the same resolution as the input. We reverse the temporal ordering of input data to condition the latent space with spatial information from the latest frame. The most recent frame carries forward the closest contextual resemblance. Recursively learning representations from each input frame, we expect to first learn a temporal regularity in the early representations and parametrize a temporal variance in the later representations.
-->
<!--<p align="center"> 
<img src="./public/abstract-net.svg" alt="abstract-net" width="537" height="258" border="0" />
</p>-->

### Experiments
We use the [JAAD](http://data.nvision2.eecs.yorku.ca/JAAD_dataset/) dataset <a href="https://arxiv.org/abs/1609.04741">[paper]</a> consisting of 346 high resolution videos in pedestrian interaction scenarios. We train our encoder-decoder stack to optimize for a combination of *l1* and *l2* losses. The losses are calculated between the *N* pixels of T predicted frames __y'__ and ground truth frames __y__. For video prediction experiments we set N = 128 × 208 and T = 16 frames. We train three kinds of models for future prediction: a fully convolutional model (Conv3D), a recurrent decoder model (Segment) and a residual encoder-decoder model (Res-EnDec). We perform ablation studies on our Res-EnDec model to determine the importance of the residual connections, dilated convolutions and reversal of image data. 

<!--
<table>
  <tr
    <td align="center"><img src="./public/conv.png" width="70%"></td>
    <td align="center"><img src="./public/segment.png" width="70%"></td>
  </tr>
  <tr>
    <td align="center">Conv model</td>
    <td align="center">Segment model</td>
  </tr>
  <tr>
    <td align="center"><img src="./public/res.png" width="70%"></td>
    <td aligh="center"><img src="./public/rendec.png"  width="70%"></td>
  </tr>
   <tr>
    <td align="center">Res model</td>
    <td align="center">Res-EnDec model</td>
  </tr>
</table>-->

<style> table, th, td { border: 0px solid black; } </style>
<table>
  <tr>
    <td align="center">History + Ground Truth</td>
    <td align="center" colspan="4">History + Prediction</td>
  </tr>
  <tr>
    <td align="center"></td>
    <td align="center">Conv</td>
  </tr>
  <tr>
    <td align="center"><img src="./public/github_examples/future/truth/rendec/vid_710.gif" width="100%"></td>
    <td align="center"><img src="./public/github_examples/future/pred/conv/vid_710.gif" width="100%"></td> 
  </tr>
  <tr>
    <td align="center"></td>
    <td align="center">Segment</td>
  </tr>
  <tr>
    <td align="center"></td>
    <td align="center"><img src="./public/github_examples/future/pred/kernel/vid_710.gif" width="100%"></td>
  </tr>
  <tr>
    <td align="center"></td>
    <td align="center">Res</td>
  </tr>
  <tr>
    <td align="center"></td>
    <td align="center"><img src="./public/github_examples/future/pred/res/vid_710.gif" width="100%"></td>
  </tr>
  <tr>
    <td align="center"></td>
    <td align="center">Res-EnDec</td>
  </tr>
  <tr>
    <td align="center"></td>
    <td align="center"><img src="./public/github_examples/future/pred/rendec/vid_710.gif" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <td align="center"></td>
    <td align="center">Conv</td>
  </tr>
  <tr>
    <td align="center"><img src="./public/github_examples/future/truth/rendec/vid_758.gif" width="100%"></td>
    <td align="center"><img src="./public/github_examples/future/pred/conv/vid_758.gif" width="100%"></td> 
  </tr>
  <tr>
    <td align="center"></td>
    <td align="center">Segment</td>
  </tr>
  <tr>
    <td align="center"></td>
    <td align="center"><img src="./public/github_examples/future/pred/kernel/vid_758.gif" width="100%"></td>
  </tr>
  <tr>
    <td align="center"></td>
    <td align="center">Res</td>
  </tr>
  <tr>
    <td align="center"></td>
    <td align="center"><img src="./public/github_examples/future/pred/res/vid_758.gif" width="100%"></td>
  </tr>
  <tr>
    <td align="center"></td>
    <td align="center">Res-EnDec</td>
  </tr>
  <tr>
    <td align="center"></td>
    <td align="center"><img src="./public/github_examples/future/pred/rendec/vid_758.gif" width="100%"></td>
  </tr>
</table>

### Crossing Intent
The task of action recognition is motivated by the idea that by looking ahead in time, we could react to a hazardous pedestrian interaction a little earlier, with safety benefits. We do this end-to-end by appending a binary action classifier to our future video generator. In this task, we want to learn to predict a pedestrian’s crossing intent across a multitude of crossing scenarios and behaviours.

<table>
  <tr>
    <td align="center">History + Ground Truth</td>
    <td align="center">History + Prediction</td>
  </tr>
  <tr>
    <td align="center"><img src="./public/github_examples/crossing/truth/vid_231.gif" width="100%"></td>
    <td align="center"><img src="./public/github_examples/crossing/pred/vid_231.gif" width="100%"></td> 
  </tr>
  <tr>
    <td align="center"><img src="./public/github_examples/crossing/truth/vid_272.gif" width="100%"></td>
    <td align="center"><img src="./public/github_examples/crossing/pred/vid_272.gif" width="100%"></td> 
  </tr>
  <tr>
    <td align="center"><img src="./public/github_examples/crossing/truth/vid_91.gif" width="100%"></td>
    <td align="center"><img src="./public/github_examples/crossing/pred/vid_91.gif" width="100%"></td> 
  </tr>
  <tr>
    <td align="center"><img src="./public/github_examples/crossing/truth/vid_526.gif" width="100%"></td>
    <td align="center"><img src="./public/github_examples/crossing/pred/vid_526.gif" width="100%"></td> 
  </tr>
  <tr>
    <td align="center"><img src="./public/github_examples/crossing/truth/vid_429.gif" width="100%"></td>
    <td align="center"><img src="./public/github_examples/crossing/pred/vid_429.gif" width="100%"></td> 
  </tr>
</table>

### Classification Performance
[Rasouli2018](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w3/Rasouli_Are_They_Going_ICCV_2017_paper.pdf)

<p align="center">
<img src="./public/AP.png" border="0" width="40%" />
</p>

<p align="center">
<img src="./public/acc_scores.png" border="0" width="40%" />
</p>

### Run-Time Analysis

<p align="center">
<img src="./public/timings.png" border="0" width="10%" />
</p>


