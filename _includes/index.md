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


### Future Prediction 
Our objective is to predict the future positions of salient objects like vehicles and pedestrians by learning their motion. Functionally, an encoder reads a sequence of frames __x__ to yield dense representations __z__. Conditioned on __z__, a decoder will then auto-regressively predict an image sequence __y'__ by minimizing a pixel-wise loss between __y'__ and ground truth frames __y__. Each generated frame is of the same resolution as the input. We reverse the temporal ordering of input data to condition the latent space with spatial information from the latest frame. The most recent frame carries forward the closest contextual resemblance. Recursively learning representations from each input frame, we expect to first learn a temporal regularity in the early representations and parametrize a temporal variance in the later representations.

<p align="center">
<img src="./public/abstract-net.svg" alt="abstract-net" border="0" /></a>
</p>

### Action Recognition
The task of action recognition is motivated by the idea that by looking ahead in time, we could react to a hazardous pedestrian interaction a little earlier, with safety benefits. We do this end-to-end by appending a binary action classifier to our future video generator. In this task, we want to learn to predict a pedestrian’s crossing intent across a multitude of crossing scenarios and behaviours.

<p align="center">
<img src="./public/abstract-net-cla.svg" alt="abstract-net" border="0" /></a>
</p>


### Experiments
We use the JAAD(http://data.nvision2.eecs.yorku.ca/JAAD_dataset/) dataset [paper](https://arxiv.org/abs/1609.04741) consisting of 346 high resolution videos in pedestrian interaction scenarios. We train our encoder-decoder stack to optimize for a combination of *l1* and *l2* losses. The losses are calculated between the *N* pixels of T predicted frames __y'__ and ground truth frames __y__. For video prediction experiments we set N = 128 × 208 and T = 16 frames.

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





