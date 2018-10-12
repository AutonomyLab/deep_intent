# Classifying Pedestrian Actions In Advance Using Predicted Video of Urban Driving Scenes

## Abstract

We explore prediction of urban pedestrian actions by generating a video future of the traffic scene, and show promising results in classifying pedestrian behaviour before it is observed. We compare several encoder-decoder network models that predict 16 frames (400-600 milliseconds of video) from the preceding 16 frames. Our main contribution is a method for learning a sequence of representations to iteratively transform features learnt from the input to the future. Then we use a binary action classifier network for determining a pedestrian’s crossing intent from predicted video. Our results show an average precision of 81%, significantly higher than previous methods. The model with best classification performance runs for 117 ms on commodity GPU, giving an effective look-ahead of 416 ms. 

[Project Website](http://autonomy.cs.sfu.ca/deep_intent/)

