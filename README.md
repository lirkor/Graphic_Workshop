# Graphic_Workshop

This Workshop is based on masked based content tranfer from here - https://github.com/rmokady/mbu-content-tansfer

The method for running is the same as above, yet this parameter can be added:

For the best variant:

    python full_net_gen.py <... params from basic mbu ... > --output_disc_weight 0.01 --noise 0.01 


For the latest (not the best) variant: 

    python full_net_gen.py <... params from basic mbu ... > --output_disc_weight 0.01 --noise 0.01 --stn_match_loss_const <> --stn_match_loss_decay <>


With stn_match_loss_decay, and stn_match_loss_const are the hyper-parameter used to make the match_loss coefficient.

### Introduction 
The task of content transfer, involves identifying specific components of a given Input, and applying it to another input of some space (in our case, the same input space), in a semantically correct and visually convincing manner. For instance in [mbu-content-transfer](https://arxiv.org/abs/1906.06558) this was done for features such as glasses and mustaches that were identified on a certain input and then applied to a different input.

### Implementation
we will provide a quick overview of the methods we tried, for ease of explanation we add the scheme of the original implementation:

![flow image](https://lh3.googleusercontent.com/PEQriyOPH-S3BXqZmvkDEUsII1R_tyQGhOItmC9yQh2X3gboTo6ArBdbtgHsCgpsiHTTQwksSLz5-jmT9VpHnnNCagNVnrRgAW9sSnxkd0J-hzzRnjU41QntjEoFGD3XnFIxqU0)

Image 2: Original model scheme

1.  Applying single STN to the 4D tensor after the output ofDB – in this method we attempt to use the out but of DB as an input to an STN model in order to get an aligned mask.
    
2.  Incorporating STN with high amount of information as input (including both input images as well as their encodings), we attempt this approach in several different models.
    
3.  Multi-level STN – in this method we attempt to use several STN models and incorporate them at different points of the originally proposed model.
    
4.  Using flow-net for pixel by pixel flow generation – taking inspiration from video processing we attempt to approach this problem as an optical flow problem, using a pre-trained network.
    
5.  Multi-patch STN – in this method we attempt to incorporate STN models at different layers, which we refer to as patch- STN, connecting them with convolution. Each layer receives several patches it needs to process as an input. This method has gone through several different iterations and is explained thoroughly in the paper.

The best variant results:
