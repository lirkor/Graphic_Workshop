# Graphic_Workshop

This Workshop is based on masked based content tranfer from here - https://github.com/rmokady/mbu-content-tansfer

The method for running is the same as above, yet this parameter can be added:

For the best variant:

    python full_net_gen.py <... params from basic mbu ... > --output_disc_weight 0.01 --noise 0.01 


For the latest (not the best) variant: 

    python full_net_gen.py <... params from basic mbu ... > --output_disc_weight 0.01 --noise 0.01 --stn_match_loss_const <> --stn_match_loss_decay <>


With stn_match_loss_decay, and stn_match_loss_const are the hyper-parameter used to make the match_loss coefficient.
