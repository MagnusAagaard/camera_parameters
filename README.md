# camera_parameters
Repository containing code for generating projection matrix/affine transform between RGB and depth camera given corresponding points in the two cameras. It uses DLT algorithm to compute the transform between the two cameras. 3D world points are given from skeleton data with the corresponding 2D RGB coordinate and 2D depth coordinate.
When the transform has been estimated, new 3D points can be estimated by transforming from RGB frame to depth frame, and find the depth at the corresponding pixel location.
