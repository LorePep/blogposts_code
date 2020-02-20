# Eight points algorithm

This tests the 8 point algorithms to compute the affine transformation between two synthetic cameras.

The code generate 2 cameras, 8 synthetic points in the world, project the points in the image spaces and then compute R and T.

Also images are generated to evaluate the process.

As an example

Generated camera frames and 3D points.
![3d world](https://github.com/LorePep/blogposts_code/blob/master/eight-points/3d_world.png)

Generated camera images.
![images](https://github.com/LorePep/blogposts_code/blob/master/eight-points/images.png)

Colors represent matching points.
