## Egocentric video relocalization pipeline

[comment]: <> (* Step 0: Follow Step 5 in the [note]&#40;https://hackmd.io/@vuong067/rJ11een9O&#41; to download and install the AMD64 version of MarsFramework.)

[comment]: <> (* Step 1: Prepare a raw scan folder and a raw egovideo folder. The sample data hierarchy can be downloaded [here]&#40;https://drive.google.com/file/d/17z2tHS7htlYrl4dw23TvODMcViqnkZ5-/view?usp=sharing&#41;.)

[comment]: <> (* Step 2: )

[comment]: <> (  Change the `MARSFRAMEWORK_PATH` in `main.sh` to the appropriate path of MarsFramework in your computer &#40;from Step 0&#41;. Run the following script for a relocalization of the egocentric video w.r.t the 3d scan mesh as:)

[comment]: <> (        `./main.sh ./sample_data/scan/raw/ kurtis_home ./sample_data/egovideo/raw/ kurtis_003`)

[comment]: <> (Like before, this will output 2 main files:)

[comment]: <> (        1. /path/to/egocentric_video/poses_reloc/camera_poses_pnp.npy)
   
[comment]: <> (This is an ndarray of size &#40;Nx3x4&#41; where N is the number of egocentric images which stores the transformation from scans to camera coordinate.)

[comment]: <> (        2. /path/to/egocentric_video/poses_reloc/good_poses_pnp.npy)
   
[comment]: <> (This is an ndarray of size &#40;N,&#41; where N is the number of egocentric images. )

[comment]: <> (* Step 3: Visualize the output with meshlab.)

[comment]: <> (Add the following file to meshlab:)

[comment]: <> (    1. /path/to/egocentric_video/poses_reloc/cameras_pnp.ply)

[comment]: <> (    2. /path/to/3dscans/matterpak/*.obj)
    
[comment]: <> (* Step 4: Additionally, the last Python script `Localization/visualize_pcd_pose_render_ba.py`, if ran successfully, will overlay the images capture from the mesh on top of the raw egovideo images &#40;to verify alignment&#41;. The output can be seen at: `/path/to/egocentric_video/poses_visualization`)

### Pose processing BLS pipeline
* Step 0: Follow Step 5 in the [note](https://hackmd.io/@vuong067/rJ11een9O) to download and install the AMD64 version of MarsFramework.
* Step 1: Prepare a raw scan folder and a raw egovideo folder. The sample data hierarchy can be downloaded [here](https://drive.google.com/file/d/1Uq5AlVMAbLA6G0fQCkOes-czrshWJnQ8/view?usp=sharing).

* Step 2: 
  Change the `MARSFRAMEWORK_PATH` in `main_bls_association.sh` to the appropriate path of MarsFramework in your computer (from Step 0). Run the following script for a relocalization of the egocentric video w.r.t the 3d scan mesh as:
        `./main_bls_association.sh ./sample_data_bls/raw_data/ ./sample_data_bls/processed_data/ ./sample_data_bls/scan_ego_pair.txt`

Like before, this will output 2 main files:

        1. /path/to/egocentric_video/poses_reloc/camera_poses_pnp.npy
   
This is an ndarray of size (Nx3x4) where N is the number of egocentric images which stores the transformation from scans to camera coordinate.

        2. /path/to/egocentric_video/poses_reloc/good_poses_pnp.npy
   
This is an ndarray of size (N,) where N is the number of egocentric images. 

* Step 3: Visualize the output with meshlab.
Add the following file to meshlab:
    1. /path/to/egocentric_video/poses_reloc/cameras_pnp.ply
    2. /path/to/3dscans/matterpak/*.obj
    
* Step 4: Additionally, the last Python script `Localization/visualize_bls.py`, if ran successfully, will overlay the images capture from the mesh on top of the raw egovideo images (to verify alignment), using the poses either from the relocalization pipeline or the BA pose transformed to the scan map. The output can be seen at: `/path/to/egocentric_video/pose_visualization` and `/path/to/egocentric_video/pose_visualization_bls`
