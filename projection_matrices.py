#!/usr/bin/env python3
import os
import glob
import cv2
import numpy as np
from read_skeleton import read_color_xy, read_depth_xy, read_xyz
from multiprocessing import Pool
from timeit import default_timer as timer

class ProjectionMatrixEstimator:
    def __init__(self, skeleton_dir, cam_color_dir, cam_depth_dir, test_videos, target_cams):
        self.skele_dir = skeleton_dir
        self.cam_color_dir = cam_color_dir
        self.cam_depth_dir = cam_depth_dir
        self.test_videos = test_videos
        self.target_cams = target_cams

    def estimate_and_save_color(self):
        for target in self.target_cams:
            C = self.estimate_color_camera_matrix(self.skele_dir, target, self.test_videos)
            if not os.path.exists(self.cam_color_dir):
                os.makedirs(self.cam_color_dir)
            with open(self.cam_color_dir + target + '.npy', 'wb') as f:
                np.save(f,C)
            print("Estimated and saved {}".format(target))

    def estimate_and_save_depth(self):
        for target in self.target_cams:
            C = self.estimate_depth_camera_matrix(self.skele_dir, target, self.test_videos)
            if not os.path.exists(self.cam_depth_dir):
                os.makedirs(self.cam_depth_dir)
            with open(self.cam_depth_dir + target + '.npy', 'wb') as f:
                np.save(f,C)
            print("Estimated and saved {} depth camera matrix.".format(target))

    def estimate_color_camera_matrix(self, skele_dir, target_cam, test_frame):
        file_list = []
        frames = [f + '.skeleton' for f in test_frame]
        for file in os.listdir(skele_dir):
            # Do not use the test videos specified in test_frame
            if target_cam in file and file not in frames:
                file_list.append(file)
            elif target_cam in file and file in frames:
                # Found test video, not using it
                print("Found {} in frames.".format(file))
        i = 0
        A = 0
        b = 0
        for f in file_list:
            # Get x,y coords of joints in RGB video
            color_xy = read_color_xy(skele_dir+f)
            # Get corresponding depth coords
            color_xyz = read_xyz(skele_dir+f)
            if type(color_xy) != int and type(color_xyz) != int:
                # Fix data issue with nan values
                color_xy = color_xy[:,:2]
                color_xyz = color_xyz[:,:3]
                if not np.isnan(color_xy).any() and not np.isnan(color_xyz).any():
                    if i == 0:
                        # First time
                        A = color_xyz
                        b = color_xy
                        i = 1
                    else:
                        # Add extra data to A and b matrices
                        A = np.concatenate((A,color_xyz))
                        b = np.concatenate((b, color_xy))
        # Solve for the projection matrix using A and b matrices
        C = self.DLT(A,b)

        return C

    def estimate_depth_camera_matrix(self, skele_dir, target_cam, test_frame):
        file_list = []
        frames = [f + '.skeleton' for f in test_frame]
        for file in os.listdir(skele_dir):
            # Do not use the test videos specified in test_frame
            if target_cam in file and file not in frames:
                file_list.append(file)
            elif target_cam in file and file in frames:
                # Found test video, not using it
                print("Found {} in frames.".format(file))
        i = 0
        A = 0
        b = 0
        for f in file_list:
            # Get x,y coords of joints in RGB video
            depth_xy = read_depth_xy(skele_dir+f)
            # Get corresponding depth coords
            depth_xyz = read_xyz(skele_dir+f)
            if type(depth_xy) != int and type(depth_xyz) != int:
                # Fix data issue with nan values
                depth_xy = depth_xy[:,:2]
                depth_xyz = depth_xyz[:,:3]
                if not np.isnan(depth_xy).any() and not np.isnan(depth_xyz).any():
                    if i == 0:
                        # First time
                        A = depth_xyz
                        b = depth_xy
                        i = 1
                    else:
                        # Add extra data to A and b matrices
                        A = np.concatenate((A, depth_xyz))
                        b = np.concatenate((b, depth_xy))
        # Solve for the projection matrix using A and b matrices
        C = self.DLT(A,b)

        return C

    def Normalization(self, nd, x):
        '''
        Normalization of coordinates (centroid to the origin and mean distance of sqrt(2 or 3).

        Input
        -----
        nd: number of dimensions, 3 here
        x: the data to be normalized (directions at different columns and points at rows)
        Output
        ------
        Tr: the transformation matrix (translation plus scaling)
        x: the transformed data
        '''

        x = np.asarray(x)
        m, s = np.mean(x, 0), np.std(x)
        if nd == 2:
            Tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
        else:
            Tr = np.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])
            
        Tr = np.linalg.inv(Tr)
        x = np.dot( Tr, np.concatenate( (x.T, np.ones((1,x.shape[0]))) ) )
        x = x[0:nd, :].T

        return Tr, x


    def DLTcalib(self, nd, xyz, uv):
        '''
        Camera calibration by DLT using known object points and their image points.

        Input
        -----
        nd: dimensions of the object space, 3 here.
        xyz: coordinates in the object 3D space.
        uv: coordinates in the image 2D space.

        The coordinates (x,y,z and u,v) are given as columns and the different points as rows.

        There must be at least 6 calibration points for the 3D DLT.

        Output
        ------
        L: array of 11 parameters of the calibration matrix.
        err: error of the DLT (mean residual of the DLT transformation in units of camera coordinates).
        '''
        if (nd != 3):
            raise ValueError('%dD DLT unsupported.' %(nd))
        
        # Converting all variables to numpy array
        xyz = np.asarray(xyz)
        uv = np.asarray(uv)

        n = xyz.shape[0]

        # Validating the parameters:
        if uv.shape[0] != n:
            raise ValueError('Object (%d points) and image (%d points) have different number of points.' %(n, uv.shape[0]))

        if (xyz.shape[1] != 3):
            raise ValueError('Incorrect number of coordinates (%d) for %dD DLT (it should be %d).' %(xyz.shape[1],nd,nd))

        if (n < 6):
            raise ValueError('%dD DLT requires at least %d calibration points. Only %d points were entered.' %(nd, 2*nd, n))
            
        # Normalize the data to improve the DLT quality (DLT is dependent of the system of coordinates).
        # This is relevant when there is a considerable perspective distortion.
        # Normalization: mean position at origin and mean distance equals to 1 at each direction.
        Txyz, xyzn = self.Normalization(nd, xyz)
        Tuv, uvn = self.Normalization(2, uv)

        A = []

        for i in range(n):
            x, y, z = xyzn[i, 0], xyzn[i, 1], xyzn[i, 2]
            u, v = uvn[i, 0], uvn[i, 1]
            A.append( [x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u] )
            A.append( [0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v] )

        # Convert A to array
        A = np.asarray(A) 

        # Find the 11 parameters:
        U, S, V = np.linalg.svd(A, full_matrices=False)

        # The parameters are in the last line of Vh and normalize them
        L = V[-1, :] / V[-1, -1]
        #print(L)
        # Camera projection matrix
        H = L.reshape(3, nd + 1)
        #print(H)

        # Denormalization
        # pinv: Moore-Penrose pseudo-inverse of a matrix, generalized inverse of a matrix using its SVD
        H = np.dot( np.dot( np.linalg.pinv(Tuv), H ), Txyz )
        #print(H)
        H = H / H[-1, -1]
        #print(H)
        #L = H.flatten()
        #print(L)

        # Mean error of the DLT (mean residual of the DLT transformation in units of camera coordinates):
        uv2 = np.dot( H, np.concatenate( (xyz.T, np.ones((1, xyz.shape[0]))) ) ) 
        uv2 = uv2 / uv2[2, :] 
        # Mean distance:
        err = np.sqrt( np.mean(np.sum( (uv2[0:2, :].T - uv)**2, 1)) ) 

        return H, err

    def DLT(self, xyz, uv):
        nd = 3
        P, err = self.DLTcalib(nd, xyz, uv)
        print('Matrix')
        print(P)
        print('\nError')
        print(err)

        return P


class AfineTransformEstimator:
    def __init__(self, skeleton_dir, test_videos, target_cams, save_dir, depth_dir, use_multiprocessing=False):
        self.skele_dir = skeleton_dir
        self.test_videos = test_videos
        self.target_cams = target_cams
        self.save_dir = save_dir
        self.depth_dir = depth_dir
        self.use_multiprocessing = use_multiprocessing
    
    def estimate_and_save_affine_transforms(self, tar_cam=None):
        if self.use_multiprocessing:
            if tar_cam != None:
                C = self.estimate_affine_transform(self.test_videos, self.skele_dir, tar_cam)
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                with open(self.save_dir + tar_cam + '.npy', 'wb') as f:
                    np.save(f,C)
                print("Estimated and saved {} depth camera matrix.".format(tar_cam))
            else:
                print("Tried to use multiprocessing, but gave no target cameras as parameter input.")
                return
        else:
            for target in self.target_cams:
                C = self.estimate_affine_transform(self.test_videos, self.skele_dir, target)
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                with open(self.save_dir + target + '.npy', 'wb') as f:
                    np.save(f,C)
                print("Estimated and saved {} depth camera matrix.".format(target))

    def draw_points_on_depth_videos(self):
        for idx, test_frame in enumerate(self.test_videos):
            cap = cv2.VideoCapture(self.depth_dir + '{}/{}/MDepth-%08d.png'.format(test_frame[:4], test_frame))
            frame_width = 512
            frame_height = 424
            if not os.path.exists('./output'):
                os.mkdir('./output')
            out = cv2.VideoWriter('./output/' + test_frame + '_depth.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
            real_xy = read_color_xy(self.skele_dir + test_frame + '.skeleton')
            real_xy_depth = read_depth_xy(self.skele_dir + test_frame + '.skeleton')
            with open(self.save_dir + self.target_cams[idx] + '.npy', 'rb') as f:
                affine_transform = np.load(f)
            
            i = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if ret == True:
                    # To better see the depth values in the video
                    frame *= 5
                    for j in range(25):
                        estimate = np.dot(affine_transform, real_xy[25*i+j])
                        if estimate.shape[0] == 3:
                            if estimate[2] == 0:
                                estimatex = 0
                                estimatey = 0
                            else:
                                estimate /= estimate[2]
                        estimatex = int(estimate[0])
                        estimatey = int(estimate[1])
                        realx = int(real_xy_depth[25*i+j][0])
                        realy = int(real_xy_depth[25*i+j][1])
                        cv2.circle(frame, (realx,realy), 2, (0,255,0), 1)
                        cv2.circle(frame, (estimatex, estimatey), 2, (0,0,255), 1)
                    out.write(frame)
                    #cv2.imshow('Frame', frame)
                    i += 1
                    #if cv2.waitKey(25) & 0xFF == ord('q'):
                    #    break
                else:
                    break
                
            cap.release()
            out.release()
            cv2.destroyAllWindows()

    def estimate_affine_transform(self, test_frame, skele_dir, target_cam):
        file_list = []
        frames = [f + '.skeleton' for f in test_frame]
        for file in os.listdir(skele_dir):
            # Do not use test videos for estimating the affine transform
            if target_cam in file and file not in frames:
                file_list.append(file)
            elif target_cam in file and file in frames:
                # Found test video in frames, not using it..
                print("Found {} in frames.".format(file))
        i = 0
        src = 0
        dst = 0
        for f in file_list:
            # Get x,y coord for joint in RGB video
            color_xy = read_color_xy(skele_dir+f)
            # Get corresponding depth x,y
            depth_xy = read_depth_xy(skele_dir+f)
            # Avoid data issues
            if type(color_xy) != int and type(depth_xy) != int:
                color_xy = color_xy[:,:2]
                depth_xy = depth_xy[:,:2]
                # Check that no nan values are present, else discard the video
                if not np.isnan(color_xy).any() and not np.isnan(depth_xy).any():
                    if i == 0:
                        src = color_xy
                        dst = depth_xy
                        i = 1
                    else:
                        src = np.concatenate((src,color_xy))
                        dst = np.concatenate((dst, depth_xy))
        print("Using {} data points for the DLT estimate".format(src.shape[0]))
        C = self.DLT(src, dst)

        return C

    def Normalization(self, nd, x):
        '''
        Normalization of coordinates (centroid to the origin and mean distance of sqrt(2 or 3).

        Input
        -----
        nd: number of dimensions, 3 here
        x: the data to be normalized (directions at different columns and points at rows)
        Output
        ------
        Tr: the transformation matrix (translation plus scaling)
        x: the transformed data
        '''

        x = np.asarray(x)
        m, s = np.mean(x, 0), np.std(x)
        if nd == 2:
            Tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
        else:
            Tr = np.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])
            
        Tr = np.linalg.inv(Tr)
        x = np.dot( Tr, np.concatenate( (x.T, np.ones((1,x.shape[0]))) ) )
        x = x[0:nd, :].T

        return Tr, x


    def DLTcalib(self, nd, xyz, uv):
        '''
        Camera calibration by DLT using known object points and their image points.

        Input
        -----
        nd: dimensions of the object space, 3 here.
        xyz: coordinates in the object 3D space.
        uv: coordinates in the image 2D space.

        The coordinates (x,y,z and u,v) are given as columns and the different points as rows.

        There must be at least 6 calibration points for the 3D DLT.

        Output
        ------
        L: array of 11 parameters of the calibration matrix.
        err: error of the DLT (mean residual of the DLT transformation in units of camera coordinates).
        '''
        if (nd != 2):
            raise ValueError('%dD DLT unsupported.' %(nd))
        
        # Converting all variables to numpy array
        xyz = np.asarray(xyz)
        uv = np.asarray(uv)

        n = xyz.shape[0]

        # Validating the parameters:
        if uv.shape[0] != n:
            raise ValueError('Object (%d points) and image (%d points) have different number of points.' %(n, uv.shape[0]))

        if (xyz.shape[1] != 2):
            raise ValueError('Incorrect number of coordinates (%d) for %dD DLT (it should be %d).' %(xyz.shape[1],nd,nd))

        if (n < 6):
            raise ValueError('%dD DLT requires at least %d calibration points. Only %d points were entered.' %(nd, 2*nd, n))
            
        # Normalize the data to improve the DLT quality (DLT is dependent of the system of coordinates).
        # This is relevant when there is a considerable perspective distortion.
        # Normalization: mean position at origin and mean distance equals to 1 at each direction.
        Txyz, xyzn = self.Normalization(nd, xyz)
        Tuv, uvn = self.Normalization(2, uv)

        A = []

        for i in range(n):
            x, y = xyzn[i, 0], xyzn[i, 1]
            u, v = uvn[i, 0], uvn[i, 1]
            A.append( [x, y, 1, 0, 0, 0, -u * x, -u * y, -u] )
            A.append( [0, 0, 0, x, y, 1, -v * x, -v * y, -v] )

        # Convert A to array
        A = np.asarray(A)

        # Find the 11 parameters:
        U, S, V = np.linalg.svd(A, full_matrices=False)

        # The parameters are in the last line of Vh and normalize them
        L = V[-1, :] / V[-1, -1]
        #print(L)
        # Camera projection matrix
        H = L.reshape(3, nd + 1)
        #print(H)

        # Denormalization
        # pinv: Moore-Penrose pseudo-inverse of a matrix, generalized inverse of a matrix using its SVD
        H = np.dot( np.dot( np.linalg.pinv(Tuv), H ), Txyz )
        #print(H)
        H = H / H[-1, -1]
        #print(H)
        #L = H.flatten()
        #print(L)

        # Mean error of the DLT (mean residual of the DLT transformation in units of camera coordinates):
        uv2 = np.dot( H, np.concatenate( (xyz.T, np.ones((1, xyz.shape[0]))) ) ) 
        uv2 = uv2 / uv2[2, :] 
        # Mean distance:
        err = np.sqrt( np.mean(np.sum( (uv2[0:2, :].T - uv)**2, 1)) ) 

        return H, err

    def DLT(self, xyz, uv):
        nd = 2
        P, err = self.DLTcalib(nd, xyz, uv)
        print('Matrix')
        print(P)
        print('\nError')
        print(err)

        return P


if __name__ == "__main__":
    # Skeleton directory should contain the original NTU-RBGD skeleton data.
    use_multiprocessing = True
    skeleton_directory = "./data/NTURGBD/skeleton/"
    depth_data_dir = './data/NTURGBD/depth/'
    # Output directories
    cam_color_dir = './test_camera_matrices/color/'
    cam_depth_dir = './test_camera_matrices/depth/'
    affine_transforms_rgb_to_depth_dir = './affine_transforms/rgb_to_depth/'
    # These videos are not used in the estimation of the projection matrices, but are used
    # for validating the result. One test video for each combination of scene and camera
    test_videos =[  "S001C001P007R001A023", "S001C002P008R002A007", "S001C003P003R002A020",
                    "S002C001P010R001A021", "S002C002P011R001A056", "S002C003P003R002A017",
                    "S003C001P001R001A016", "S003C002P019R002A003", "S003C003P001R002A031",
                    "S004C001P003R002A027", "S004C002P003R001A036", "S004C003P003R002A058",
                    "S005C001P017R002A014", "S005C002P016R001A048", "S005C003P004R001A032",
                    "S006C001P008R002A007", "S006C002P008R001A028", "S006C003P007R002A017",
                    "S007C001P017R002A021", "S007C002P001R002A006", "S007C003P007R001A010",
                    "S008C001P034R001A030", "S008C002P008R002A040", "S008C003P015R001A051",
                    "S009C001P019R001A047", "S009C002P017R001A023", "S009C003P016R002A051",
                    "S010C001P019R001A013", "S010C002P019R001A028", "S010C003P019R001A019",
                    "S011C001P038R001A046", "S011C002P007R001A044", "S011C003P019R002A046",
                    "S012C001P017R002A058", "S012C002P015R001A010", "S012C003P018R002A036",
                    "S013C001P017R002A044", "S013C002P025R002A020", "S013C003P016R002A041",
                    "S014C001P008R001A038", "S014C002P007R001A036", "S014C003P007R001A012",
                    "S015C001P017R001A011", "S015C002P016R001A022", "S015C003P025R001A032",
                    "S016C001P039R002A049", "S016C002P021R001A017", "S016C003P007R001A047",
                    "S017C001P015R001A023", "S017C002P017R001A008", "S017C003P009R001A045"
                    ]
    # Target cameras for the estimation
    target_cameras = [  "S001C001", "S001C002", "S001C003",
                        "S002C001", "S002C002", "S002C003",
                        "S003C001", "S003C002", "S003C003",
                        "S004C001", "S004C002", "S004C003",
                        "S005C001", "S005C002", "S005C003",
                        "S006C001", "S006C002", "S006C003",
                        "S007C001", "S007C002", "S007C003",
                        "S008C001", "S008C002", "S008C003",
                        "S009C001", "S009C002", "S009C003",
                        "S010C001", "S010C002", "S010C003",
                        "S011C001", "S011C002", "S011C003",
                        "S012C001", "S012C002", "S012C003",
                        "S013C001", "S013C002", "S013C003",
                        "S014C001", "S014C002", "S014C003",
                        "S015C001", "S015C002", "S015C003",
                        "S016C001", "S016C002", "S016C003",
                        "S017C001", "S017C002", "S017C003",
                    ]
    ate = AfineTransformEstimator(skeleton_directory, test_videos, target_cameras, affine_transforms_rgb_to_depth_dir, depth_data_dir, use_multiprocessing)
    # Estimate affine transforms between RGB and depth camera
    start = timer()
    if use_multiprocessing:
        with Pool(os.cpu_count()-1) as pool:
            pool.map(ate.estimate_and_save_affine_transforms, target_cameras)
    else:
        ate.estimate_and_save_affine_transforms()
    end = timer()
    print("Estimated in: {}".format(end-start))
    # Show result of affine transform. Points are shown on depth frames (green = GT, red = estimates).
    # The red and green dots should be more or less on top of each other for a good estimate.
    #ate.draw_points_on_depth_videos()
    # Calculate projection matrices. Only depth projection matrix is needed, as the x,y coordinate in the RGB frame
    # will be transformed to the x,y coordinate in the depth frame (using the just estimated affine transform).
    # The projection matrix will be used to estimate the epipolar line in the depth image at the estimate x,y coordinate
    # and the depth of this can be found from the depth image data
    #pme = ProjectionMatrixEstimator(skeleton_directory, cam_color_dir, cam_depth_dir, test_videos, target_cameras)
    #pme.estimate_and_save_color()
    #pme.estimate_and_save_depth()