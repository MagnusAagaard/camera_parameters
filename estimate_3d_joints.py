#!/usr/bin/python3
import os
import glob
import sys
from timeit import default_timer as timer
import logging
import random
import cv2
import numpy as np
from read_skeleton import read_color_xy, read_depth_xy, read_xyz
import pickle
from numpy.lib.format import open_memmap
from tqdm import tqdm

##############################################
# Load CenterNet
CENTERNET_PATH = '/home/magnus/CenterNet/src/'
sys.path.insert(0, CENTERNET_PATH)

import _init_paths
from detectors.detector_factory import detector_factory
from opts import opts
MODEL_PATH = '/home/magnus/CenterNet/models/multi_pose_dla_3x.pth'
TASK = 'multi_pose' # 'multi_pose' for human pose estimation
opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
detector = detector_factory[opt.task](opt)
# CenterNet outputs 17 joints..
NUMBER_OF_JOINTS = 17
##############################################

class JointEstimator:
    def __init__(self, output_data_dir, rgb_data_dir, depth_data_dir, affine_transforms_dir, depth_cam_projection_matrix_path, numpy_data_out_folder):
        self.output_data_dir = output_data_dir
        self.rgb_data_dir = rgb_data_dir
        self.depth_data_dir = depth_data_dir
        self.affine_transforms_dir = affine_transforms_dir
        self.depth_cam_projection_matrix_path = depth_cam_projection_matrix_path
        self.numpy_data_out_folder = numpy_data_out_folder
        self.check_dirs()

    def check_dirs(self):
        if not os.path.exists(self.output_data_dir):
            os.makedirs(self.output_data_dir)
        
        self.file_list = []
        self.already_generated = []
        # Ignore videos that data is already generated for
        for fil in os.listdir(self.output_data_dir):
            self.already_generated.append(fil[:-4])
        print("Already generated: {} files.".format(len(self.already_generated)))
        # Get the rest of the videos that data needs to be generated for
        for fil in os.listdir(self.rgb_data_dir):
            if fil[:-8] not in self.already_generated:
                self.file_list.append(fil)
        print("Files left: {}".format(len(self.file_list)))
        self.total_number_of_files = len(self.already_generated) + len(self.file_list)

    def run(self, reprocess=False):
        # Run the 3D joint estimation
        # Start a timer to estimate time left
        start = timer()
        nb_files = 0
        for fil in self.file_list:
            rgb_path = self.rgb_data_dir + fil
            depth_path = self.depth_data_dir + fil[:4] + '/' + fil[:-8]
            affine_transform_path = self.affine_transforms_dir + fil[:8] + '.npy'
            depth_camera_matrix_path = self.depth_cam_projection_matrix_path + fil[:8] + '.npy'
            with open(affine_transform_path, 'rb') as f:
                affine_transform = np.load(f)
            with open(depth_camera_matrix_path, 'rb') as f:
                Pd = np.load(f)
            if reprocess:
                # Run reprocessing after all videos have been run through
                nb_persons = self.get_nb_person_in_action(fil[-3:])
                self.estimate_and_save_points(rgb_path, depth_path, affine_transform, Pd, nb_persons)
            else:
                # Run normal
                self.estimate_and_save_points(rgb_path, depth_path, affine_transform, Pd)
            self.already_generated.append(fil)
            files_generated = len(self.already_generated)
            nb_files += 1
            end = timer()
            print("Generating files.. Files generated: {0}/{1} - ETA: {2:.2f} min.".format(files_generated, self.total_number_of_files, ((end-start)/nb_files)*(self.total_number_of_files-files_generated)/60))

    def estimate_and_save_points(self, rgb_path, depth_path, affine_transform, Pd, n_persons=0):
        frames = []
        cap = cv2.VideoCapture(rgb_path)
        ret = True
        persons = 0
        # Debug flags used for estimate_points. They are used to check if the detector 
        # suddenly goes from detecting 1 to 2 persons or vice versa
        debug_flags = [True, True, True]
        while ret:
            ret, img = cap.read()
            if ret:
                if n_persons:
                    # Run reprocessing
                    persons = self.reprocess_estimate_points(img, persons, rgb_path, n_persons)
                else:
                    persons = self.estimate_points(img, persons, rgb_path, debug_flags)
                if persons.shape[0] != 0:
                    pers = persons
                    #If only 1 person is detected append with zeros
                    if persons.shape[0] == 1:
                        x = np.zeros((2, persons.shape[1], persons.shape[2]))
                        x[0,:,:] = persons[0,:,:]
                        pers = x
                    frames.append(pers)
                else:
                    persons = 0
        frames = np.asarray(frames)
        if frames.shape[1] > 2:
            frames = frames[:,:2,:,:]
            print("{} contained more than two detected bodies after frames created.".format(rgb_path[-28:-8]))
            logging.debug("{} contained more than two detected bodies after frames created.".format(rgb_path[-28:-8]))
        
        #Check nb of persons detected
        nb_persons = 1
        if np.any(frames[:,1,:,:]):
            nb_persons = 2
        
        frames_estimated_points = []
        # Iterate over all frames
        for j in range(frames.shape[0]):
            # Get the corresponding depth image
            depth_img = cv2.imread(depth_path + '/MDepth-%08d.png' % (j+1), cv2.IMREAD_ANYDEPTH)
            body_estimated_points = []
            # Iterate over all bodies (persons)
            for b in range(frames.shape[1]):
                estimated_points = []
                # Iterate over all joints
                for i in range(frames.shape[2]):
                    # Get x,y for frame j, person/body b, joint i
                    i_xy = frames[j,b,i,:]
                    # Use homogenous coordinates
                    i_xy = np.append(i_xy, [1])
                    # If only 1 person, body #2 is zero
                    if nb_persons == 1 and b == 1:
                        estimate = np.zeros(3)
                    else:
                        # Transform estimated x,y from RGB to depth image coords
                        estimate = np.dot(affine_transform, i_xy)
                    if estimate[2] != 0:
                        estimate /= estimate[2]
                    else:
                        estimate[0] = 0
                        estimate[1] = 0
                        estimate[2] = 0
                    # Estimate 3D world coords using the estimated projection matrix
                    Xest = np.dot(np.linalg.pinv(Pd), estimate)
                    if Xest[3] != 0:
                        Xest /= Xest[3]
                    else:
                        Xest[0] = 0
                        Xest[1] = 0
                        Xest[2] = 0
                        Xest[3] = 0
                    Xest = Xest[:3]
                    # Get unit vector
                    norm = np.linalg.norm(Xest)
                    if norm != 0:
                        Xest /= norm
                    # Get length of vector (depth) from the depth image
                    depthy = int(estimate[1])
                    depthx = int(estimate[0])
                    if depthy < 0:
                        #logging.debug("y out of bounds: {}".format(depthy))
                        depthy = 0
                        #print(estimate, b, nb_persons)
                    elif depthy > 423:
                        #logging.debug("y out of bounds: {}".format(depthy))
                        depthy = 423
                        #print(estimate, b, nb_persons)
                    if depthx < 0:
                        #logging.debug("x out of bounds: {}".format(depthx))
                        depthx = 0
                        #print(estimate, b, nb_persons)
                    elif depthx > 511:
                        #logging.debug("x out of bounds: {}".format(depthx))
                        depthx = 511
                        #print(estimate, b, nb_persons)
                    # Get depth in meters
                    try:
                        depth = depth_img[depthy, depthx]/1000
                    except:
                        depth = 5
                    # Scale unit vector with depth
                    Xest *= depth
                    # Save estimated 3D joint information
                    estimated_points.append(Xest)
                # Save all joint estimations for bodies
                body_estimated_points.append(estimated_points)
            # Save for all frames in video
            frames_estimated_points.append(body_estimated_points)
        # Transform estimated data to numpy array
        frames_estimated_points = np.asarray(frames_estimated_points)
        # Save data as numpy array. Filename is the corresponding video
        # Format: (frame, body, joint, xyz)
        with open(self.output_data_dir + rgb_path[-28:-8] + '.npy', 'wb') as f:
            np.save(f, frames_estimated_points)

    def estimate_points(self, img, lastpoints, rgb_path, debug_flags):
        # Run detector and get result
        detector.pause = False
        result = detector.run(img)['results']
        values = result.get(1)
        keypoints = []
        reverse_keypoints = []

        # Itterate through detected persons, use only those with 0.4 and higher prob.
        # We know for a fact that max 2 persons are in the frames, keep the 2 with highest prob.
        # Problem? : If person 1 is detected two times with higher prob on both than person 2
        for person in values:
            if person[4] > 0.4 and len(keypoints) < 2:
                points = np.array(person[5:], dtype=np.int32).reshape(NUMBER_OF_JOINTS, 2)
                keypoints.append(points)
                reverse_keypoints.insert(0, points)
            #problem should be caught here if more than two persons are detected, the probs can be checked
            elif person[4] > 0.4 and len(keypoints) >= 2:
                if debug_flags[0]:
                    print("{} more than two persons detected.".format(rgb_path[-28:-8]))
                    logging.debug("{} more than two persons detected.".format(rgb_path[-28:-8]))
                    debug_flags[0] = False
        keypoints = np.asarray(keypoints)
        reverse_keypoints = np.asarray(reverse_keypoints)

        # Check mismatch between nb of detected persons between first two frames
        if type(lastpoints) != int and debug_flags[2]:
            debug_flags[2] = False
            if keypoints.shape[0] != lastpoints.shape[0]:
                print("{} contained different number of bodies between frames (FIRST AND SECOND!). {} --> {}".format(rgb_path[-28:-8], lastpoints.shape[0], keypoints.shape[0]))
                logging.debug("{} contained different number of bodies between frames (FIRST AND SECOND!). {} --> {}".format(rgb_path[-28:-8], lastpoints.shape[0], keypoints.shape[0]))

        # Check mismatch between nb of detected persons
        # Only logs and prints once pr. video
        if type(lastpoints) != int and keypoints.shape[0] != 0:
            if keypoints.shape[0] != lastpoints.shape[0]:
                if debug_flags[1]:
                    print("{} contained different number of bodies between frames. {} --> {}".format(rgb_path[-28:-8], lastpoints.shape[0], keypoints.shape[0]))
                    logging.debug("{} contained different number of bodies between frames. {} --> {}".format(rgb_path[-28:-8], lastpoints.shape[0], keypoints.shape[0]))
                    if keypoints.shape[0] == 1:
                        logging.debug("Returning last detected keypoints..")
                    else:
                        logging.debug("Returning highest prob with shape {}".format(keypoints[np.newaxis,0,:,:].shape))
                    debug_flags[1] = False
                # From 2 bodies to one, return last detected bodies
                # Else from 1 to 2 bodies, return highest prob
                if keypoints.shape[0] == 1:
                    return lastpoints
                else:
                    return keypoints[np.newaxis,0,:,:]
            # Check if detected keypoints match the same person or switched
            # Switches if higest probability value between person 1 and person 2 switch
            dist = np.linalg.norm(keypoints - lastpoints)
            reverse_dist = np.linalg.norm(reverse_keypoints - lastpoints)
            if dist > reverse_dist:
                return reverse_keypoints
        return keypoints

    def reprocess_estimate_points(self, img, lastpoints, rgb_path, nb_persons):
        detector.pause = False
        result = detector.run(img)['results']
        values = result.get(1)
        keypoints = []
        reverse_keypoints = []

        if nb_persons == 1:
            # We know for a fact that only one person should be present
            # so only use the highest prob
            person = values[0]
            if person[4] > 0.4:
                points = np.array(person[5:], dtype=np.int32).reshape(NUMBER_OF_JOINTS, 2)
                keypoints.append(points)
            elif type(lastpoints) != int and lastpoints.shape[0] == 1:
                # If we could not detect the person with high enough prob return lastpoints
                return lastpoints
            else:
                # If there are no last points, ie. it fails detecting a person at the first frame
                # then return zeros
                points = np.zeros((NUMBER_OF_JOINTS, 2), dtype=np.int32)
                keypoints.append(points)
            keypoints = np.asarray(keypoints)
            return keypoints
        elif nb_persons == 2:
            # We know for a fact that there are two persons
            for person in values:
                # Check if two persons are detected with a high enough prob
                if person[4] > 0.4 and len(keypoints) < 2:
                    points = np.array(person[5:], dtype=np.int32).reshape(NUMBER_OF_JOINTS, 2)
                    keypoints.append(points)
                    reverse_keypoints.insert(0, points)
            # If two persons were not detected, add zeros for the other person (ie. person moved out of frame)
            if len(keypoints) != 2:
                points = np.zeros((NUMBER_OF_JOINTS, 2), dtype=np.int32)
                keypoints.append(points)
                reverse_keypoints.insert(0, points)
            # If no persons were detected, add zeros for the other person as well
            if len(keypoints) != 2:
                points = np.zeros((NUMBER_OF_JOINTS, 2), dtype=np.int32)
                keypoints.append(points)
                reverse_keypoints.insert(0, points)
            keypoints = np.asarray(keypoints)
            reverse = np.asarray(reverse_keypoints)
            if type(lastpoints) != int:
                # Check if order of persons have been reversed (highest probability switched between the two)
                dist = np.linalg.norm(keypoints - lastpoints)
                reverse_dist = np.linalg.norm(reverse_keypoints - lastpoints)
                if dist > reverse_dist:
                    return reverse_keypoints
                return keypoints

    def gen_data(self, data_path, out_path, benchmark, part):
        print("Running {}, {}".format(benchmark, part))
        # for cross subject dataset, according to original dataset
        training_subjects = [
        1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
        # for cross view dataset
        training_cameras = [2, 3]
        max_body = 2
        num_joint = NUMBER_OF_JOINTS
        max_frame = 300
        
        ignored_samples = []
        sample_name = []
        sample_label = []
        for filename in os.listdir(data_path):
            if filename in ignored_samples:
                continue
            action_class = int(
                filename[filename.find('A') + 1:filename.find('A') + 4])
            subject_id = int(
                filename[filename.find('P') + 1:filename.find('P') + 4])
            camera_id = int(
                filename[filename.find('C') + 1:filename.find('C') + 4])

            # Videos used for training are different depending on the desired dataset is
            # cross-view or cross-subject 
            if benchmark == 'cv':
                istraining = (camera_id in training_cameras)
            elif benchmark == 'cs':
                istraining = (subject_id in training_subjects)
            else:
                raise ValueError()

            if part == 'train':
                issample = istraining
            elif part == 'test':
                issample = not (istraining)
            else:
                raise ValueError()

            if issample:
                sample_name.append(filename)
                sample_label.append(action_class - 1)

        if part == 'train':
            # Use only 80% of train data for training, 20% for validation
            number_of_train_samples = int(len(sample_name)*0.8)
            train_list = random.sample(range(0,len(sample_name)),number_of_train_samples)
            val_list = [i for i in range(0,len(sample_name)) if i not in train_list]
            sample_name_train = [sample_name[i] for i in train_list]
            sample_label_train = [sample_label[i] for i in train_list]
            sample_name_val = [sample_name[i] for i in val_list]
            sample_label_val = [sample_label[i] for i in val_list]

            # Save information in pickle file for identifying training, validation and test data easily
            with open('{}/{}_label.pkl'.format(out_path, 'train'), 'wb') as f:
                pickle.dump((sample_name_train, list(sample_label_train)), f)
            # Save entire training data as a single numpy array.
            # Format: (video number, XYZ, frame, joint, person/body)
            fp_train = open_memmap(
                '{}/{}_data.npy'.format(out_path, 'train'),
                dtype='float32',
                mode='w+',
                shape=(len(sample_label_train), 3, max_frame, num_joint, max_body))

            for i in tqdm(range(len(sample_name_train))):
                s = sample_name_train[i]
                with open(os.path.join(data_path, s), 'rb') as f:
                    # Frames contains (frame, body, joint, xyz)
                    # Should be (xyz, frame, joint, body)
                    # ie. transpose(3, 0, 2, 1)
                    data = np.load(f).transpose(3, 0, 2, 1)
                    fp_train[i, :, 0:data.shape[1], :, :] = data
            # Do same thing for validation data
            with open('{}/{}_label.pkl'.format(out_path, 'val'), 'wb') as f:
                pickle.dump((sample_name_val, list(sample_label_val)), f)
            fp_val = open_memmap(
                '{}/{}_data.npy'.format(out_path, 'val'),
                dtype='float32',
                mode='w+',
                shape=(len(sample_label_val), 3, max_frame, num_joint, max_body))

            for i in tqdm(range(len(sample_name_val))):
                s = sample_name_val[i]
                with open(os.path.join(data_path, s), 'rb') as f:
                    # Frames contains (frame, body, joint, xyz)
                    # Should be (xyz, frame, joint, body)
                    # ie. transpose(3, 0, 2, 1)
                    data = np.load(f).transpose(3, 0, 2, 1)
                    fp_val[i, :, 0:data.shape[1], :, :] = data

        else:
            # Do same thing for test data
            with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
                pickle.dump((sample_name, list(sample_label)), f)

            fp = open_memmap(
                '{}/{}_data.npy'.format(out_path, part),
                dtype='float32',
                mode='w+',
                shape=(len(sample_label), 3, max_frame, num_joint, max_body))

            for i in tqdm(range(len(sample_name))):
                s = sample_name[i]
                with open(os.path.join(data_path, s), 'rb') as f:
                    # Frames contains (frame, body, joint, xyz)
                    # Should be (xyz, frame, joint, body)
                    # ie. transpose(3, 0, 2, 1)
                    data = np.load(f).transpose(3, 0, 2, 1)
                    fp[i, :, 0:data.shape[1], :, :] = data

    def get_nb_person_in_action(self, str):
        if int(str) < 50:
            return 1
        else:
            return 2

    def sort_log(self):
        lines = []
        with open('logfile.log', 'r') as f:
            lines = f.readlines()
        lines = [l[11:-1] for l in lines]
        lines_to_keep = []
        for l in lines:
            if l[:2] == 'S0':
                # Check if one or two persons should be in the action, based on action number
                nb_persons = self.get_nb_person_in_action(l[17:20])
                # If one person in action and detected persons went from two to one then keep the line
                if nb_persons == 1 and l.count('2 --> 1') and not l.count('(FIRST AND SECOND!)'):
                    lines_to_keep.append(l)
                # If two persons in action and detected persons went from one to two then keep the line
                elif nb_persons == 2 and l.count('1 --> 2') and not l.count('(FIRST AND SECOND!)'):
                    lines_to_keep.append(l)

        print('Sorting logfile, saving revised version. Number of videos to reprocess: {}'.format(len(lines_to_keep)))
        # Save revised logfile
        with open('logfile_revised.txt', 'w') as f:
            for l in lines_to_keep:
                f.write(l + '\n')

    def reprocess_from_log(self):
        # Fix faulty detections that detected one or two persons different between frames
        # Open the revised logfile
        with open('logfile_revised.txt', 'r') as f:
            lines = f.readlines()
        lines = [l[:-1] for l in lines]
        # Reset files to process
        self.file_list = []
        for l in lines:
            fil = l[:20]
            self.file_list.append(fil)
        print("Reprocessing videos. Number of videos that need reprocessing left: {}".format(len(self.file_list)))
        self.run(reprocess=True)

    def save_numpy_as_one(self):
        data_path = self.output_data_dir
        out_folder = self.numpy_data_out_folder
        benchmark = ['cs', 'cv']
        part = ['train', 'test']
        for b in benchmark:
            for p in part:
                out_path = os.path.join(out_folder, b)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                self.gen_data(data_path, out_path, benchmark=b, part=p)


if __name__ == "__main__":
    # PRIOR TO RUNNING THIS, THE AFFINE TRANSFORM AND PROJECTION MATRIX SHOULD HAVE BEEN ESTIMATED
    # USING "projection_matrices.py"
    # The logfile is used afterwards, so should NOT be disabled!
    logging.basicConfig(filename='logfile.log',level=logging.DEBUG)
    output_data_dir = '/home/magnus/VSCODIUM_projects/camera_parameters/data/generated_data/'
    rgb_data_dir = '/home/magnus/VSCODIUM_projects/camera_parameters/data/NTURGBD/rgb_videos/'
    depth_data_dir = '/home/magnus/VSCODIUM_projects/camera_parameters/data/NTURGBD/depth/'
    affine_transforms_dir = '/home/magnus/VSCODIUM_projects/camera_parameters/camera_matrices/affine_transforms/rgb_to_depth/'
    depth_cam_projection_matrix_path = '/home/magnus/VSCODIUM_projects/camera_parameters/camera_matrices/depth/'
    numpy_data_out_folder = '/home/magnus/VSCODIUM_projects/camera_parameters/data/numpy_dataset/'

    je = JointEstimator(output_data_dir, rgb_data_dir, depth_data_dir, affine_transforms_dir, depth_cam_projection_matrix_path, numpy_data_out_folder)
    # Run estimation
    je.run()
    # Use logfile to decide which videos that needs reprocessing
    je.sort_log()
    # Reprocess some videos based on the revised logfile (fix issues with multiple people)
    je.reprocess_from_log()
    # Save all generated data into a single numpy array, divided into training and test data, according to the original dataset
    je.save_numpy_as_one()