"""
This file is to test the performance of the fall detection system.
The model is trained by the Le2i Fall Detection Dataset where the situation is Home_01 and Coffee room_01. 
We test the model with Home_02 and Coffee room_02.
"""


import os
import argparse
import cv2
import torch
import pandas as pd
import numpy as np

from Detection.Utils import ResizePadding
from DetectorLoader import TinyYOLOv3_onecls
from PoseEstimateLoader import SPPE_FastPose
from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

# Create .csv file to store the label of the frame
# TODO 利用标注的人物框的信息，摒弃追踪的步骤
def create_csv(video_folder, anno_folder, csv_file='file.csv'):
    list_file = sorted(os.listdir(video_folder))
    list_anno = sorted(os.listdir(anno_folder))
    cols = ['video', 'frame', 'label']
    df = pd.DataFrame(columns=cols)
    for file, anno in zip(list_file, list_anno):
        anno_file = open(os.path.join(anno_folder, anno))
        fall_start = int(anno_file.readline().strip())
        fall_stop = int(anno_file.readline().strip())
        cap = cv2.VideoCapture(os.path.join(video_folder, file))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video = np.array([file] * frame_count)
        frame = np.arange(1, frame_count + 1)
        label = np.array([0] * frame_count)
        label[fall_start - 1:fall_stop] = 1
        rows = np.stack([video, frame, label], axis=1)
        df = df.append(pd.DataFrame(rows, columns=cols),
                        ignore_index=True)
        cap.release()
    df.to_csv(os.path.join(anno_folder, csv_file), index=False)


def preproc(image):
    """preprocess function for CameraLoader.
    """
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human Fall Detection Test.")
    parser.add_argument('--root', type=str, default='/mnt/data/DATASET/FallDataset',
                            help='Root of the Dataset')
    parser.add_argument('--situation', type=str, default='Home',
                            help='situation of the fall dataset')
    parser.add_argument('--detection_input_size', type=int, default=384,
                            help='Size of input in detection model in square must be divisible by 32 (int).')
    parser.add_argument('--pose_input_size', type=str, default='224x160',
                            help='Size of input in pose model must be divisible by 32 (h, w)')
    parser.add_argument('--pose_backbone', type=str, default='resnet50',
                            help='Backbone model for SPPE FastPose model.')
    parser.add_argument('--save_out', type=str, default='',
                            help='Save the result to .csv file.')
    parser.add_argument('--device', type=str, default='cuda',
                            help='Device to run model on cpu or cuda.')
    args = parser.parse_args()

    device = args.device
    video_folder = os.path.join(args.root, args.situation + '_02', 'Videos')
    anno_folder = os.path.join(args.root, args.situation + '_02', 'Annotation_files')
    csv_file = args.situation + '.csv'

    # Load .csv file to get the label of each frame of video
    if not os.path.exists(os.path.join(anno_folder, csv_file)):
        create_csv(video_folder, anno_folder, csv_file)
    annotation = pd.read_csv(os.path.join(anno_folder, csv_file))

    # DETECTION MODEL.
    inp_dets = args.detection_input_size
    resize_fn = ResizePadding(inp_dets, inp_dets)
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    # POSE MODEL.
    inp_pose = args.pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=device)

    # Tracker.
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    # Actions Estimate.
    action_model = TSSTG(device=device)

    # Write TP FP TN FN of the detection result in a .csv file
    cols = ['video', 'TP', 'FP', 'TN', 'FN']
    df = pd.DataFrame(columns=cols)
    videos = []
    tps = []
    fps = []
    tns = []
    fns = []

    video_list = annotation.iloc[:, 0].unique()
    for video in video_list:
        video_file = os.path.join(video_folder, video)
        annot = annotation[annotation['video'] == video].reset_index(drop=True)
        frame_idx = annot.iloc[:, 1].tolist()

        cam = cv2.VideoCapture(video_file)
        if int(cam.get(cv2.CAP_PROP_FRAME_COUNT)) != len(frame_idx):
            continue
        f = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        while True:
            ret, frame = cam.read()
            if ret:
                cls_label = annot.iloc[f, -1]
                frame = preproc(frame)
                # Detect humans bbox in the frame with detector model.
                detected = detect_model.detect(frame, need_resize=False, expand_bb=10)
                # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
                tracker.predict()
                # Merge two source of predicted bbox together.
                for track in tracker.tracks:
                    det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
                    detected = torch.cat([detected, det], dim=0) if detected is not None else det
                
                detections = []
                if detected is not None:
                    # Predict skeleton pose of each bboxs.
                    poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])
                    # Create Detections object.
                    detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                            np.concatenate((ps['keypoints'].numpy(),
                                                            ps['kp_score'].numpy()), axis=1),
                                            ps['kp_score'].mean().numpy()) for ps in poses]

                tracker.update(detections)
                predict = False
                for i, track in enumerate(tracker.tracks):
                    if not track.is_confirmed():
                        continue
                    
                    if predict:
                        break

                    track_id = track.track_id
                    # Use 30 frames time-steps to prediction.
                    if len(track.keypoints_list) == 30:
                        pts = np.array(track.keypoints_list, dtype=np.float32)
                        out = action_model.predict(pts, frame.shape[:2])
                        action_name = action_model.class_names[out[0].argmax()]
                        if action_name == 'Fall Down' and cls_label == 1:
                            tp += 1
                        elif action_name == 'Fall Down' and cls_label == 0:
                            fp += 1
                        elif action_name != 'Fall Down' and cls_label == 0:
                            tn += 1
                        elif action_name != 'Fall Down' and cls_label == 1:
                            fn += 1
                        predict = True
                f += 1
            else:
                break
        videos.append(video)
        tps.append(tp)
        fps.append(fp)
        tns.append(tn)
        fns.append(fn)

        cam.release()
    rows = np.stack([np.array(videos), np.array(tps), np.array(fps), np.array(tns), np.array(fns)], axis=1)
    df = df.append(pd.DataFrame(rows, columns=cols), ignore_index=True)
    if args.save_out != '' and not os.path.exists(os.path.join(anno_folder, args.save_out)):
        result_file = os.path.join(anno_folder, args.save_out)
        df.to_csv(result_file, index=False)


