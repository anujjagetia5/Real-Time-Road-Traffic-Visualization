import os
import sys
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
from config import cfg
import cv2
import json
import ipdb
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default='0')
    parser.add_argument('--img_folder', type=str, default='pedestrian_pose/')
    parser.add_argument('--output_folder', type=str, default='output')
    parser.add_argument('--encoder_setting', type=str, default='osx_l', choices=['osx_b', 'osx_l'])
    parser.add_argument('--decoder_setting', type=str, default='normal', choices=['normal', 'wo_face_decoder', 'wo_decoder'])
    parser.add_argument('--pretrained_model_path', type=str, default='../pretrained_models/osx_l.pth.tar')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    return args

args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

# load model
cfg.set_additional_args(encoder_setting=args.encoder_setting, decoder_setting=args.decoder_setting, pretrained_model_path=args.pretrained_model_path)
from common.base import Demoer
demoer = Demoer()
demoer._make_model()
from common.utils.preprocessing import load_img, process_bbox, generate_patch_image
from common.utils.vis import render_mesh, save_obj
from common.utils.human_models import smpl_x
model_path = args.pretrained_model_path
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
os.makedirs(args.output_folder, exist_ok=True)
demoer.model.eval()

# prepare input image
transform = transforms.ToTensor()
detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
repo = "isl-org/ZoeDepth"
model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_nk.to(device)

def calculate_center(x_min, y_min, x_max, y_max):
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return int(y_center), int(x_center)
results_centers = []
results_depths = []
filename_list = []
obj_list = []
bbox_list = []
# Function to extract the numeric part of the filename
def get_numeric_part(filename):
    return int(filename.split(".")[0])

filenames = os.listdir(args.img_folder)
# Sort the filenames based on the numeric part
sorted_filenames = sorted([filename for filename in filenames if filename.endswith(('.jpg', '.jpeg', '.png'))], key=get_numeric_part)

# Iterate over all files in the folder
for filename in sorted_filenames:
    # Construct the full path to the image file
    print(filename)
    image_path = os.path.join(args.img_folder, filename)
    original_img = load_img(image_path)
    original_img_height, original_img_width = original_img.shape[:2]
    # detect human bbox with yolov5s
    with torch.no_grad():
        results = detector(original_img)
    person_results = results.xyxy[0][results.xyxy[0][:, 5] == 0]
    depth_zoe = zoe.infer_pil(original_img)
    class_ids, confidences, boxes, center_for_detected_objs, depths_for_detected_objs, bbox_tmp = [], [], [], [], [], []
    for detection in person_results:
        x1, y1, x2, y2, confidence, class_id = detection.tolist()
        center = calculate_center(x1, y1, x1, y2)
        center_for_detected_objs.append(center)
        class_ids.append(class_id)
        confidences.append(confidence)
        box = [x1, y1, x2, y2,]
        bbox_tmp.append(box)
        depth_of_label = depth_zoe[center]
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        depths_for_detected_objs.append(int(depth_of_label))
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    

    for num, indice in enumerate(indices):
        bbox = boxes[indice]  # x,y,h,w
        bbox = process_bbox(bbox, original_img_width, original_img_height)
        final_center = center_for_detected_objs[indice]
        final_depth = depths_for_detected_objs[indice]
        bbox_final = bbox_tmp[indice]
        img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
        img = transform(img.astype(np.float32))/255
        img = img.cuda()[None,:,:,:]
        inputs = {'img': img}
        targets = {}
        meta_info = {}
        # mesh recovery
        with torch.no_grad():
            out = demoer.model(inputs, targets, meta_info, 'test')
        mesh = out['smplx_mesh_cam'].detach().cpu().numpy()
        mesh = mesh[0]
        # save mesh
        save_obj(mesh, smpl_x.face, os.path.join(args.output_folder, f'{filename}_person_{num}.obj'))
        results_centers.append(final_center)
        results_depths.append(final_depth)
        filename_list.append(filename)
        obj_list.append(f'{filename}_person_{num}.obj')
        bbox_list.append(bbox_final)
    


# Create a dictionary to store the data_json
data_json = {}

# Populate the dictionary
for filename, center, depth, obj_name, bounding_box in zip(filename_list, results_centers, results_depths, obj_list, bbox_list):
    if filename not in data_json:
        data_json[filename] = []
    data_json[filename].append({
        'center': center,
        'depth': depth,
        'obj_name': obj_name,
        'Bounding_box': bounding_box
    })

# Write the dictionary to a JSON file
with open('output_motorcycle.json', 'w') as json_file:
    json.dump(data_json, json_file, indent=4)
