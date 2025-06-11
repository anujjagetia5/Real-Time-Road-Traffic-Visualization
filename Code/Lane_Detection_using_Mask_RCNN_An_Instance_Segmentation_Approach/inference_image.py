import torch
import torchvision
import cv2
import argparse
import numpy as np
import torch.nn as nn
import glob
import os
import matplotlib.pyplot as plt 
from PIL import Image
from infer_utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms
from class_names import INSTANCE_CATEGORY_NAMES as class_names
import ipdb 
parser = argparse.ArgumentParser()
import json 

def convert_numpy_ints(data):
    if isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, list):
        return [convert_numpy_ints(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_numpy_ints(value) for key, value in data.items()}
    else:
        return data
    

parser.add_argument(
    '-t', 
    '--threshold', 
    default=0.5, 
    type=float,
    help='score threshold for discarding detection'
)
parser.add_argument(
    '-w',
    '--weights',
    default='outputs/training/road_line/model_15.pth',
    help='path to the trained wieght file'
)
parser.add_argument(
    '--show',
    action='store_true',
    help='whether to visualize the results in real-time on screen'
)
parser.add_argument(
    '--no-boxes',
    action='store_true',
    help='do not show bounding boxes, only show segmentation map'
)
args = parser.parse_args()

OUT_DIR = os.path.join('outputs', 'scene10')
os.makedirs(OUT_DIR, exist_ok=True)

model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
    pretrained=False, num_classes=91
)

model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=len(class_names), bias=True)
model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=len(class_names)*4, bias=True)
model.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(256, len(class_names), kernel_size=(1, 1), stride=(1, 1))

# initialize the model
ckpt = torch.load(args.weights)
model.load_state_dict(ckpt['model'])
# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the modle on to the computation device and set to eval mode
model.to(device).eval()
print(model)

# transform to convert the image to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])
images_path = "case10/"
# dir_list = os.listdir(path)
#print(dir_list)
def get_numeric_part(filename):
    return int(filename.split(".")[0])

# frame_names = os.listdir(f"Cases/scene1/front_undistort/")
filenames = os.listdir(images_path)

# Sort the filenames based on the numeric part
sorted_filenames = sorted([filename for filename in filenames if filename.endswith(('.png'))], key=get_numeric_part)

frame_id = 0
entrie = []
total_list = len(sorted_filenames)
for iddx in range(0, total_list):
    im = sorted_filenames[iddx]
    file_path = os.path.join(images_path, im)
    # ipdb.set_trace()
    # print(image_path)
    # image = Image.open(file_path)
    image = cv2.imread(file_path)   
    # keep a copy of the original image for OpenCV functions and applying masks
    orig_image = image.copy()
    
    # transform the image
    image = transform(image)
    # add a batch dimension
    image = image.unsqueeze(0).to(device)
    
    masks, boxes, labels = get_outputs(image, model, args.threshold)
    # print(masks)
    # print(boxes)
    # print(labels)
    result, json_entry = draw_segmentation_map(orig_image, masks, boxes, labels, args, im)
    
    entrie.append(json_entry)
    # visualize the image
    if args.show:
        cv2.imshow('Segmented image', np.array(result))
        cv2.waitKey(1)
    # plt.imshow(result)  
    # plt.show()  
    # set the save path
    save_path = os.path.join(OUT_DIR, str(frame_id) + '.png')
    cv2.imwrite(save_path, result)
    frame_id += 1
    # break   
# ipdb.set_trace()
entries = convert_numpy_ints(entrie)
with open(OUT_DIR+'lane.json', 'w') as json_file:
    json.dump(entries, json_file)
