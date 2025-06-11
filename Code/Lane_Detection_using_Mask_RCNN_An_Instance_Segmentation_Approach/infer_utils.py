import cv2
import numpy as np
import torch
import ipdb
import json
from class_names import INSTANCE_CATEGORY_NAMES as coco_names
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
np.random.seed(2023)
def func(x, a,b,c,d):
    return a*x**3 + b*x**2 + c*x + d
# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))


def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 50, 50)
    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=2)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    return img_erode


def find_tip(points, convex_hull):
    length = len(points)
    indices = np.setdiff1d(range(length), convex_hull)
    for i in range(2):
        j = indices[i] + 2
        if j > length - 1:
            j = length - j
        if np.all(points[j] == points[indices[i - 1] - 2]):
            return tuple(points[j])

def find_orientation(road_sign,box):
    try:

        p1, p2 = box[0], box[1]   
        center_coord = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2  
        contours, hierarchy = cv2.findContours(preprocess(road_sign), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        arrow_tip = None
        for cnt in contours:
            if cnt.shape[0] < 100:
                return None, None, None
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            hull = cv2.convexHull(approx, returnPoints=False)
            sides = len(hull)
            defects = cv2.convexityDefects(cnt,hull)
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                cv2.line(road_sign,start,end,[0,255,0],2)
                cv2.circle(road_sign,far,5,[0,0,255],-1)
            if 8 > sides > 3 and sides + 2 == len(approx):
                arrow_tip = find_tip(approx[:,0,:], hull.squeeze())
            if arrow_tip:
                cv2.arrowedLine(road_sign, center_coord, arrow_tip, (255, 0, 0), 3, tipLength=0.4)
            cv2.drawContours(road_sign, [cnt], -1, (0, 255, 0), 3)
            # plt.imshow(road_sign)
            # plt.show()

        return arrow_tip, center_coord, cnt
    except:
        return None, None, None



def fitCurve(component_mask):	
    #fit a curve to the connected component using linear least squares
    points = np.argwhere(component_mask == True)
    x_pt , y_pt = points[:, 1], points[:, 0]
    region = (x_pt[0], x_pt[-1]) 
    popt, pcov = curve_fit(func, x_pt, y_pt, maxfev=5000)

    # here popt is the optimized parameters ie values of a, b, c	
    return popt, region


def lane_entry(mask,temp_im,lane_type):   
    ss= mask.flatten()
    if np.bincount(ss)[1] < 100:
        return None     
    curve, region = fitCurve(mask)
    x_mid = (region[0] + region[1]) // 2
    if x_mid < 640:
        side = "left"
    else:
        side = "right"
    def func(x, a,b,c,d):
        return a*x**3 + b*x**2 + c*x + d

    if side == "left":
        st = min(region[1], region[0])
        en = max(region[1], region[0])
    else:
        st = min(region[1], region[0])
        en = max(region[1], region[0])
    
    lane_types = ["solid-line", "dotted-line", "divider-line", "double-line"] 
    coordinates = []
    for x in range(st, en):	
        y = func(x, *curve)
        if y < 500 or y >= 960:
            continue
        if lane_type == "solid-line":
            temp_im = cv2.circle(temp_im, (x, int(y)), 3, (255, 255, 255), -1)
        elif lane_type == "dotted-line":    
            temp_im = cv2.circle(temp_im, (x, int(y)), 3, (255, 255, 0), -1)
        elif lane_type == "double-line":    
            temp_im = cv2.circle(temp_im, (x, int(y)), 3, (255, 0, 0), -1)
        else :
            temp_im = cv2.circle(temp_im, (x, int(y)), 3, (0, 255, 0), -1)
        coordinates.append((x, int(y)))

    return {"type": lane_type, "coordinates": coordinates}   


def get_outputs(image, model, threshold):
    with torch.no_grad():
        # forward pass of the image through the model.
        outputs = model(image)
    
    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    # get the masks
    masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    # discard masks for objects which are below threshold
    masks = masks[:thresholded_preds_count]

    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]
    # discard bounding boxes below threshold value
    boxes = boxes[:thresholded_preds_count]
    # get the classes labels
    labels = [coco_names[i] for i in outputs[0]['labels']]
    return masks, boxes, labels    

def convert_numpy_ints(data):
    if isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, list):
        return [convert_numpy_ints(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_numpy_ints(value) for key, value in data.items()}
    else:
        return data



def draw_segmentation_map(image, masks, boxes, labels, args, im):
    alpha = 1.0
    beta = 1.0 # transparency for the segmentation map
    gamma = 0.0 # scalar added to each sum
    #convert the original PIL image into NumPy format
    image = np.array(image)
    # convert from RGN to OpenCV BGR format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lane_types = ["solid-line", "dotted-line", "divider-line", "double-line"] 
    josn_dict = {}  
    josn_dict['frame_id'] = im	
    josn_dict['lanes'] = []
    lanes = []
    arrow_signs = []    
    for i in range(len(masks)):
        # apply a randon color mask to each object
        color = COLORS[coco_names.index(labels[i])]
        if masks[i].any() == True:
            if labels[i] in lane_types:
                lane = lane_entry(masks[i],image,labels[i]) 
                if lane:
                    lanes.append(lane)
            
            red_map = np.zeros_like(masks[i]).astype(np.uint8)
            green_map = np.zeros_like(masks[i]).astype(np.uint8)
            blue_map = np.zeros_like(masks[i]).astype(np.uint8)
            red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1] = color
            # combine all the masks into a single image
            segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
            if labels[i] == "road-sign-line":
                road_sign = cv2.cvtColor(masks[i].astype(np.uint8), cv2.COLOR_GRAY2RGB) * 255
                arrow_tip, center_coord, cnt = find_orientation(road_sign,boxes[i])   
                if cnt is None:
                    continue
                if arrow_tip:
                    cv2.arrowedLine(image, center_coord, arrow_tip, (255, 0, 0), 3, tipLength=0.4)     
                # plt.imsave(f"outputs/roadsign/" + str(frame_id) + ".png", road_sign) 
                cnt = cnt.squeeze().tolist()
                arrow_signs.append({"type": "road-sign-line", "center": center_coord, "box": boxes[i], "contour": cnt})     

                

            # apply mask on the image
            cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)

            lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # Line width.
            tf = max(lw - 1, 1) # Font thickness.
            p1, p2 = boxes[i][0], boxes[i][1]
            if not args.no_boxes:
                # draw the bounding boxes around the objects
                cv2.rectangle(
                    image, 
                    p1, p2, 
                    color=color, 
                    thickness=lw,
                    lineType=cv2.LINE_AA
                )
                w, h = cv2.getTextSize(
                    labels[i], 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=lw / 3, 
                    thickness=tf
                )[0]  # text width, height
                w = int(w - (0.20 * w))
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                # put the label text above the objects
                cv2.rectangle(
                    image, 
                    p1, 
                    p2, 
                    color=color, 
                    thickness=-1, 
                    lineType=cv2.LINE_AA
                )
                cv2.putText(
                    image, 
                    labels[i], 
                    (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=lw / 3.8, 
                    color=(255, 255, 255), 
                    thickness=tf, 
                    lineType=cv2.LINE_AA
                )
    # lanes = convert_numpy_ints(lanes)   
    # arrow_signs = convert_numpy_ints(arrow_signs)   
    josn_dict['lanes'] = lanes
    josn_dict['arrow_signs'] = arrow_signs
    return image, josn_dict

