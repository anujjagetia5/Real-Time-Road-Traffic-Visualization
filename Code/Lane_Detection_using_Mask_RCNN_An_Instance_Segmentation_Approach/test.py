import json
import ipdb
# Load your existing JSON data
with open("outputs\scene7lane.json", 'r') as f:
    data = json.load(f)
# ipdb.set_trace()
# for i in range(len(data[0]['lanes'])):
#     ipdb.set_trace()
#     selectedentry = data[0]['lanes'][i]
#     lane_type = selectedentry['type']
#     print(lane_type)
#     points = selectedentry['coordinates']
#     print(points)
# Function to extract data based on image name
# def extract_data(image_name):
    
image_name = "0.png"
for entry in data:
    if entry['frame_id'] == image_name:
        frame_id = entry['frame_id']
        ipdb.set_trace()
        print("Frame ID:", frame_id)
        # Iterate over lanes
        for lane in entry['lanes']:
            lane_type = lane['type']
            print("Type:", lane_type)
            # Print coordinates
            print("Coordinates:")
            print(lane['coordinates'])
                    

# Provide the image name you want to extract data for

# extract_data(image_name)