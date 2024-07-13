import numpy as np
import cv2

# Define the Cityscapes label IDs to Train IDs mapping
label_to_train_id = {
    0: 255,  # unlabeled
    1: 255,  # ego vehicle
    2: 255,  # rectification border
    3: 255,  # out of roi
    4: 255,  # static
    5: 255,  # dynamic
    6: 255,  # ground
    7: 0,    # road
    8: 1,    # sidewalk
    9: 255,  # parking
    10: 255, # rail track
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    14: 255, # guard rail
    15: 255, # bridge
    16: 255, # tunnel
    17: 5,   # pole
    18: 255, # polegroup
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    29: 255, # caravan
    30: 255, # trailer
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
    -1: 255, # license plate
}

def convert_label_id_to_train_id(label_id_image):
    # Create an empty image with the same shape as label_id_image
    train_id_image = np.zeros(label_id_image.shape, dtype=np.uint8)

    for label_id, train_id in label_to_train_id.items():
        train_id_image[label_id_image == label_id] = train_id

    return train_id_image

# Load the label ID image
label_id_path = '../data/cityscapes/gtFine/train/aachen/aachen_000002_000019_gtFine_labelIds.png'
label_id_path = 'mmsegmentation/work_dirs/format_results_mask2former/aachen_000002_000019_leftImg8bit.png'
label_id_image = cv2.imread(label_id_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded properly
if label_id_image is None:
    raise ValueError("Error loading label ID image")

# Convert label ID image to Train ID image
train_id_image = convert_label_id_to_train_id(label_id_image)

# Save or display the Train ID image
output_path = 'path_to_output_train_id_image.png'
cv2.imwrite(output_path, train_id_image)
# cv2.imshow('Train ID Image', train_id_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
