import numpy as np
import cv2

# Define the Cityscapes palette (B, G, R) for each class
cityscapes_palette = {
    0: [128, 64, 128],    # road
    1: [232, 35, 244],    # sidewalk
    2: [70, 70, 70],      # building
    3: [156, 102, 102],   # wall
    4: [153, 153, 190],   # fence
    5: [153, 153, 153],   # pole
    6: [30, 170, 250],    # traffic light
    7: [0, 220, 220],     # traffic sign
    8: [35, 142, 107],    # vegetation
    9: [152, 251, 152],   # terrain
    10: [180, 130, 0],    # sky
    11: [60, 20, 220],    # person
    12: [0, 0, 255],      # rider
    13: [142, 0, 0],      # car
    14: [70, 0, 0],       # truck
    15: [100, 60, 0],     # bus
    16: [100, 80, 0],     # train
    17: [230, 0, 0],      # motorcycle
    18: [32, 11, 119],    # bicycle
    255: [0, 0, 0]        # void
    # 7: [128, 64, 128],    # road
    # 8: [232, 35, 244],    # sidewalk
    # 11: [70, 70, 70],      # building
    # 12: [156, 102, 102],   # wall
    # 13: [153, 153, 190],   # fence
    # 17: [153, 153, 153],   # pole
    # 19: [30, 170, 250],    # traffic light
    # 20: [0, 220, 220],     # traffic sign
    # 21: [35, 142, 107],    # vegetation
    # 22: [152, 251, 152],   # terrain
    # 23: [180, 130, 0],    # sky
    # 24: [60, 20, 220],    # person
    # 25: [0, 0, 255],      # rider
    # 26: [142, 0, 0],      # car
    # 27: [70, 0, 0],       # truck
    # 28: [100, 60, 0],     # bus
    # 31: [100, 80, 0],     # train
    # 32: [230, 0, 0],      # motorcycle
    # 33: [32, 11, 119],    # bicycle
    # 0: [0, 0, 0]        # void
}

def convert_label_id_to_color(label_id_image):
    # Create an empty image with the same shape as label_id_image, but with 3 channels for BGR
    color_image = np.zeros((*label_id_image.shape, 3), dtype=np.uint8)
    print(label_id_image)

    for label_id, color in cityscapes_palette.items():
        print("label_id", label_id)
        print(color_image[label_id_image == label_id].shape)
        color_image[label_id_image == label_id] = color

    return color_image

# Load the label ID image
label_id_path = '../data/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png'
label_id_path = 'mmsegmentation/work_dirs/format_results/aachen_000000_000019_leftImg8bit.png'
label_id_path = 'path_to_output_train_id_image.png'
label_id_image = cv2.imread(label_id_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded properly
if label_id_image is None:
    raise ValueError("Error loading label ID image")

# Convert label ID image to color image
color_image = convert_label_id_to_color(label_id_image)

# Save or display the color image
output_path = 'seg_pred.png'
cv2.imwrite(output_path, color_image)
# cv2.imshow('Color Image', color_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
