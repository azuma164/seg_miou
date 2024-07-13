import numpy as np
import cv2

def calculate_iou(pred, gt, num_classes):
    iou_list = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        gt_inds = gt == cls
        intersection = np.logical_and(pred_inds, gt_inds).sum()
        union = np.logical_or(pred_inds, gt_inds).sum()
        if union == 0:
            iou = float('nan')  # Ignore the class if there is no ground truth or prediction
        else:
            iou = intersection / union
        iou_list.append(iou)
    return iou_list

def calculate_miou(pred_path, gt_path, num_classes=19):
    # Load images
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    # gt = cv2.resize(gt, (1024, 512), interpolation=cv2.INTER_NEAREST)

    # Check if images are loaded properly
    if pred is None or gt is None:
        raise ValueError("Error loading images")

    # Calculate IoU for each class
    iou_list = calculate_iou(pred, gt, num_classes)

    # Calculate mIoU
    miou = np.nanmean(iou_list)
    return miou

# Example usage
pred_path = 'path_to_output_train_id_image.png'
gt_path = '../data/cityscapes/gtFine/train/aachen/aachen_000002_000019_gtFine_labelTrainIds.png'
# pred_path = gt_path
miou = calculate_miou(pred_path, gt_path)
print(f'mIoU: {miou}')
