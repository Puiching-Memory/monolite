import os
import cv2
import numpy as np
from tqdm import tqdm

def convert_kitti_to_yolo(root_dir, out_dir, class_map, split="train"):
    """
    KITTI format: type truncation occlusion alpha bbox_left bbox_top bbox_right bbox_bottom dimensions location rotation_y
    YOLO format: class_id x_center y_center width height (normalized)
    """
    image_dir = os.path.join(root_dir, "training/image_2")
    label_dir = os.path.join(root_dir, "training/label_2")
    split_file = os.path.join(root_dir, "ImageSets", f"{split}.txt")
    
    if not os.path.exists(split_file):
        print(f"Split file {split_file} not found, using all labels.")
        idx_list = [f.split(".")[0] for f in os.listdir(label_dir) if f.endswith(".txt")]
    else:
        with open(split_file, "r") as f:
            idx_list = [line.strip() for line in f.readlines()]
    
    # Create output directories
    os.makedirs(os.path.join(out_dir, f"labels/{split}"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, f"images/{split}"), exist_ok=True)
    
    print(f"Converting {split} split ({len(idx_list)} files)...")
    
    for idx in tqdm(idx_list):
        label_file = f"{idx}.txt"
        img_path = os.path.join(image_dir, f"{idx}.png")
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        label_path = os.path.join(label_dir, label_file)
        if not os.path.exists(label_path):
            continue
            
        with open(label_path, "r") as f:
            lines = f.readlines()
            
        yolo_labels = []
        for line in lines:
            parts = line.split()
            cls_type = parts[0]
            
            if cls_type not in class_map:
                continue
                
            cls_id = class_map[cls_type]
            
            # 2D Bounding Box (left, top, right, bottom)
            xmin = float(parts[4])
            ymin = float(parts[5])
            xmax = float(parts[6])
            ymax = float(parts[7])
            
            # Convert to YOLO format (center_x, center_y, width, height) normalized
            x_center = (xmin + xmax) / 2.0 / w
            y_center = (ymin + ymax) / 2.0 / h
            box_w = (xmax - xmin) / w
            box_h = (ymax - ymin) / h
            
            yolo_labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
            
        # Save YOLO labels
        if yolo_labels:
            with open(os.path.join(out_dir, f"labels/{split}/{idx}.txt"), "w") as f:
                f.write("\n".join(yolo_labels))
            
            # Symlink or copy image
            dst_img_path = os.path.join(out_dir, f"images/{split}/{idx}.png")
            if not os.path.exists(dst_img_path):
                # Using symbolic link to save space
                try:
                    os.symlink(img_path, dst_img_path)
                except OSError:
                    import shutil
                    shutil.copy(img_path, dst_img_path)

if __name__ == "__main__":
    KITTI_ROOT = "/desay120T/ct/dev/uid01955/data/KITTIDataset"
    OUTPUT_DIR = "/desay120T/ct/dev/uid01954/monolite/data/kitti_yolo"
    
    # Only map the common classes
    CLASS_MAP = {
        "Car": 0,
        "Pedestrian": 1,
        "Cyclist": 2,
        "Van": 0,
        "Truck": 0,
    }
    
    convert_kitti_to_yolo(KITTI_ROOT, OUTPUT_DIR, CLASS_MAP, split="train")
    convert_kitti_to_yolo(KITTI_ROOT, OUTPUT_DIR, CLASS_MAP, split="val")
    print(f"Done! YOLO dataset created at {OUTPUT_DIR}")
