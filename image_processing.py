import os
import json
from tqdm import tqdm
from PIL import Image, ImageOps
import xml.etree.ElementTree as ET

def load_annotations(annotation_dir):
    """Load bounding box annotations from XML files in the given directory."""
    annotations = {}
    for xml_file in os.listdir(annotation_dir):
        if not xml_file.endswith('.xml'):
            continue
        xml_path = os.path.join(annotation_dir, xml_file)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            filename = root.find('filename').text
            # Iterate over all objects in the XML.
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name.lower() == 'waldo':
                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    annotations[filename] = {"bbox": [xmin, ymin, xmax, ymax]}
                    # If there's only one Waldo per image, break after finding it.
                    break
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
    return annotations

def crop_and_size_with_bbox(input_file_path, output_file_path, dimensions, annotations):
    """
    Crops and resizes images using ImageOps.fit and transforms the bounding box locations
    to match the cropped/resized image.
    
    Parameters:
        input_file_path (str): Directory containing original images.
        output_file_path (str): Directory to save cropped/resized images.
        dimensions (tuple): Target dimensions (width, height) for the output images.
        annotations (dict): A dictionary mapping image filenames to their original bounding box.
                            Format: { "image.jpg": { "bbox": [xmin, ymin, xmax, ymax] } }
    
    Returns:
        new_annotations (dict): Dictionary with updated bounding boxes for each image.
    """
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
    
    new_annotations = {}
    target_width, target_height = dimensions
    
    for image_filename in tqdm(os.listdir(input_file_path)):
        if image_filename.startswith('.'):
            continue
        
        img_path = os.path.join(input_file_path, image_filename)
        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"Error opening {image_filename}: {e}")
            continue
        
        orig_width, orig_height = img.size
        
        # Calculate scale factor used by ImageOps.fit
        scale = max(target_width / orig_width, target_height / orig_height)
        
        # New size after scaling
        new_width = orig_width * scale
        new_height = orig_height * scale
        
        # Compute crop offsets (assuming centering is (0.5, 0.5))
        offset_x = (new_width - target_width) / 2
        offset_y = (new_height - target_height) / 2
        
        # Perform the crop and resize
        cropped_and_sized = ImageOps.fit(img, dimensions, Image.LANCZOS)
        cropped_and_sized.save(os.path.join(output_file_path, image_filename), 'JPEG')
        
        # Transform the bounding box if available
        if image_filename in annotations and annotations[image_filename].get("bbox") is not None:
            orig_bbox = annotations[image_filename]["bbox"]  # [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = orig_bbox
            
            # Transform coordinates: scale then subtract the crop offset
            new_xmin = xmin * scale - offset_x
            new_ymin = ymin * scale - offset_y
            new_xmax = xmax * scale - offset_x
            new_ymax = ymax * scale - offset_y
            
            # Optionally clip the values so they remain within [0, target_width/height]
            new_xmin = max(0, new_xmin)
            new_ymin = max(0, new_ymin)
            new_xmax = min(target_width, new_xmax)
            new_ymax = min(target_height, new_ymax)
            
            new_annotations[image_filename] = {"bbox": [new_xmin, new_ymin, new_xmax, new_ymax]}
        else:
            new_annotations[image_filename] = {"bbox": None}
    
    return new_annotations


def adjust_bbox_for_patch(bbox, patch_origin, patch_size):
    """
    Adjusts a bounding box from the full image coordinates to the patch's coordinate system.
    
    Parameters:
        bbox (list or tuple): [xmin, ymin, xmax, ymax] in the full image.
        patch_origin (tuple): (patch_x, patch_y) of the top-left corner of the patch.
        patch_size (int): The size (width and height) of the square patch.
    
    Returns:
        list or None: The transformed bounding box [new_xmin, new_ymin, new_xmax, new_ymax]
                      if there's any overlap with the patch, or None if there is no overlap.
    """
    patch_x, patch_y = patch_origin
    x_min, y_min, x_max, y_max = bbox

    # Compute coordinates relative to the patch
    new_xmin = max(x_min - patch_x, 0)
    new_ymin = max(y_min - patch_y, 0)
    new_xmax = min(x_max - patch_x, patch_size)
    new_ymax = min(y_max - patch_y, patch_size)

    # Check if the bounding box has a valid area in the patch
    if new_xmin < new_xmax and new_ymin < new_ymax:
        return [new_xmin, new_ymin, new_xmax, new_ymax]
    else:
        return None


# import os
# from PIL import Image
# from tqdm import tqdm

# def chop_cropped_images(patch_size, input_dir, output_dir, annotations):
#     """
#     Chops cropped/resized images into patches of a given size and maps the updated bounding boxes
#     to the patches' coordinate systems.

#     Parameters:
#         patch_size (int): The width/height of the square patch (e.g. 256, 128, or 64).
#         input_dir (str): Directory containing the cropped/resized images.
#         output_dir (str): Directory where the patch images will be saved.
#         annotations (dict): Dictionary mapping image filenames to their updated bounding box.
#                             Format: { "image.jpg": {"bbox": [xmin, ymin, xmax, ymax]} }

#     Returns:
#         dict: A dictionary mapping patch filenames to their bounding box in patch coordinates,
#               e.g. { "image_0_0.jpg": {"bbox": [x_min, y_min, x_max, y_max]}, ... }
#               If the patch does not contain Waldo, "bbox" will be None.
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     patch_annotations = {}

#     # Process each image in the input directory.
#     for image_filename in tqdm(os.listdir(input_dir), desc=f"Chopping patches of size {patch_size}"):
#         if image_filename.startswith('.'):
#             continue
        
#         image_path = os.path.join(input_dir, image_filename)
#         try:
#             img = Image.open(image_path)
#         except Exception as e:
#             print(f"Error opening {image_filename}: {e}")
#             continue
        
#         img_width, img_height = img.size
#         num_x = img_width // patch_size
#         num_y = img_height // patch_size
        
#         # Process each patch.
#         for i in range(num_x):
#             for j in range(num_y):
#                 patch_origin = (i * patch_size, j * patch_size)
#                 patch_bbox = (patch_origin[0], patch_origin[1], patch_origin[0] + patch_size, patch_origin[1] + patch_size)
#                 patch_img = img.crop(patch_bbox)
                
#                 # Define a patch filename.
#                 patch_filename = f"{os.path.splitext(image_filename)[0]}_{i}_{j}.jpg"
#                 patch_path = os.path.join(output_dir, patch_filename)
#                 patch_img.save(patch_path, 'JPEG')
                
#                 # Map the bounding box (if any) to the patch.
#                 if image_filename in annotations and annotations[image_filename].get("bbox") is not None:
#                     orig_bbox = annotations[image_filename]["bbox"]  # [xmin, ymin, xmax, ymax]
#                     new_bbox = adjust_bbox_for_patch(orig_bbox, patch_origin, patch_size)
#                 else:
#                     new_bbox = None
                
#                 patch_annotations[patch_filename] = {"bbox": new_bbox}
    
#     return patch_annotations


import os
from PIL import Image
from tqdm import tqdm

import os
from PIL import Image
from tqdm import tqdm

def chop_cropped_images(patch_size, input_dir, output_dir, annotations):
    """
    Chops cropped/resized images into patches of a given size and maps the updated bounding boxes
    to the patches' coordinate systems, while also computing the width, height, and label for each patch.

    Parameters:
        patch_size (int): The width/height of the square patch (e.g. 256, 128, or 64).
        input_dir (str): Directory containing the cropped/resized images.
        output_dir (str): Directory where the patch images will be saved.
        annotations (dict): Dictionary mapping image filenames to their updated bounding box.
                            Format: { "image.jpg": {"bbox": [xmin, ymin, xmax, ymax]} }

    Returns:
        dict: A dictionary mapping patch filenames to a dictionary with:
              - "bbox": The bounding box in patch coordinates [x_min, y_min, x_max, y_max] or None.
              - "width": x_max - x_min if bbox exists, else 0.
              - "height": y_max - y_min if bbox exists, else 0.
              - "label": ["waldo"] if bbox is not None, else ["notwaldo"].
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    patch_annotations = {}

    # Process each image in the input directory.
    for image_filename in tqdm(os.listdir(input_dir), desc=f"Chopping patches of size {patch_size}"):
        if image_filename.startswith('.'):
            continue
        
        image_path = os.path.join(input_dir, image_filename)
        try:
            img = Image.open(image_path)
        except Exception as e:
            print(f"Error opening {image_filename}: {e}")
            continue
        
        img_width, img_height = img.size
        num_x = img_width // patch_size
        num_y = img_height // patch_size
        
        # Process each patch.
        for i in range(num_x):
            for j in range(num_y):
                patch_origin = (i * patch_size, j * patch_size)
                patch_bbox = (patch_origin[0], patch_origin[1], patch_origin[0] + patch_size, patch_origin[1] + patch_size)
                patch_img = img.crop(patch_bbox)
                
                # Define a patch filename.
                patch_filename = f"{os.path.splitext(image_filename)[0]}_{i}_{j}.jpg"
                patch_path = os.path.join(output_dir, patch_filename)
                patch_img.save(patch_path, 'JPEG')
                
                # Map the bounding box (if any) to the patch.
                if image_filename in annotations and annotations[image_filename].get("bbox") is not None:
                    orig_bbox = annotations[image_filename]["bbox"]  # [xmin, ymin, xmax, ymax]
                    new_bbox = adjust_bbox_for_patch(orig_bbox, patch_origin, patch_size)
                else:
                    new_bbox = None
                
                # Calculate width, height and assign label based on new_bbox.
                if new_bbox is not None:
                    x_min, y_min, x_max, y_max = new_bbox
                    width = x_max - x_min
                    height = y_max - y_min
                    label = ["waldo"]
                else:
                    width = 0
                    height = 0
                    label = ["notwaldo"]
                
                patch_annotations[patch_filename] = {
                    "bbox": new_bbox,
                    "width": width,
                    "height": height,
                    "label": label
                }
    
    return patch_annotations

# The helper function to adjust the bbox for a patch should be defined as:
def adjust_bbox_for_patch(bbox, patch_origin, patch_size):
    """
    Adjusts a bounding box from the full image coordinates to the patch's coordinate system.
    
    Parameters:
        bbox (list or tuple): [xmin, ymin, xmax, ymax] in the full image.
        patch_origin (tuple): (patch_x, patch_y) of the top-left corner of the patch.
        patch_size (int): The size (width and height) of the square patch.
    
    Returns:
        list or None: The transformed bounding box [new_xmin, new_ymin, new_xmax, new_ymax]
                      if there's any overlap with the patch, or None if there is no overlap.
    """
    patch_x, patch_y = patch_origin
    x_min, y_min, x_max, y_max = bbox

    # Compute coordinates relative to the patch
    new_xmin = max(x_min - patch_x, 0)
    new_ymin = max(y_min - patch_y, 0)
    new_xmax = min(x_max - patch_x, patch_size)
    new_ymax = min(y_max - patch_y, patch_size)

    # Check if the bounding box has a valid area in the patch
    if new_xmin < new_xmax and new_ymin < new_ymax:
        return [new_xmin, new_ymin, new_xmax, new_ymax]
    else:
        return None
