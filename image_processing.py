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
