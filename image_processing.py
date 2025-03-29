import os
import json
from tqdm import tqdm
from PIL import Image, ImageOps
import xml.etree.ElementTree as ET


# Example: annotations.json format:
# {
#     "image1.jpg": { "bbox": [x_min, y_min, x_max, y_max] },
#     "image2.jpg": { "bbox": [x_min, y_min, x_max, y_max] }
# }

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

def adjust_bounding_box(original_bbox, patch_origin, patch_size):
    """
    Transforms the original bounding box (in original image coordinates) into the coordinate system
    of the patch.
    
    Parameters:
        original_bbox (list): [x_min, y_min, x_max, y_max] for the original image.
        patch_origin (tuple): (patch_x0, patch_y0) - top-left corner of the patch in the original image.
        patch_size (int): Size of the square patch (width and height).
        
    Returns:
        A list [rel_x_min, rel_y_min, rel_x_max, rel_y_max] if there is an intersection,
        or None if there is no overlap.
    """
    patch_x0, patch_y0 = patch_origin
    orig_x_min, orig_y_min, orig_x_max, orig_y_max = original_bbox

    # Calculate coordinates relative to the patch
    rel_x_min = max(orig_x_min - patch_x0, 0)
    rel_y_min = max(orig_y_min - patch_y0, 0)
    rel_x_max = min(orig_x_max - patch_x0, patch_size)
    rel_y_max = min(orig_y_max - patch_y0, patch_size)

    # Check if there's a valid (non-zero area) intersection.
    if rel_x_min < rel_x_max and rel_y_min < rel_y_max:
        return [rel_x_min, rel_y_min, rel_x_max, rel_y_max]
    else:
        return None

def chop_with_bbox(patch_size, input_file_path, output_file_path, annotation_file):
    """
    Chops the images into patches of given size and maps the original bounding box to each patch.
    
    Parameters:
        patch_size (int): The width and height of the square patch (e.g. 256, 128, 64).
        input_file_path (str): Folder containing the original (cropped and resized) images.
        output_file_path (str): Folder where the patch images will be saved.
        annotation_file (str): JSON file with original image bounding box annotations.
    """
    # Load bounding box annotations
    annotations = load_annotations(annotation_file)
    
    # Dictionary to hold the patch annotations.
    patch_annotations = {}

    # Create the output directory if it doesn't exist.
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    # Loop over each image in the input directory.
    for image in tqdm(os.listdir(input_file_path), desc=f"Chopping patches of size {patch_size}"):
        if image.startswith('.'):
            continue
        img_path = os.path.join(input_file_path, image)
        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"Could not open {image}: {e}")
            continue

        imageWidth, imageHeight = img.size
        num_x = imageWidth // patch_size
        num_y = imageHeight // patch_size
        
        # Get the original bounding box for this image, if available.
        # Expecting the annotation key to match the image filename.
        orig_annotation = annotations.get(image, {})
        orig_bbox = orig_annotation.get("bbox", None)

        for x in range(num_x):
            for y in range(num_y):
                patch_origin = (x * patch_size, y * patch_size)
                bbox_patch = (patch_origin[0], patch_origin[1], patch_origin[0] + patch_size, patch_origin[1] + patch_size)
                patch_img = img.crop(bbox_patch)
                
                # Define a filename that includes the original image name and patch grid position.
                patch_filename = f"{os.path.splitext(image)[0]}_{x}_{y}.jpg"
                patch_filepath = os.path.join(output_file_path, patch_filename)
                patch_img.save(patch_filepath, optimize=True)

                # If there is a bounding box for the original image, compute its mapping.
                if orig_bbox:
                    rel_bbox = adjust_bounding_box(orig_bbox, patch_origin, patch_size)
                    # Only record a bounding box if there's overlap.
                    patch_annotations[patch_filename] = {"bbox": rel_bbox}
                else:
                    patch_annotations[patch_filename] = {"bbox": None}

    # Save the patch annotations to a JSON file in the output folder.
    annotations_outfile = os.path.join(output_file_path, "patch_annotations.json")
    with open(annotations_outfile, "w") as f:
        json.dump(patch_annotations, f, indent=4)
    print(f"Patch annotations saved to {annotations_outfile}")