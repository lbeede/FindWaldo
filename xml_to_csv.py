import os
import xml.etree.ElementTree as ET
import pandas as pd

def convert_each_xml_to_csv(xml_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(xml_folder):
        if not file.endswith('.xml'):
            continue
        
        xml_path = os.path.join(xml_folder, file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        rows = []
        for obj in root.findall('object'):
            cls = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            rows.append([filename, width, height, cls, xmin, ymin, xmax, ymax])

        df = pd.DataFrame(rows, columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])

        # Save each CSV using the XML filename
        base_name = os.path.splitext(file)[0]
        csv_path = os.path.join(output_folder, f"{base_name}.csv")
        df.to_csv(csv_path, index=False, header=False)
        print(f"Saved: {csv_path}")

