from tqdm import tqdm
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import os 
import re

def get_shape(output_dict, key_for_shape):
    for key in tqdm(output_dict.keys()):
        output_dict[key]['shape'] = output_dict[key][key_for_shape].shape
    return output_dict

def get_aspect_ratio(df, columns_list):
    df['max'] = df[columns_list].max(axis=1)
    df['min'] = df[columns_list].min(axis=1)
    df['aspect_ratio'] = df['min'] / df['max']
    return df

def rotate_images(df, image_column, dimensions_list):
    df['requires_rotate'] = np.where((df[dimensions_list[0]] > df[dimensions_list[1]]), True, False)
    new_column_name = image_column + '_rotated'
    df[new_column_name] = np.where(
        (df['requires_rotate'] == True), 
        df.apply(lambda row : np.rot90(row[image_column]), axis=1), 
        df[image_column])
    return df

## Class for resizing??
def find_resize_shape(df, how='max'):
    if how == 'max':
        resize_to_dim1 = df['max'].max()
        resize_to_dim0 = df[df['max'] == resize_to_dim1]['min'].max()
    elif how == 'min':
        resize_to_dim1 = df['max'].min()
        resize_to_dim0 = df[df['max'] == resize_to_dim1]['min'].min()
    return (resize_to_dim1, resize_to_dim0)

def resize_image_in_column(df, image_column, how='max'):
    new_size = find_resize_shape(df=df, how=how)
    new_column_name = image_column + '_resized'
    df[new_column_name] = df.apply(lambda row: cv2.resize(row[image_column], dsize = new_size, interpolation=cv2.INTER_LANCZOS4).flatten(), axis=1)
    return df

def resize_images(df, image_column, how='max', respect_aspect_ratio=False, aspect_ratio_column='aspect_ratio'):
    if respect_aspect_ratio == False:
        resized_df = resize_image_in_column(df=df, image_column=image_column, how=how)

    elif respect_aspect_ratio == True:
        aspect_ratios = set(df[aspect_ratio_column])
        resized_df = pd.DataFrame()
        for value in aspect_ratios:
            small_df = df[df[aspect_ratio_column] == value]
            small_df = resize_image_in_column(df=small_df, image_column=image_column, how=how)
            resized_df = pd.concat([resized_df, small_df])

    return resized_df

def get_image_metadata(imagename):
    image = Image.open(imagename)
    info_dict = {
        "Filename": image.filename,
        "Image Size": image.size,
        "Image Height": image.height,
        "Image Width": image.width,
        "Image Format": image.format,
        "Image Mode": image.mode,
        "Image is Animated": getattr(image, "is_animated", False),
        "Frames in Image": getattr(image, "n_frames", 1)
    }
    exifdata = image.getexif()
    
    for tag_id in exifdata:
        tag = TAGS.get(tag_id, tag_id)
        data = exifdata.get(tag_id)
 
        if isinstance(data, bytes):
            data = data.decode()
        info_dict[tag] = data
    return info_dict

def get_metadata_for_images_in_path_list(path_list):
    meta_data_dict = {}
    for path in path_list:
        directory = os.fsencode(path)
        for file in tqdm(os.listdir(directory)):
            filename = os.fsdecode(file)
            path_strip = re.sub(r'[^a-zA-Z0-9]', '', path)
            if filename == '.ipynb_checkpoints':
                continue
            else:
                meta_data_dict[path_strip + '_' + filename] = get_image_metadata(path + filename)
    return meta_data_dict