
import json
import numpy as np
import pydicom
import SimpleITK as sitk
import pydicom
import pandas as pd
def convert_osirix_json_to_nifti(json_url, reference_image_url, output_url = "same", close_voxels = 2):
        if isinstance(reference_image_url, str):
            reference_image = sitk.ReadImage(reference_image_url)
        else:
            reference_image = reference_image_url
        with open(json_url, 'r') as file:
            data = json.load(file)
    
        image_height = data['Images'][0]['ImageHeight']
        image_width = data['Images'][0]['ImageWidth']
        num_images = data['Images'][0]['ImageTotalNum']
    
        segmentation_mask = np.zeros((num_images, image_height, image_width), dtype=np.uint8)
    
        for image in data['Images']:
            image_index = image['ImageIndex']
            for roi in image['ROIs']:
                for point in roi['Point_px']:
                    x, y = map(float, point.strip('()').split(','))
                    x, y = int(round(x)), int(round(y))
                    segmentation_mask[image_index, y, x] = 1
    
        segment = sitk.GetImageFromArray(segmentation_mask)
        segment.CopyInformation(reference_image)
        segment = sitk.BinaryMorphologicalClosing(segment, [close_voxels, close_voxels, close_voxels])
        for slice in range(segment.GetSize()[2]):
            segment[:,:,slice] = sitk.BinaryFillhole(segment[:,:,slice], fullyConnected = True)
            
        if output_url == "same":
            sitk.WriteImage(segment, json_url.replace(".json", ".nii.gz"))
        else:
            sitk.WriteImage(segment, output_url)
        return segment

def info(dicom_url):

        info_dictionary = pydicom.filereader.dcmread(dicom_url)
        try:
            try:
                info_data_frame = pd.DataFrame(info_dictionary.values())
                info_data_frame[0] = info_data_frame[0].apply(lambda x: pydicom.dataelem.DataElement_from_raw(x) if isinstance(x, pydicom.dataelem.RawDataElement) else x)
                info_data_frame['name'] = info_data_frame[0].apply(lambda x: x.name)
                info_data_frame['value'] = info_data_frame[0].apply(lambda x: x.value)
            except:
                try:
                    # print(info_dictionary)
                    info_data_frame = pd.DataFrame(info_dictionary.values())
                    info_data_frame[0] = info_data_frame[0].apply(lambda x: pydicom.dataelem.DataElement_from_raw(x) if isinstance(x, pydicom.dataelem.RawDataElement) else x)
                    info_data_frame['name'] = info_data_frame[0].apply(lambda x: x.name)
                    info_data_frame['value'] = info_data_frame[0].apply(lambda x: x.value)
                except:
                    #######
                    attributes_to_include = info_dictionary.keys()
                    data = {attribute: info_dictionary.get(attribute, '') for attribute in attributes_to_include}
                    info_data_frame = pd.DataFrame(data, index = [0])
                    info_data_frame = info_data_frame.apply(lambda x: pydicom.dataelem.DataElement_from_raw(x) if isinstance(x, pydicom.dataelem.RawDataElement) else x)
                    info_data_frame['name'] = info_data_frame.apply(lambda x: x.name)
                    info_data_frame['value'] = info_data_frame.apply(lambda x: x.value)
                    #######  
            info_data_frame = info_data_frame[['name', 'value']]
            info_data_frame = info_data_frame.set_index('name').T.reset_index(drop=True)
            info_data_frame.index = [dicom_url]
            info_data_frame.name = "dicom_url"
            info_data_frame = info_data_frame.drop(columns = "Pixel Data")
        except:
            info_data_frame = "not availablle"
        
        return info_dictionary, info_data_frame

def SearchInfo(pydicom_object, search_pattern, max_depth=30):
    import pydicom
    import fnmatch
    matching_attributes = []
    def recursive_search(ds, depth):
        if depth > max_depth:
            return
        for elem in ds:
            if isinstance(elem.value, pydicom.sequence.Sequence):
                for item in elem.value:
                    recursive_search(item, depth + 1)
            elif isinstance(elem.name, pydicom.dataset.Dataset):
                recursive_search(elem.name, depth + 1)
            elif isinstance(elem, pydicom.dataelem.DataElement):
                if fnmatch.fnmatch(str(elem.name), search_pattern):
                    matching_attributes.append(elem)
    recursive_search(pydicom_object, 1)
    return matching_attributes


def read_rtstruct(rtstruct_url):
    dicom_info_df = info(rtstruct_url)[1]
    # Read RTSTRUCT file using Pydicom
    rtstruct = pydicom.dcmread(rtstruct_url)
    pixel_array = rtstruct.pixel_array
   
    rtstruct_origin = list(SearchInfo(rtstruct, "*Image*Position*")[0])
    first_slice_position = list(SearchInfo(rtstruct, "*Image*Position*")[0])
    first_slice_position = [float(x) for x in first_slice_position]
    second_slice_position = list(SearchInfo(rtstruct, "*Image*Position*")[1])
    second_slice_position = [float(x) for x in second_slice_position]
    rtstruct_slice_thickness = max([np.abs(x-y) for x,y in zip(first_slice_position, second_slice_position)])
    rtstruct_spacing = [float(x) for x in list(SearchInfo(rtstruct, "*Spacing*")[0])] + [rtstruct_slice_thickness]
    
    rtstruct_direction = [float(x) for x in list(SearchInfo(rtstruct, "*Image*Orientation*")[0])]
    
    iop = np.array(rtstruct_direction)
    direction_matrix = np.reshape(iop, (2, 3)).T
    # Compute the third column for a complete 3x3 matrix
    normal_vector = np.cross(direction_matrix[:, 0], direction_matrix[:, 1])
    direction_matrix = np.column_stack([direction_matrix, normal_vector]).flatten()
    
    rtstruct_sitk_image = sitk.GetImageFromArray(pixel_array)
    # Set the image origin, spacing, and direction from the DICOM information
    rtstruct_sitk_image.SetOrigin(rtstruct_origin)
    rtstruct_sitk_image.SetSpacing(rtstruct_spacing)
    rtstruct_sitk_image.SetDirection(direction_matrix)
    dicom_info_df.to_excel(rtstruct_url.replace(".dcm", ".xlsx"))
    sitk.WriteImage(rtstruct_sitk_image,  rtstruct_url.replace(".dcm", ".nii.gz"))
    return rtstruct_sitk_image, dicom_info_df, rtstruct_url.replace(".dcm", ".nii.gz"), rtstruct_url.replace(".dcm", ".xlsx")
