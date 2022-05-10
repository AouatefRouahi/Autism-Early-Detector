from PIL import Image
import os
from os.path import isfile, isdir, join
from os import listdir, mkdir, makedirs
import glob

#**************************************************************************************

def replace_png_by_jpg(origin_directory_autistic,
    origin_directory_non_autistic,
    target_directory_global,
    target_directory_autistic,
    target_directory_non_autistic):
    
    '''
    Creates new folders in which replacing png images by jpg images
    '''
    
    AUTISTIC_DIRECTORY = origin_directory_autistic
    NON_AUTISTIC_DIRECTORY = origin_directory_non_autistic
    NEW_DATASET_DIRECTORY = target_directory_global
    NEW_AUTISTIC_DIRECTORY = target_directory_autistic
    NEW_NON_AUTISTIC_DIRECTORY = target_directory_non_autistic

    
    # count number of jpg and png in "old" dataset :
    count_jpg_in_old = []
    count_png_in_old = []
    for i, old_directory in enumerate([AUTISTIC_DIRECTORY, NON_AUTISTIC_DIRECTORY]) :
        jpg_files_in_old = glob.glob(old_directory + "/*.jpg")
        png_files_in_old = glob.glob(old_directory + "/*.png")
        count_jpg_in_old.append(0)
        count_png_in_old.append(0)
        for jpg_file in jpg_files_in_old:
            count_jpg_in_old[i] += 1
        for png_file in png_files_in_old:
            count_png_in_old[i] += 1
            
    # make new datasets directories if they don't already exist :
    if not isdir(NEW_DATASET_DIRECTORY) :
        mkdir(NEW_DATASET_DIRECTORY)
    if not isdir(NEW_AUTISTIC_DIRECTORY) :
        mkdir(NEW_AUTISTIC_DIRECTORY)
    if not isdir(NEW_NON_AUTISTIC_DIRECTORY) :
        mkdir(NEW_NON_AUTISTIC_DIRECTORY)


    # empty and fill new directories :
    for old_directory, new_directory in [[AUTISTIC_DIRECTORY, NEW_AUTISTIC_DIRECTORY],
                          [NON_AUTISTIC_DIRECTORY, NEW_NON_AUTISTIC_DIRECTORY]] :

        # empty already existing files in new directories :
        jpg_files_in_new = glob.glob(new_directory + "/*.jpg")
        png_files_in_new = glob.glob(new_directory + "/*.png")
        for jpg_file in jpg_files_in_new:
            try:
                os.remove(jpg_file)
            except OSError as e:
                print(f"Error:{e.strerror}")
        for png_file in png_files_in_new:
            try:
                os.remove(png_file)
            except OSError as e:
                print(f"Error:{e.strerror}")

        # fill in with new files :
        jpg_files_in_old = glob.glob(old_directory + "/*.jpg")
        png_files_in_old = glob.glob(old_directory + "/*.png")
        for jpg_file in jpg_files_in_old:
            with Image.open(jpg_file) as image :
                new_total_path = new_directory + jpg_file[-8:]
                image.save(new_total_path)

        for png_file in png_files_in_old:
            with Image.open(png_file).convert('RGB') as image :
                new_total_path = new_directory + png_file[-8:-3] + "jpg"
                image.save(new_total_path)


    # count number of jpg and png in "new" dataset :
    count_jpg_in_new = []
    count_png_in_new = []
    for i, new_directory in enumerate([NEW_AUTISTIC_DIRECTORY, NEW_NON_AUTISTIC_DIRECTORY]) :
        jpg_files_in_new = glob.glob(new_directory + "/*.jpg")
        png_files_in_new = glob.glob(new_directory + "/*.png")
        count_jpg_in_new.append(0)
        count_png_in_new.append(0)
        for jpg_file in jpg_files_in_new:
            count_jpg_in_new[i] += 1
        for png_file in png_files_in_new:
            count_png_in_new[i] += 1
    
    count_image_by_type = {
        "jpg_in_old_autistic" : count_jpg_in_old[0],
        "png_in_old_autistic" : count_png_in_old[0],
        "jpg_in_old_non_autistic" : count_jpg_in_old[1],
        "png_in_old_non_autistic" : count_png_in_old[1],
        
        "jpg_in_new_autistic" : count_jpg_in_new[0],
        "png_in_new_autistic" : count_png_in_new[0],
        "jpg_in_new_non_autistic" : count_jpg_in_new[1],
        "png_in_new_non_autistic" : count_png_in_new[1]
        
    }
    
    return count_image_by_type

def create_file_name(image_id) :
    '''
    create the image file name based on the number of the image
    '''
    while len(image_id) < 4 :
        image_id = '0' + image_id
    image_id = image_id + '.jpg'
    return image_id

def remove_annoted_images(
    target_directory_autistic,
    target_directory_non_autistic,
    df_autistic,
    df_non_autistic,
    keep_older = False,
    keep_tilted = True,
    keep_side = True,
    keep_not_centered = True,
    keep_potoshopped = False,
    keep_bw = False):
    
    '''
    Filter the images based on the visual annotations
    '''
    
    # dataframe transformation
    df_autistic.dropna(how = 'all', inplace = True)
    df_non_autistic.dropna(how = 'all', inplace = True)
    df_autistic.fillna(False, inplace = True)
    df_non_autistic.fillna(False, inplace = True)
    df_autistic.replace("X", True, inplace = True)
    df_non_autistic.replace("X", True, inplace = True)
    df_autistic['image id'] = df_autistic['image id'].astype(int)
    df_non_autistic['image id'] = df_non_autistic['image id'].astype(int)
    df_autistic['image id'] = df_autistic['image id'].astype(str)
    df_non_autistic['image id'] = df_non_autistic['image id'].astype(str)
    df_autistic['file_name'] = df_autistic['image id'].apply(create_file_name)
    df_non_autistic['file_name'] = df_non_autistic['image id'].apply(create_file_name)
    
    # create lists identifying images we want to delete
    autistic_images_to_delete = []
    non_autistic_images_to_delete = []
    
    if keep_older == False :
        autistic_images_to_delete += df_autistic[df_autistic['older'] == True]["file_name"].to_list()
        non_autistic_images_to_delete += df_non_autistic[df_non_autistic['older'] == True]["file_name"].to_list()
        
    if keep_tilted == False :
        autistic_images_to_delete += df_autistic[df_autistic['tilted face'] == True]["file_name"].to_list()
        non_autistic_images_to_delete += df_non_autistic[df_non_autistic['tilted face'] == True]["file_name"].to_list()
        
    if keep_side == False :
        autistic_images_to_delete += df_autistic[df_autistic['pose sideways'] == True]["file_name"].to_list()
        non_autistic_images_to_delete += df_non_autistic[df_non_autistic['pose sideways'] == True]["file_name"].to_list()
        
    if keep_not_centered == False :
        autistic_images_to_delete += df_autistic[df_autistic['not only face / face not centered'] == True]["file_name"].to_list()
        non_autistic_images_to_delete += df_non_autistic[df_non_autistic['not only face / face not centered'] == True]["file_name"].to_list()
        
    if keep_potoshopped == False :
        autistic_images_to_delete += df_autistic[df_autistic['photoshoped'] == True]["file_name"].to_list()
        non_autistic_images_to_delete += df_non_autistic[df_non_autistic['photoshoped'] == True]["file_name"].to_list()
        
    if keep_bw == False :
        autistic_images_to_delete += df_autistic[df_autistic['black and white'] == True]["file_name"].to_list()
        non_autistic_images_to_delete += df_non_autistic[df_non_autistic['black and white'] == True]["file_name"].to_list()
        
    # remove duplicates
    autistic_images_to_delete = list(set(autistic_images_to_delete))
    non_autistic_images_to_delete = list(set(non_autistic_images_to_delete))
    
    # deleting images found in list 
    files_to_delete_autistic = target_directory_autistic + "/*.jpg"
    jpg_files_autistic = glob.glob(files_to_delete_autistic)
    nb_autistic_images_deleted = 0
    for jpg_file in jpg_files_autistic:
        if jpg_file[-8:] in autistic_images_to_delete :
            nb_autistic_images_deleted += 1
            try:
                os.remove(jpg_file)
            except OSError as e:
                print(f"Error:{ e.strerror}")
      
    files_to_delete_non_autistic = target_directory_non_autistic + "/*.jpg"
    jpg_files_non_autistic = glob.glob(files_to_delete_non_autistic)
    nb_non_autistic_images_deleted = 0
    for jpg_file in jpg_files_non_autistic:
        if jpg_file[-8:] in non_autistic_images_to_delete :
            nb_non_autistic_images_deleted += 1
            try:
                os.remove(jpg_file)
            except OSError as e:
                print(f"Error:{ e.strerror}")
    
    nb_images_deleted = {
        "autistic" : nb_autistic_images_deleted,
        "non_autistic" : nb_non_autistic_images_deleted
    }

    return nb_images_deleted

def remove_images_based_on_dimension(
    target_directory_autistic,
    target_directory_non_autistic,
    min_width = 0,
    max_width = 5000, #remplacer par = None et check en début de fonction
    min_heigth = 0,
    max_heigth = 5000,
    min_ratio = 0,
    max_ratio = 3):
    
    list_nb_images_deleted = []
    
    for directory in [target_directory_autistic, target_directory_non_autistic] :
        
        nb_images_deleted = 0
        
        jpg_files = glob.glob(directory + "/*.jpg")

        for jpg_file in jpg_files :
            with Image.open(jpg_file) as image :
                width, heigth = image.size
                
            ratio = heigth / width

            width_nok = not (min_width <= width <= max_width)
            heigth_nok = not (min_heigth <= heigth <= max_heigth)
            ratio_nok = not (min_ratio <= ratio <= max_ratio)

            if any((width_nok, heigth_nok, ratio_nok)):
                nb_images_deleted += 1
                try:
                    os.remove(jpg_file)
                except OSError as e:
                    print(f"Error:{ e.strerror}")
                    
        list_nb_images_deleted.append(nb_images_deleted)
    
    count_images_deleted = {
        "autistic_images_deleted" : list_nb_images_deleted[0],
        "non_autistic_images_deleted" : list_nb_images_deleted[1]
    }
    
    return count_images_deleted

def image_pre_processing(
    origin_directory_autistic,
    origin_directory_non_autistic,
    target_directory_global,
    target_directory_autistic,
    target_directory_non_autistic,
    df_autistic,
    df_non_autistic,
    keep_older = False,
    keep_tilted = True,
    keep_side = True,
    keep_not_centered = True,
    keep_potoshopped = False,
    keep_bw = False,
    min_width = 0,
    max_width = 5000,
    min_heigth = 0,
    max_heigth = 5000,
    min_ratio = 0,
    max_ratio = 3) :
    
    # replace png by jpg and create new folders
    results_png_replacement = replace_png_by_jpg(origin_directory_autistic,
    origin_directory_non_autistic,
    target_directory_global,
    target_directory_autistic,
    target_directory_non_autistic)
    
    
    # remove images based on annotations
    results_images_removed_label = remove_annoted_images(
    target_directory_autistic,
    target_directory_non_autistic,
    df_autistic,
    df_non_autistic,
    keep_older,
    keep_tilted,
    keep_side,
    keep_not_centered,
    keep_potoshopped,
    keep_bw)
    
    # remove images based on dimensions
    results_images_removed_dimension = remove_images_based_on_dimension(
    target_directory_autistic,
    target_directory_non_autistic,
    min_width,
    max_width,
    min_heigth,
    max_heigth,
    min_ratio,
    max_ratio)
    
    results = {"png_deletion_number" : results_png_replacement,
               "images_deletion_by_label_number" : results_images_removed_label,
               "images_deletion_by_size_number" : results_images_removed_dimension }
    
    return results