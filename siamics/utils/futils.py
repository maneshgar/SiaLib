import os
import glob

def list_files(root, pattern="*", extension=None, depth=1):

    if extension is not None: 
        # Create the search pattern
        glob_pattern = f"{pattern}{extension}"

    result = []
    if depth == 0:
        return glob.glob(os.path.join(root, glob_pattern))
    
    for dirpath, dirnames, filenames in os.walk(root):
        # Calculate the current depth by counting the separators in dirpath
        current_depth = dirpath[len(root):].count(os.sep)
        if current_depth < depth:
            result.extend(glob.glob(os.path.join(dirpath, glob_pattern)))
        else:
            # Prevent descending into subdirectories deeper than the specified depth
            dirnames[:] = []  
            
    return result

def create_directories(fullpath, is_dir=False):
    """
    Create all directories in the given path, excluding the filename, if they do not already exist.
    
    Parameters:
    path_with_filename (str): The absolute path that includes the filename.
    
    Returns:
    None
    """
    if is_dir:
        try:            
            # os.makedirs creates all intermediate-level directories needed to contain the leaf directory
            os.makedirs(fullpath, exist_ok=True)
        except Exception as e:
            print(f"An error occurred: {e}")
    else: 
        try:
            # Extract the directory part of the path
            directory_path = os.path.dirname(fullpath)
            
            # os.makedirs creates all intermediate-level directories needed to contain the leaf directory
            if directory_path:  # Ensure there's a directory path to create
                os.makedirs(directory_path, exist_ok=True)
            else:
                print("No directory path to create.")
        except Exception as e:
            print(f"An error occurred: {e}")

def get_basename(file_path, extention=False):
    """
    Get the file name without the extension from a full file path.

    Parameters:
    - file_path: str, full path to the file

    Returns:
    - str, file name without the extension
    """
    # Get the base name (file name with extension)
    base_name = os.path.basename(file_path)
    if extention: 
        return base_name
    # Split the base name into name and extension
    base_name, _ = os.path.splitext(base_name)
    return base_name

def save_list(list, filename):
    with open(filename, 'w') as f:
        for element in list:
            f.write(f"{element}\n")