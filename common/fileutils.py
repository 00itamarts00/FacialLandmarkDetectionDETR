import os


def fileparts(filename):
    p = os.path.dirname(filename)
    f = os.path.basename(filename)
    f, e = os.path.splitext(f)
    return p, f, e


def get_files_list_(dir_name):
    # create a list of file and sub directories
    # names in the given directory
    files_list = os.listdir(dir_name)

    all_files = list()
    # Iterate over all the entries
    for entry in files_list:
        # Create full path
        full_path = os.path.join(dir_name, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(full_path):
            all_files = all_files + get_files_list_(full_path)

        else:
            all_files.append(full_path)

    return all_files


def get_files_list(dir_name, types):
    fileslist = get_files_list_(dir_name)
    fileslist_ = [f for f in fileslist if os.path.isfile(os.path.join('', f)) and f.endswith(types)]
    return fileslist_
