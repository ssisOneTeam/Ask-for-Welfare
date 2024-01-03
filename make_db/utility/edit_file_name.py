import os

def add_string_to_filenames(folder_path, string_to_add):
    for file_name in os.listdir(folder_path):
        old_file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(old_file_path):
            new_file_name = string_to_add + file_name
            new_file_path = os.path.join(folder_path, new_file_name)
            os.rename(old_file_path, new_file_path)

if __name__ == "__main__" :
    folder_path = './DB_work/5_보건의료_지원'
    string_to_add = '0'
    add_string_to_filenames(folder_path, string_to_add)
