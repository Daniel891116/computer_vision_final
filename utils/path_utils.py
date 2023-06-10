import os

def check_directory_valid(path: str) -> bool:
    """
    check whether a directory exists, if not, create a new one

    param:
        path: path to check
    return:
        flag: whether the directory exists
    """
    flag = os.path.exists(path)
    if not flag:
        os.makedirs(path)
    return flag