from typing import List
import os
import fnmatch


def search_for_files(path: str, pattern: str = "*") -> List[str]:
    """
    Find files in a directory. If a pattern is provided, then it returns only
    the files matching this pattern.
    """
    files_found = []
    for _, _, files in os.walk(path):
        for file_ in fnmatch.filter(files, pattern):
            files_found.append(os.path.join(path, file_))
    return sorted(files_found)
