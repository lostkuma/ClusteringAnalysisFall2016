#!usr/bin/env python3 #_init.py

import os

def FilePath(directory, filename):
    """ for each file, directory and filename complete path"""
    script_dir = os.getcwd()
    file_path = os.path.join(script_dir, directory, filename)
    return file_path
