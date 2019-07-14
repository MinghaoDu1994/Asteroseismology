#!/usr/bin/env/ python
# coding: utf-8

import os

__all__ = ["traverse_dir"]


def traverse_dir(file_path: str,file_type: str=None): #search files\n",
    files_name=[]  
    files_path=[]
    for root, dirs, files in os.walk(file_path):
        if file_type != None:
            for file in files:              
                if os.path.splitext(file)[1] == '.%s' % file_type:  
                    files_name.append(file)
                    files_path.append(os.path.join(root,file))
        else:
            for file in files:               
                files_name.append(file)
                files_path.append(os.path.join(root,file))
    return files_name,files_path