import glob
import os
import MFCC
import re

if __name__ == '__main__':
    #print(os.path.join(os.getcwd()))
    files = os.listdir(os.path.join(os.getcwd()))
    print(files)
    raw_raw_folders = []
    raw_folders = []
    for folder in files:
        if re.match("raw\w+", folder):
            raw_folders.append(re.match("raw\w+", folder).group())
            print(raw_folders)
            for raw_folder in raw_folders:
                in_raw_folders = os.listdir(os.path.join(os.getcwd(), raw_folder))
                for raw_raw_folder in in_raw_folders:
                    raw_raw_folders.append(os.path.join(os.getcwd(), raw_folder, raw_raw_folder))
    print(raw_raw_folders)
    for raw_raw_folder in raw_raw_folders:
        for fn in glob.glob(os.path.join(raw_raw_folder, "*", "*.wav")):
            MFCC.create_ceps(fn)
