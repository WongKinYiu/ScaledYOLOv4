# This file contains google utils: https://cloud.google.com/storage/docs/reference/libraries
# pip install --upgrade google-cloud-storage
# from google.cloud import storage

import os
import platform
import time
from pathlib import Path


def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    weights = weights.strip().replace("'", '')
    msg = weights + ' missing'

    r = 1  # return
    if len(weights) > 0 and not os.path.isfile(weights):
        d = {'',
             }

        file = Path(weights).name
        if file in d:
            r = gdrive_download(id=d[file], name=weights)

        if not (r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # weights exist and > 1MB
            os.remove(weights) if os.path.exists(weights) else None  # remove partial downloads
            s = ''
            r = os.system(s)  # execute, capture return values

            # Error check
            if not (r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # weights exist and > 1MB
                os.remove(weights) if os.path.exists(weights) else None  # remove partial downloads
                raise Exception(msg)


def gdrive_download(id='1n_oKgR81BJtqk75b00eAjdv03qVCQn2f', name='coco128.zip'):
    # Downloads a file from Google Drive, accepting presented query
    # from utils.google_utils import *; gdrive_download()
    t = time.time()

    print('Downloading https://drive.google.com/uc?export=download&id=%s as %s... ' % (id, name), end='')
    os.remove(name) if os.path.exists(name) else None  # remove existing
    os.remove('cookie') if os.path.exists('cookie') else None

    # Attempt file download
    out = "NUL" if platform.system() == "Windows" else "/dev/null"
    os.system('curl -c ./cookie -s -L "drive.google.com/uc?export=download&id=%s" > %s ' % (id, out))
    if os.path.exists('cookie'):  # large file
        s = 'curl -Lb ./cookie "drive.google.com/uc?export=download&confirm=%s&id=%s" -o %s' % (get_token(), id, name)
    else:  # small file
        s = 'curl -s -L -o %s "drive.google.com/uc?export=download&id=%s"' % (name, id)
    r = os.system(s)  # execute, capture return values
    os.remove('cookie') if os.path.exists('cookie') else None

    # Error check
    if r != 0:
        os.remove(name) if os.path.exists(name) else None  # remove partial
        print('Download error ')  # raise Exception('Download error')
        return r

    # Unzip if archive
    if name.endswith('.zip'):
        print('unzipping... ', end='')
        os.system('unzip -q %s' % name)  # unzip
        os.remove(name)  # remove zip to free space

    print('Done (%.1fs)' % (time.time() - t))
    return r


def get_token(cookie="./cookie"):
    with open(cookie) as f:
        for line in f:
            if "download" in line:
                return line.split()[-1]
    return ""
