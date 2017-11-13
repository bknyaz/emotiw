# python3

import numpy as np
import pickle
import urllib, json
from http import client as httplib
from urllib import parse as urlparse
import urllib.parse
import os
import subprocess

#assert sklearn.__version__ == '0.18.1', 'the results can differ a lot if another (e.g., 0.19.0) version is used'

data_dir = os.path.dirname(os.path.realpath(__file__))

def download_load(url):
    api = httplib.HTTPSConnection('cloud-api.yandex.net')
    url ='/v1/disk/public/resources/download?public_key=%s' % urllib.parse.quote(url)
    api.request('GET', url)
    resp = api.getresponse()
    file_info = json.loads(resp.read())
    api.close()
    filename = urlparse.parse_qs(urlparse.urlparse(file_info['href']).query)['filename'][0]
    if not os.path.isfile(filename):
        print('downloading %s ...' % filename)
        cmd = "wget -O \"/%s/%s\" \"%s\"" % (data_dir, filename, file_info['href'])
        return_code = subprocess.call(cmd, shell=True)
        print('return code for %s = %d' % (filename, return_code))

    with open(filename, 'rb') as f:
        features = pickle.load(f)
    return features


# ======= Download and load features of the audio model and 4 networks =========
features_urls = [# Audio
                'https://yadi.sk/d/hgu71s_x3NSMop',
                 'https://yadi.sk/d/P3nm6pSv3NSMrE',
                 'https://yadi.sk/d/CZQ8NP3U3NSMsv',
                 # Model-A
                 'https://yadi.sk/d/TihIWuoL3NSLau',
                 'https://yadi.sk/d/VbHW2P_i3NSLo3',
                 'https://yadi.sk/d/4qM_7BDk3NSLwx',
                 # Model-B
                 'https://yadi.sk/d/SPDLu7uQ3NSM6a',
                 'https://yadi.sk/d/V-XGO2ZD3NSM8q',
                 'https://yadi.sk/d/AXMXnb3Y3NSMCo',
                 # Model-C
                 'https://yadi.sk/d/Btml0WpJ3NSMLW',
                 'https://yadi.sk/d/YbidqsiF3NSMNS',
                 'https://yadi.sk/d/NoAj-XQM3NSMSm',
                 # VGG-Face
                 'https://yadi.sk/d/L9t1ldfb3NSMYV',
                 'https://yadi.sk/d/k-0Kft3F3NSMaw',
                 'https://yadi.sk/d/eE9vNkkC3NSMgD']

# Audio
train_audio = download_load(features_urls[0])
val_audio = download_load(features_urls[1])
test_audio = download_load(features_urls[2])

# Network features
train_features = [download_load(url) for url in features_urls[3::3]]
val_features = [download_load(url) for url in features_urls[4::3]]
test_features = [download_load(url) for url in features_urls[5::3]]

# =============================================================================
