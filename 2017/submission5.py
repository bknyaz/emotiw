# python3

import numpy as np
import pickle
import urllib, json
from http import client as httplib
from urllib import parse as urlparse
import urllib.parse
import os
import subprocess

import sklearn

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import confusion_matrix

from data_loader_emotiw import DataLoader

assert sklearn.__version__ == '0.18.1', 'the results can differ a lot if another (e.g., 0.19.0) version is used'

SPLIT = False
rootsift = True
audio = True
SCALER = MinMaxScaler
print('SPLIT = %s' % str(SPLIT))
print('audio = %s' % str(audio))
print('rootsift = %s' % str(rootsift))
print('SCALER = %s' % str(SCALER))

features = [np.mean, np.std, np.min]
print(features)

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


def identity(features, axis=0):
    return features


def split_concat(features, fn, axis):
    if not SPLIT:
        return fn(features, axis=axis)
    if isinstance(features, list):
        features = np.stack(features)
    n_split = np.int(features.shape[0]/2.)
    features_new = np.concatenate((fn(features[:n_split,:], axis=axis), fn(features[n_split:,:], axis=axis)))
    return features_new


def root_sift(data):
    if not rootsift:
        return data
    data /= np.linalg.norm(data,ord=1,axis=1).reshape((-1,1))
    data = np.sign(data)*(abs(data)**0.5)
    return data


def classify(data_train, data_val, train_labels, val_labels, cross_val):
    print(data_train.shape)
    print(data_val.shape)

    SVM_C_range = [5e-6, 1e-5, 2e-5, 3e-5, 4e-5, 6e-5, 8e-5, 1e-4, 2e-4, 4e-4, 6e-4, 8e-4, 1e-3]

    if cross_val:
        print('Cross-validation')
        cv_acc = []
        for SVM_C in SVM_C_range:
            print('C = %f' % SVM_C)

            # Fix random_state, otherwise not reproducable
            cv = StratifiedKFold(5, shuffle=True, random_state=4096)

            clf = make_pipeline(StandardScaler(), LinearSVC(C=SVM_C))
            print(data_train.shape)
            print(train_labels.shape)
            scores = cross_val_score(clf, data_train, train_labels, cv=cv)
            cv_acc.append(np.mean(scores))
            print(cv_acc[-1])

        ind = np.argmax(cv_acc)
        SVM_C = SVM_C_range[ind]
        print('best_C = %f' % SVM_C)
        clf = make_pipeline(StandardScaler(), LinearSVC(C=SVM_C))
        clf.fit(data_train, train_labels)#[np.random.permutation(len(dloader.train_labels))])
        pred_train = clf.predict(data_train)
        train_acc = 100*np.mean(np.equal(np.array(pred_train), train_labels))
        print('train acc = %f' % train_acc)
        pred = clf.predict(data_val)
        decision_values = clf.decision_function(data_val)
        val_acc = 100*np.mean(np.equal(np.array(pred), val_labels))
        print('val acc = %f' % val_acc)
        return train_acc, val_acc, SVM_C, decision_values, clf
    else:
        train_acc, val_acc, decision_values, clfs = [], [], [], []

        for SVM_C in SVM_C_range:
            print('C = %f' % SVM_C)
            clf = make_pipeline(StandardScaler(), LinearSVC(C=SVM_C))
            clf.fit(data_train, train_labels)
            train_acc.append(100*np.mean(np.equal(np.array(clf.predict(data_train)), train_labels)))
            print('train acc = %f' % train_acc[-1])
            decision_values.append(clf.decision_function(data_val))
            val_acc.append(100*np.mean(np.equal(np.array(clf.predict(data_val)), val_labels)))
            clfs.append(clf)
            print('val acc = %f' % val_acc[-1])
        ind = np.argmax(val_acc)
        print('best_C = %f' % SVM_C_range[ind])
        return train_acc[ind], val_acc[ind], SVM_C_range[ind], decision_values[ind], clfs[ind]



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


dloader = DataLoader(data_dir=data_dir, shuffle_train=False)

train_features_video_dict, val_features_video_dict, test_features_video_dict = [], [], []
for i in range(len(train_features)):
    train_features[i] = (np.vstack([f.flatten().reshape((1,-1)) for f in train_features[i]]))
    val_features[i] = (np.vstack([f.flatten().reshape((1,-1)) for f in val_features[i]]))
    test_features[i] = (np.vstack([f.flatten().reshape((1,-1)) for f in test_features[i]]))
    train_features_video_dict.append({})
    val_features_video_dict.append({})
    test_features_video_dict.append({})

print(len(dloader.val_labels_images))
print(len(dloader.val_images), val_features[0].min(), val_features[0].max())

for i, s in enumerate(dloader.train_images):
    video = str.join('/', s.split('/')[:-1])
    for j in range(len(train_features)):
        if video not in train_features_video_dict[j]:
            train_features_video_dict[j][video] = []
        train_features_video_dict[j][video].append(train_features[j][i])

for i, s in enumerate(dloader.val_images):
    video = str.join('/', s.split('/')[:-1])
    for j in range(len(val_features)):
        if video not in val_features_video_dict[j]:
            val_features_video_dict[j][video] = []
        val_features_video_dict[j][video].append(val_features[j][i])

with open('%s/Test-frames-faces/content.txt' % data_dir, 'r') as f:
    test_images = f.readlines()
with open('%s/test_videos.txt' % data_dir, 'r') as f:
    test_videos = f.readlines()

test_videos = [v.rstrip()[:-4] for v in test_videos]

for i, s in enumerate(test_images):
    video = str.join('/', s.split('/')[:-1])
    for j in range(len(test_features)):
        if video not in test_features_video_dict[j]:
            test_features_video_dict[j][video] = []
        test_features_video_dict[j][video].append(test_features[j][i])


for j in range(len(val_features)):
    print(len(train_features_video_dict[j]), len(val_features_video_dict[j]), len(test_features_video_dict[j]))

train_features_video, val_features_video, test_features_video = [], [], []

for model in train_features_video_dict:
    train_features_video.append([])
    for feature_fn in features:
        feat = []
        for video in dloader.train_videos:
            feat.append(split_concat(model[video], feature_fn, axis=0))
        train_features_video[-1].append(np.stack(feat))

for model in val_features_video_dict:
    val_features_video.append([])
    for feature_fn in features:
        feat = []
        for video in dloader.val_videos:
            feat.append(split_concat(model[video], feature_fn, axis=0))
        val_features_video[-1].append(np.stack(feat))

for model in test_features_video_dict:
    test_features_video.append([])
    for feature_fn in features:
        feat = []
        for video in test_videos:
            feat.append(split_concat(model[video], feature_fn, axis=0))
        test_features_video[-1].append(np.stack(feat))

train_features_audio = []
for i, video in enumerate(dloader.train_videos):
    train_features_audio.append(train_audio[video+'.mp4'])
if audio:
    train_features_video.append([np.stack(train_features_audio)])

val_features_audio = []
for video in dloader.val_videos:
    val_features_audio.append(val_audio[video+'.mp4'])
if audio:
    val_features_video.append([np.stack(val_features_audio)])

test_features_audio = []
for video in test_videos:
    test_features_audio.append(test_audio['/%s.mp4' % video])
if audio:
    test_features_video.append([np.stack(test_features_audio)])

decision_values, decision_values_test, decision_values_test_train_plus_val, SVM_Cs, clfs, clf_sigmoids, probs, probs_test, probs_train_plus_val = [], [], [], [], [], [], [], [], []
models = ['Model-A', 'Model-B', 'Model-C', 'VGG-Face', 'Model-Audio']
for model in range(len(train_features_video)):
    train_features_video_cat, val_features_video_cat, test_features_video_cat = [], [], []
    for i, (feat_train, feat_val, feat_test) in enumerate(zip(train_features_video[model], val_features_video[model], test_features_video[model])):
        scaler = SCALER()
        feat_train = scaler.fit_transform(feat_train)
        feat_val = scaler.transform(feat_val)
        feat_test = scaler.transform(feat_test)

        train_features_video_cat.append(feat_train)
        val_features_video_cat.append(feat_val)
        test_features_video_cat.append(feat_test)

    if feat_train.shape[1] == 1582:
        fn_norm = identity
    else:
        fn_norm = root_sift

    data_train = fn_norm(np.concatenate(train_features_video_cat, axis=1))
    data_val = fn_norm(np.concatenate(val_features_video_cat, axis=1))

    train_acc, val_acc, SVM_C, decision_values_stat, clf_svm = classify(data_train,
                                                                data_val,
                                                                dloader.train_labels, dloader.val_labels, True)

    SVM_Cs.append(SVM_C)

    clf_sigmoid = CalibratedClassifierCV(clf_svm, method='sigmoid', cv=3)
    clf_sigmoid.fit(data_train, dloader.train_labels)


    print('%s accuracy on Emotiw Validation data: %1.2f %s' % (models[model], val_acc, '%'))

    data_test = fn_norm(np.concatenate(test_features_video_cat, axis=1))
    decision_values_stat_test = clf_svm.decision_function(data_test)

    probs.append(clf_sigmoid.predict_proba(data_val))
    probs_test.append(clf_sigmoid.predict_proba(data_test))

    decision_values.append(decision_values_stat)
    decision_values_test.append(decision_values_stat_test)

    print('Retraining an SVM on train+val')
    clf_svm_train_plus_val = make_pipeline(StandardScaler(), LinearSVC(C=SVM_C))
    data_train_all = fn_norm(np.concatenate((np.concatenate(train_features_video_cat, axis=1),
                                                     np.concatenate(val_features_video_cat, axis=1))))
    train_labels_all = np.concatenate((dloader.train_labels, dloader.val_labels))
    clf_svm_train_plus_val.fit(data_train_all, train_labels_all)

    clfs.append(clf_svm_train_plus_val)

    decision_values_stat_test_train_plus_val = clf_svm_train_plus_val.decision_function(data_test)
    decision_values_test_train_plus_val.append(decision_values_stat_test_train_plus_val)


    clf_sigmoid = CalibratedClassifierCV(clf_svm_train_plus_val, method='sigmoid', cv=3)
    clf_sigmoid.fit(data_train_all, train_labels_all)
    clf_sigmoids.append(clf_sigmoid)

    prob_sigmoid = clf_sigmoid.predict_proba(data_test)
    probs_train_plus_val.append(prob_sigmoid)

# We use order: ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
W = np.array([98, 40, 70, 144, 80, 28, 193]).reshape(1,-1)**0.5
W = W/np.sum(W)
print(W)
print(SVM_Cs)

decision_values_avg = np.mean(np.stack(decision_values, axis=2), axis=2)
pred_no_sigmoid = np.argmax(decision_values_avg, axis=1)
pred_no_weights = np.argmax(np.sum(np.stack(probs, axis=2), axis=2), axis=1)
pred = np.argmax(np.concatenate([p*W for p in np.sum(np.stack(probs, axis=2), axis=2)]), axis=1)
print('final val acc without sigmoid %f' % (100*np.mean(np.equal(pred_no_sigmoid, np.array(dloader.val_labels)))))
print('final val acc with/without weights %f/%f' % (100*np.mean(np.equal(pred, np.array(dloader.val_labels))), 100*np.mean(np.equal(pred_no_weights, np.array(dloader.val_labels)))))
values = []
total = []
for i in range(7):
    ind = np.where(np.array(dloader.val_labels) == i)[0]
    eq = np.equal(pred[ind], np.array(dloader.val_labels)[ind])
    values.append(np.sum(eq)*W[0,i])
    total.append(len(ind)*W[0,i])
    print('%s: %.1f (%d/%d)' % (dloader.classes[i], 100*(values[-1]/total[-1]), values[-1], total[-1]))
print('accuracy %.2f' % (100*np.sum(values)/np.sum(total)))
#final val acc using prob 48.302872
#Angry: 70.3 (67/96)
#Disgust: 2.5 (0/24)
#Fear: 8.7 (4/49)
#Happy: 88.9 (123/138)
#Neutral: 87.3 (162/186)
#Sad: 37.7 (28/74)
#Surprise: 2.2 (0/19)

# Prediction on Test data
test_labels_train_plus_val = np.argmax(np.concatenate([p*W for p in np.sum(np.stack(probs_train_plus_val, axis=2), axis=2)]), axis=1)
print('labels distribution')
print([len(np.where(test_labels_train_plus_val == i)[0]) for i in range(7)])

out_dir = os.path.join(data_dir, 'submission5_reproduce/')
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

for video in zip(test_videos, test_labels_train_plus_val):
    with open(out_dir + '%s.txt' % video[0], 'w') as f:
        f.write('%s' % dloader.emotion_names[video[1]])

# Check that results are the same as in Submission 5
submission_dir = os.path.join(data_dir, 'submission5/')
test_labels_s5 = []
for video in test_videos:
    with open(submission_dir + '%s.txt' % video, 'r') as f:
        label = f.readlines()[0].rstrip()
        test_labels_s5.append(np.where([c == label for c in dloader.classes])[0][0])
test_labels_s5 = np.array(test_labels_s5)
agreement = 100*np.mean(np.equal(test_labels_train_plus_val, test_labels_s5))
print('labels agreement %f' % agreement)
for l in range(7):
    mask1 = np.zeros(len(test_labels_train_plus_val))
    mask2 = np.zeros(len(test_labels_train_plus_val))
    mask1[np.where(test_labels_train_plus_val == l)[0]] = 1
    mask2[np.where(test_labels_s5 == l)[0]] = 1
    print('labels agreement (%s) %.1f' % (dloader.classes[l], 100*np.mean(np.equal(mask1, mask2))))

assert np.equal(agreement, 100.0), 'reproduction of results failed'
