import os
import numpy as np
#import cv2  # uncomment to recompute features


class DataLoader(object):

    def __init__(self, data_dir='~/data/AFEW/2017', shuffle_train=True, rand_seed=11):

        self.data_dir = data_dir.rstrip()
        self.shuffle_train = shuffle_train
        self.random  = np.random.RandomState(rand_seed) # to reproduce


        self.train_dir = 'Train-frames-faces'
        self.val_dir = 'Val-frames-faces'

        self.emotion_names = self.classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        with open('training_videos.txt', 'r') as f:
            self.train_videos = f.readlines()
        self.train_videos = [s.rstrip() for s in self.train_videos]
        self.train_labels = [np.where([e == s.split('/')[0] for e in self.emotion_names])[0] for s in self.train_videos]

        with open('validation_videos.txt', 'r') as f:
            self.val_videos = f.readlines()
        self.val_videos = [s.rstrip() for s in self.val_videos]
        self.val_labels = [np.where([e == s.split('/')[0] for e in self.emotion_names])[0] for s in self.val_videos]

        with open(os.path.join(self.data_dir, self.train_dir, 'content.txt'), 'r') as f:
            self.train_images = f.readlines()

        self.train_labels_images = [np.where([e == s.split('/')[0] for e in self.emotion_names])[0] for s in self.train_images]

        if shuffle_train:
            ids = self.random.permutation(len(self.train_labels_images))
            self.train_labels_images = [self.train_labels_images[i] for i in ids]
            self.train_images = [self.train_images[i] for i in ids]

        with open(os.path.join(self.data_dir, self.val_dir, 'content.txt'), 'r') as f:
            self.val_images = f.readlines()

        self.val_labels_images = [np.where([e == s.split('/')[0] for e in self.emotion_names])[0] for s in self.val_images]

        # Check dataset consistency
        assert(len(self.train_labels_images) == len(self.train_images))
        assert(len(self.val_labels_images) == len(self.val_images))
        assert(len(np.unique(self.train_labels_images)) == len(np.unique(self.val_labels_images)) == len(self.classes) == 7)

        self.train_labels = np.array(self.train_labels)
        self.val_labels = np.array(self.val_labels)

        s = sorted(zip(self.val_labels, self.val_videos), key=lambda x:x[1])  # fix order
        self.val_labels, self.val_videos = np.array([x for (x,y) in s]), [y for (x,y) in s]

        if shuffle_train:
            ids = self.random.permutation(len(self.train_labels))
            self.train_labels = [self.train_labels[i] for i in ids]
            self.train_videos = [self.train_videos[i] for i in ids]


        self.train_labels = np.squeeze(self.train_labels)
        self.val_labels = np.squeeze(self.val_labels)

        # Check dataset consistency
        assert len(self.train_labels) == len(self.train_videos) == 773, '%d, %d' % (len(self.train_labels), len(self.train_videos))
        assert(len(self.val_labels) == len(self.val_videos) == 383)
        assert(len(np.unique(self.train_labels)) == len(np.unique(self.val_labels)) == len(self.classes) == 7)

        print('DATA loaded: %d training videos, %d validation videos, %d classes' % (len(self.train_videos), len(self.val_labels), len(self.classes)))

        self.crop_size = 224


    # Gets a batch of samples from the dataset
    def get_batch_videos(self, iterable_samples, iterable_labels, batch_size):
        l = len(iterable_samples)
        for ndx in range(0, l, batch_size):
            yield iterable_samples[ndx:min(ndx + batch_size, l)], iterable_labels[ndx:min(ndx + batch_size, l)]


    def im_crop(self, im, MEAN_BGR, center=None, flip=None, train=False):
        # resize
        k = 256./np.min((im.shape[1],im.shape[0]))
        im = cv2.resize(im,(int(im.shape[1]*k),int(im.shape[0]*k))).astype(np.float32)
        # crop
        if train:
            if center== None:
                center = [np.random.randint(0, high=im.shape[0]-self.crop_size), np.random.randint(0, high=im.shape[1]-self.crop_size)]
            if flip or (flip==None and np.random.rand() > 0.5): # random flip
                im = im[:,::-1,:] # height x width x channels
        elif center== None:
            center = [int(np.round((im.shape[0]-self.crop_size)/2.)), int(np.round((im.shape[1]-self.crop_size)/2.))]
        im = im.transpose((2,0,1))[None,:,center[0]:center[0]+self.crop_size, center[1]:center[1]+self.crop_size]
        return im-MEAN_BGR

    # Gets random frame (random crop) from each video
    def get_frames_train(self, frames_dir, videos, MEAN_BGR):
        frames = np.zeros((len(videos),3,self.crop_size,self.crop_size), dtype='float32')
        for v_id,video in enumerate(videos):
            frames_dir_video = os.path.join(frames_dir, video)
            n_frames = len(os.listdir(frames_dir_video))
            frame_path = '%s/%08d.png' % (frames_dir_video, np.random.permutation(n_frames)[0]) # get random frame
            frame = cv2.imread(frame_path)
            if frame is None:
                raise ValueError('frame invalid')
            frames[v_id] = self.im_crop(frame, MEAN_BGR, train=True)

        return frames

    # Gets at most N_FRAMES frames (central crops) from each video
    def get_frames_val(self, frames_dir, videos, MEAN_BGR, video_labels, STRIDE=10, N_FRAMES=5):
        frames = []
        labels = []
        for v_id,video in enumerate(videos):
            frames_dir_video = os.path.join(frames_dir, video)
            n_frames = len(os.listdir(frames_dir_video))
            frames.append([])
            labels.append([])
            for frame_id in range(0,n_frames,STRIDE)[:N_FRAMES]:
                frame_path = '%s/%08d.png' % (frames_dir_video,frame_id)
                frame = cv2.imread(frame_path)
                if frame is None:
                    raise ValueError('frame invalid')
                frames[-1].append(self.im_crop(frame, MEAN_BGR).astype(np.float32))
                labels[-1].append(video_labels[v_id])

            frames[-1] = np.concatenate(frames[-1])
            labels[-1] = np.asarray(labels[-1])

        return frames, labels

    # Gets random set of frames (random crops) from each video
    def get_frames_rnn(self, frames_dir, videos, MEAN_BGR, train, STRIDE=1, N_FRAMES=2):
        frames = np.zeros((len(videos)*N_FRAMES,3,self.crop_size,self.crop_size), dtype='float32')
        ids = []
        center, flip = None, None
        for v_id, video in enumerate(videos):
            frames_dir_video = os.path.join(frames_dir, video)
            n_frames = len(os.listdir(frames_dir_video))
            frame_ref = np.random.permutation(np.max((1, n_frames-N_FRAMES*STRIDE+1)))[0] # get random reference frame

            for frame_id in range(1, np.min((n_frames, N_FRAMES))):

                frame_path = '%s/%08d.png' % (frames_dir_video, frame_ref+frame_id*STRIDE)
                frame = cv2.imread(frame_path)
                if frame is None:
                    print(frame_path)
                    raise ValueError('frame invalid: %s' % frame_path)

                if train and frame_id == 0:
                    center = [np.random.randint(0, high=frame.shape[0]-self.crop_size), np.random.randint(0, high=frame.shape[1]-self.crop_size)]
                    flip = np.random.rand() > 0.5

                try:
                    frames[v_id*N_FRAMES+frame_id] = self.im_crop(frame, MEAN_BGR, center=center, flip=flip, train=train)
                except:
                    print(frame_path, frame.shape, center)
                    raise

                ids.append(v_id)

        return frames, ids

if __name__ == "__main__":
    dloader = DataLoader(shuffle_train=True)
