"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import os
import scipy.misc as misc
import scipy.io as sio

class LungDatset:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print("Options: " + str(options))
        
        self.files = records_list
        self.num_records = len(self.files)
        
        self.options = { 'region_size': (64, 64, 16) }
        self.options.update(options)
        self.region_size = self.options['region_size']
        
        self._compile_records()
        
    def _compile_records(self):
        self.images = []
        self.annotations = []
        
        for record in self.files:
            try:
                imgfile = os.path.join(record, 'CT.mat')
                img = np.array(sio.loadmat(imgfile)['V'])
                
                bonefile = os.path.join(record, 'Bone.mat')
                bone = np.array(sio.loadmat(bonefile)['Bone'])
                
                leftfile = os.path.join(record, 'Left.mat')
                left = np.array(sio.loadmat(leftfile)['Left'])
                
                rightfile = os.path.join(record, 'Right.mat')
                right = np.array(sio.loadmat(rightfile)['Right'])
                
                trachfile = os.path.join(record, 'Trachea.mat')
                trach = np.array(sio.loadmat(trachfile)['Trachea'])
                
                annotation = np.stack((
                    left,
                    right,
                    bone,
                    trach), axis=3)
                    
            except:
                print("Error loading from " + record)
                return False
                
            self.images.append(img)
            self.annotations.append(annotation)
        return True
            

    def get_records(self, indices):
        batch_images = []
        batch_annotations = []
        for ind in indices:
            raw_image = self.images[ind]
            raw_annotation = self.annotations[ind]
            l, h = self._get_random_range(self.region_size, (0,0,0), np.shape(raw_image))
            processed_img = raw_image[l[0]:h[0],l[1]:h[1],l[2]:h[2],...]
            processed_annotation = raw_annotation[l[0]:h[0],l[1]:h[1],l[2]:h[2],...]
            batch_images.append(processed_img[..., np.newaxis])
            batch_annotations.append(processed_annotation)
            
        batch_images = np.stack(batch_images, axis=0)
        batch_annotations = np.stack(batch_annotations, axis=0)

        return batch_images, batch_annotations
        
        
    def _get_random_range(self, region_size, min_extent, max_extent):
        true_max = np.subtract(max_extent, region_size)
        x = np.random.randint(min_extent[0], true_max[0])
        y = np.random.randint(min_extent[1], true_max[1])
        z = np.random.randint(min_extent[2], true_max[2])
        low_pos = (x, y, z)
        high_pos = np.add(low_pos, region_size)
        return low_pos, high_pos
        
    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.num_records:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            #self.reset_batch_offset() #not needed
            # Shuffle the data
            np.random.shuffle(self.files)
            # Start next epoch
            start = 0
            self.batch_offset = batch_size
        end = self.batch_offset
        return self.get_records(range(start, end))

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.num_records, size=[batch_size]).tolist()
        return self.get_records(indexes)
