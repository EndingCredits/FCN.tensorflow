"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc
import scipy.io as sio


class LungDatset:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, matfile, options={}):
        
        images_var_name     = 'CT2'
        left_lung_var_name  = 'Left2'
        right_lung_var_name = 'Right2'
        trachea_var_name    = 'Trachea2'
        
        self.start_offset = 128
        self.output_size = 64
        self.output_depth = 10
        self.slices = lung_slices
        self.slices = np.array([ s[:self.output_depth] for s in self.slices if len(s) >= self.output_depth ])
        
        self.size = len(self.slices)
        self.batch_offset = 0
        self.epochs_completed = 0

        print("Initializing Batch Dataset Reader...")
        print("Options: " + str(options))
        try:
            mat_contents = sio.loadmat(matfile)
        except:
            print("Could not load .mat file"  + matfile + " !")
        
        self.images = mat_contents[images_var_name][..., np.newaxis]
        left = mat_contents[left_lung_var_name]
        right = mat_contents[right_lung_var_name]
        trachea = mat_contents[trachea_var_name]
        empty = np.zeros(np.shape(left))
        self.annotations = np.squeeze(np.stack((
            empty[..., np.newaxis],
            left[..., np.newaxis],
            right[..., np.newaxis],
            trachea[..., np.newaxis]), axis=3),axis=4)
            
    def _get_images(self, indices):
        batch_images = self.images[:, :, self.slices[indices], :]
        batch_annotations = self.annotations[:, :, self.slices[indices], :]
        
        #get random ranges
        box_left = self.start_offset ; box_right = 512 - self.start_offset
        x = np.random.randint(box_left, box_right) - self.output_size//2
        y = np.random.randint(box_left, box_right) - self.output_size//2 
         
        batch_images = np.transpose(batch_images[x:x+self.output_size,y:y+self.output_size,...], (2,0,1,3,4))
        batch_annotations = np.transpose(batch_annotations[x:x+self.output_size,y:y+self.output_size,...], (2,0,1,3,4))
        return batch_images, batch_annotations
        

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.size:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            np.random.shuffle(self.slices)
            # Start next epoch
            start = 0
            self.batch_offset = batch_size
        end = self.batch_offset
        
        return self._get_images(range(start,end))


    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.size, size=[batch_size]).tolist()
        return self._get_images(indexes)
        
lung_slices = [
    range(0,11),
    range(11,22),
    range(22,33),
    range(33,43),
    range(43,53),
    range(53,63),
    range(63,70),
    range(70,82),
    range(82,98),
    range(98,111),
    range(111,124),
    range(124,135),
    range(135,145),
    range(145,161),
    range(161,174),
    range(174,185),
    range(185,195),
    range(195,206),
    range(206,211),
    range(211,222),
    range(222,237),
    range(237,244),
    range(244,248),
    range(248,256),
    range(256,268),
    range(268,275),
    range(275,284),
    range(284,292),
    range(292,300),
    range(300,309),
    range(309,316),
    range(316,324),
    range(324,334),
    range(334,346),
    range(346,358),
    range(358,364),
    range(364,374),
    range(374,381),
    range(381,389),
    range(389,397),
    range(397,405),
    range(405,414),
    range(414,422),
    range(422,431),
    range(431,439),
    range(439,447),
    range(447,455),
    range(455,467),
    range(467,475),
    range(475,483),
    range(483,490),
    range(490,494),
    range(494,501),
    range(501,508),
    range(508,514)
]
