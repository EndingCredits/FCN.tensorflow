import os
import numpy as np
import scipy.misc
import scipy.io as sio
import sys

# defaults
# 'CT2', 'Right2', 'Left2', '__header__', '__globals__', '__version__', 'Trachea2'
mat_file            = 'LungTrainingData.mat'
images_var_name     = 'CT2'
left_lung_var_name  = 'Left2'
right_lung_var_name = 'Right2'
trachea_var_name    = 'Trachea2'

# Loads matlab data and returns it as a dict of variables
print("Loading " + mat_file + " ...")
try:
  mat_contents = sio.loadmat(mat_file)
except:
  print("Could not load .mat file"  + mat_file + " !")
  sys.exit()

print("These are the variables in the mat file:")
for k in mat_contents.keys():
  print(k)
print("Please update the python script with these names")
print(" ")
images = mat_contents[images_var_name]
left_lung = mat_contents[left_lung_var_name]
right_lung = mat_contents[right_lung_var_name]
trachea = mat_contents[trachea_var_name]

print("Making directories...")
data_dir = 'lung_data'
try:
  os.stat(data_dir)
except:
  os.mkdir(data_dir)
    
for a in ['train', 'validation']:
  directory = data_dir + '/' + a
  try:
    os.stat(directory)
  except:
    os.mkdir(directory)
print(" ")
    

#Theses matrices are of the form [x,y,n]
# ith image can be gotten by [:, :, i]

num_images = np.shape(images)[2]
num_train_images = 500

# Get shuffled indices
shuffled = range(num_images)
np.random.shuffle(shuffled)

# Get max and mean values of images so we can normalise
m = np.max(images)
n = np.min(images)
      
print("Generating images...")
for i in range(num_images):
  print("Generating image " + str(i))
  #Split
  valid = i>num_train_images
  t_v = 'validation' if valid else 'train'
  
  # Save normalised image
  ind = shuffled[i]
  scipy.misc.toimage(images[:,:,ind], cmin=n, cmax=m).save(data_dir + '/' + t_v + '/img' + str(i) + '.png')
  
  # Save annotations
  left = left_lung[:,:,ind]
  right = right_lung[:,:,ind]
  trach = trachea[:,:,ind]
  # We map all three to one channel with this hack, so 0 = None, 1 = Left, 2 = Right, ....
  ann = left + right * 2 + trach * 3
  scipy.misc.toimage(ann, cmin=0, cmax=255).save(data_dir +'/' + t_v + '/annotation' + str(i) + '.png')
