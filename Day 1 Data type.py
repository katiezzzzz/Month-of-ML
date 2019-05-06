from PIL import Image
import numpy as np
#import torch
'''
im = Image.open('Day 1 picture.jpeg')
#im.show()
print(type(im))

tensor = np.array(im)
print(tensor)
# numerical representation of image
print(tensor.shape)
# 750 x 1000 x 3 channal image
# it is an RGB image

torch_tensor = torch.tensor(tensor)
print(torch_tensor)
# same thing but is a torch tensor


# CSV
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Day 1 FL_insurance_sample.csv')

my_cols = data.iloc[:,3:9]
# iloc = index location
print(my_cols)
my_array = np.array(my_cols)
print(my_array)

t_2011 = my_array[:,-2]
t_2012 = my_array[:,-1]

idx = range(len(t_2012))
diff = t_2012 - t_2011
pos = diff > 0
neg = diff < 0
print(pos)
print(neg)
pos_idxs = idx[pos]
neg_idxs = idx[neg]

plt.figure()
plt.scatter(idx, pos)
#plt.scatter(idx, t_2011, c='b')
#plt.scatter(idx, t_2012, c='r')
plt.show()
'''

# Audio
import librosa
import matplotlib.pyplot as plt

audio = librosa.load('Day 1 SampleAudio_0.4mb.mp3')
print(audio)
plt.plot(audio[0])
plt.show()

'''
# Video
import pims
import matplotlib.pyplot as plt

vid = pims.Video('Day 1 small.mp4')
frame = vid[0]
print(frame)
plt.imshow(frame)
'''