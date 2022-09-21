#%%
from ambiguous.dataset.dataset import DatasetTriplet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


#%%
# Manually load dataset
import numpy as np

idx=2000
set_type='train'
img_size=28


image = np.load(f'/Users/guillaumeetter/Documents/datasets/amnistV2/{set_type}/{idx}_image.npy')
label = np.load(f'/Users/guillaumeetter/Documents/datasets/amnistV2/{set_type}/{idx}_label.npy')

clean1 = image[:, :, :img_size]
amb = image[:, :, img_size:2*img_size]
clean2 = image[:, :, 2*img_size:3*img_size] 

print(label)
plt.subplot(1,3,1)
plt.imshow(clean1[0,:,:])
plt.subplot(1,3,2)
plt.imshow(amb[0,:,:])
plt.subplot(1,3,3)
plt.imshow(clean2[0,:,:])

#%%
# Manually debug datasetTriplet loading function
import glob
from torch.utils.data import DataLoader
root='/Users/guillaumeetter/Documents/datasets/amnistV2'
image_list = sorted(glob.glob(root+'/train/*_image.npy'))
label_list = sorted(glob.glob(root+'/train/*_label.npy'))

data_len = len(image_list)

# %%
# Using dataloader
root='/Users/guillaumeetter/Documents/datasets/emnist'
trainset = DatasetTriplet(root=root, train=True, transform=None)
testset = DatasetTriplet(root=root, train=False, transform=None)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

(clean1, amb, clean2), label = next(iter(testloader))

idx=15
plt.figure()
plt.subplot(1,3,1)
plt.imshow(clean1[idx,0,:,:])
plt.subplot(1,3,2)
plt.imshow(amb[idx,0,:,:])
plt.subplot(1,3,3)
plt.imshow(clean2[idx,0,:,:])
print(label[idx])
# %%
# Using dataset indexing
idx=18
plt.figure()
plt.subplot(1,3,1)
plt.imshow(testset[idx][0][0][0,:,:])
plt.subplot(1,3,2)
plt.imshow(testset[idx][0][1][0,:,:])
plt.subplot(1,3,3)
plt.imshow(testset[idx][0][2][0,:,:])
print(testset[idx][1])

# %%
