import os
import h5py
import numpy as np
np.random.seed(309)
import glob

loadPath = "../../../benchmark_self-supervise/datasets/femur_new/h5/train/"
savePath = "../../../benchmark_self-supervise/datasets/femur_new/processed_h5_4/train/"
def getRangImageDepth(image):
    fistflag = True
    startposition = 0
    endposition = 0
    for z in range(image.shape[2]):
        notzeroflag = np.max(image[z])
        if notzeroflag and fistflag:
            startposition = z
            fistflag = False
        if notzeroflag:
            endposition = z
    return startposition, endposition

if __name__ == "__main__":
    image_list = glob.glob(os.path.join(loadPath, "*.h5"))
    image_list.sort()
    for i in range(len(image_list)):
        image_name = image_list[i]
        h5f = h5py.File(image_list[i], 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]

        startpostion, endpostion = getRangImageDepth(label)
        rdm_start = startpostion + (endpostion - startpostion) / 5
        rdm_end = endpostion - (endpostion - startpostion) / 5
        lbl_idx = np.random.randint(rdm_start, rdm_end, size=1)
        print(rdm_start, rdm_end, lbl_idx)

        new_lbl = np.zeros(label.shape)
        if i < 16:
            new_lbl[..., lbl_idx] = label[..., lbl_idx]


        save_file = h5py.File(os.path.join(savePath, os.path.basename(image_list[i])), 'w')
        save_file.create_dataset('image', data=image)
        save_file.create_dataset('label_full', data=label)
        save_file.create_dataset('label', data=new_lbl)
        save_file.close()

