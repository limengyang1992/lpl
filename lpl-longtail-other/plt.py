# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 20:21:29 2021

@author: lenovo
"""
import numpy as np
import matplotlib.pyplot as plt
plt.figure(dpi = 1080)
def plt_img(file, img_name,final_epoch = 160, start_epoch = 0, class_num = 10):
    mean_res = np.zeros(shape = (class_num))
    data = np.load(file, allow_pickle=True).item()
    for e in range(start_epoch, final_epoch):
        res_epoch = data[e]
        init_loss = res_epoch['init_loss']
        finl_loss = res_epoch['finl_loss']
        labels = res_epoch['labels']
        for idx in range(len(labels)):
            label_idx = labels[idx]
            if init_loss[idx] == 0.:
                continue
            mean_res[label_idx] += (finl_loss[idx] - init_loss[idx]) / init_loss[idx]
                
    mean_res /= (final_epoch - start_epoch)
    
    for i in range(class_num):
        mean_res[i] = mean_res[i] / len(labels[labels == i])
    print(mean_res)
    plt.plot(mean_res,color='b', linewidth=2)
    plt.xlabel('Class Id')
    plt.ylabel('Relative Increment')
    plt.savefig(img_name)
    
if __name__ == '__main__':
    file = './res100_110.npy'
    img_name = './1cifar100_resnet110.png'
    plt_img(file, img_name, final_epoch = 1200, start_epoch = 1000, class_num = 100)
    


