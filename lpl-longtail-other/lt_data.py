from datasets.builder import build_datasets
from datasets.loader.build_loader import build_dataloader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def imshow(inp,name=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.savefig(name+".png")


trainset, samples_per_cls = build_datasets(name="CIFAR100", mode='train',
                                           num_classes=100,
                                           imbalance_ratio=0.1,
                                           root='../data')
testset, _ = build_datasets(name="CIFAR100", mode='test',
                            num_classes=100, root='../data')

train_loader = build_dataloader(trainset, imgs_per_gpu=128, dist=False, shuffle=True)
val_loader = build_dataloader(testset, imgs_per_gpu=128, dist=False, shuffle=False)

print(samples_per_cls)

# label_freq = {}
# for _, (imgs, target) in enumerate(train_loader):
#     target = target.cuda()
#     for i,j in enumerate(target):
#         key = str(j.item())
#         if key == "9":
#             imshow(imgs[i],"1"+str(i))
            
#             label_freq[key] = label_freq.get(key, 0)+1
# label_freq = dict(sorted(label_freq.items()))
# label_freq_array = list(label_freq.values())
# print(label_freq_array)


# for _, (imgs, target) in enumerate(train_loader):
#     target = target.cuda()
#     for i,j in enumerate(target):
#         key = str(j.item())
#         if key == "9":
#             imshow(imgs[i],"2"+str(i))