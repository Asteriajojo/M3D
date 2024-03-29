import numpy as np
import torch
import torchvision.transforms as transforms
import augmentations
import torchvision.datasets as datasets

# Transformations
class TwoCropTransform:
    def __init__(self, transform, img_size):
        self.transform = transform
        self.img_size = img_size
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.img_size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.ToTensor()])

    def __call__(self, x):
        return [self.transform(x), self.data_transforms(x)]

def rotation(input):
    batch = input.shape[0]
    target = torch.tensor(np.random.permutation([0,1,2,3] * (int(batch / 4) + 1)), device = input.device)[:batch]
    target = target.long()
    image = torch.zeros_like(input)
    image.copy_(input)
    for i in range(batch):
        image[i, :, :, :] = torch.rot90(input[i, :, :, :], target[i], [1, 2])

    return image, target

target_dict = { 3 : 'n01491361',
               16 : 'n01560419',
               24 : 'n01622779',
               36 : 'n01667778',
               48 : 'n01695060',
               52 : 'n01728572',
               69 : 'n01768244',
               71 : 'n01770393',
               85 : 'n01806567',
               99 : 'n01855672',
               107 : 'n01910747',
               114 : 'n01945685',
               130 : 'n02007558',
               138 : 'n02018795',
               142 : 'n02033041',
               151 : 'n02085620',
               162 : 'n02088364',
               178 : 'n02092339',
               189 : 'n02095570',
               193 : 'n02096294',
               207 : 'n02099601',
               212 : 'n02100735',
               228 : 'n02105505',
               240 : 'n02107908',
               245 : 'n02108915',
               260 : 'n02112137',
               261 : 'n02112350',
               276 : 'n02117135',
               285 : 'n02124075',
               291 : 'n02129165',
               309 : 'n02206856',
               317 : 'n02259212',
               328 : 'n02319095',
               340 : 'n02391049',
               344 : 'n02398521',
               358 : 'n02443114',
               366 : 'n02480855',
               374 : 'n02488291',
               390 : 'n02526121',
               393 : 'n02607072',
               404 : 'n02690373',
               420 : 'n02787622',
               430 : 'n02802426',
               438 : 'n02815834',
               442 : 'n02825657',
               453 : 'n02870880',
               464 : 'n02910353',
               471 : 'n02950826',
               485 : 'n02988304',
               491 : 'n03000684',
               506 : 'n03065424',
               513 : 'n03110669',
               523 : 'n03141823',
               538 : 'n03220513',
               546 : 'n03272010',
               555 : 'n03345487',
               569 : 'n03417042',
               580 : 'n03457902',
               582 : 'n03461385',
               599 : 'n03530642',
               605 : 'n03584254',
               611 : 'n03598930',
               629 : 'n03676483',
               638 : 'n03710637',
               646 : 'n03733281',
               652 : 'n03763968',
               661 : 'n03777568',
               678 : 'n03814639',
               689 : 'n03866082',
               701 : 'n03888257',
               707 : 'n03902125',
               717 : 'n03930630',
               724 : 'n03947888',
               735 : 'n03980874',
               748 : 'n04026417',
               756 : 'n04049303',
               766 : 'n04111531',
               779 : 'n04146614',
               786 : 'n04179913',
               791 : 'n04204347',
               802 : 'n04252077',
               813 : 'n04270147',
               827 : 'n04330267',
               836 : 'n04355933',
               849 : 'n04398044',
               859 : 'n04442312',
               866 : 'n04465501',
               879 : 'n04507155',
               885 : 'n04525038',
               893 : 'n04548362',
               901 : 'n04579145',
               919 : 'n06794110',
               929 : 'n07615774',
               932 : 'n07695742',
               946 : 'n07730033',
               958 : 'n07802026',
               963 : 'n07873807',
               980 : 'n09472597',
               984 : 'n11879895',
               992 : 'n12998815'
}