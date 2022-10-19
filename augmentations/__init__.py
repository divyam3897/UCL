from .simsiam_aug import SimSiamTransform
from .eval_aug import Transform_single


def get_aug(name='simsiam', image_size=224, train=True, train_classifier=None, **aug_kwargs):
    if train==True:
        augmentation = SimSiamTransform(image_size, **aug_kwargs)
    elif train==False:
        if train_classifier is None:
            raise Exception
        augmentation = Transform_single(image_size, train=train_classifier)
    else:
        raise Exception
    
    return augmentation








