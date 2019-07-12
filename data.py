from os.path import join

from dataset import MSRA10K, ECSSD


def get_training_set(dataset, root_dir, visual_effect):

    if dataset=='MSRA10K':
        return MSRA10K(root_dir, visual_effect, True)
    elif dataset=='ECSSD':
        return ECSSD(root_dir, visual_effect, True)



def get_test_set(dataset, root_dir, visual_effect):

    if dataset=='MSRA10K':
        return MSRA10K(root_dir, visual_effect, False)
    elif dataset=='ECSSD':
        return ECSSD(root_dir, visual_effect, False)
