import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy

def add_seed_name(name, seed):
    '''
    Add seed number to logging file
    '''
    if name == '':
        return ''
    elif 'baseline' in name:
        new_name = name.replace('baseline', 'baseline%d' %seed)
    elif 'exp' in name:
        new_name = name.replace('exp', 'exp%d' %seed)
    elif 'moments_combined' in name:
        new_name = name.replace(name.split('/')[-1].split('_')[3], name.split('/')[-1].split('_')[3] + str(seed))
    elif 'moments' in name or 'combined' in name:
        new_name = name.replace(name.split('/')[-1].split('_')[2], name.split('/')[-1].split('_')[2] + str(seed))
    else:
        new_name = name.replace(name.split('/')[-1].split('_')[1], name.split('/')[-1].split('_')[1] + str(seed))

    return new_name


def plot_confusion_matrix(conf_matrix, classes, saved_exp_name,
                      normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    '''
    Display confusion matrix
    '''

    plt.figure()
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 fontsize=8,
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if normalize:
        plt.savefig('%s.png' %saved_exp_name)
    else:
        plt.savefig('%s.png' %saved_exp_name)
