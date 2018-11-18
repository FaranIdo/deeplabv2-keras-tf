import numpy

# voc12 palette - 21 labels
palette = numpy.array([[0, 0, 0],  # "background"
                    [128, 0, 0],  # "aeroplane"
                    [0, 128, 0],  # "bicycle"
                    [128, 128, 0],  # "bird"
                    [0, 0, 128],  # "boat"
                    [128, 0, 128],  # "bottle"
                    [0, 128, 128],  # "bus"
                    [128, 128, 128],  # "car"
                    [64, 0, 0],  # "cat"
                    [192, 0, 0],  # "chair"
                    [64, 128, 0],  # "cow"
                    [192, 128, 0],  # "diningtable"
                    [64, 0, 128],  # "dog"
                    [192, 0, 128],  # "horse"
                    [64, 128, 128],  # "motorbike"
                    [192, 128, 128],  # "person"
                    [0, 64, 0],  # "potted plant"
                    [128, 64, 0],  # "sheep"
                    [0, 192, 0],  # "sofa"
                    [128, 192, 0],  # "train"
                    [0, 64, 128]], dtype='uint8')  # "tv/monitor"

VOID_COLOR = 255
VOID_RGB = [224, 224, 192]


def calculate_accuracy(y_true, y_pred):
    tp = 0.0
    fp = 0.0
    ignore = 0

    for index in numpy.ndindex(y_true.shape):
        if y_true[index] == VOID_COLOR:
            ignore += 1
        elif y_true[index] == y_pred[index]:
            tp += 1
        else:
            fp += 1

    accuracy = tp / (tp + fp)
    ignore_percentage = float(ignore) / y_true.size * 100
    print "Ignore count (void pixels): {} out of {} pixels ({}%)".format(ignore, y_true.size, ignore_percentage)
    print "Accuracy: {}".format(accuracy)


# Source: https://github.com/tensorflow/models/blob/master/research/deeplab/utils/get_dataset_colormap.py
DATASET_MAX_ENTRIES = 256


def bit_get(val, idx):
    return (val >> idx) & 1


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
    A colormap for visualizing segmentation results.
    """
    colormap = numpy.zeros((DATASET_MAX_ENTRIES, 3), dtype=int)
    ind = numpy.arange(DATASET_MAX_ENTRIES, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap


if __name__ == '__main__':
    print create_pascal_label_colormap()  # This creates the entire 256 colors, we use only the first 21
