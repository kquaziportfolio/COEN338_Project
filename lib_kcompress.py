import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
from sklearn.mixture import GaussianMixture
import math
import cv2

plt.style.use("ggplot")
SHOW_COLOR = True  # True if we want to show colorspace (before AND after)
SHOW_PREIMAGE = False  # Show image before compression
SHOW_POSTIMAGE = False  # Show image after compression


# COLOR CLUSTERING PHASE
def colorshow(im):
    arr = im.transpose()
    print(arr.shape)
    ax = plt.axes(projection="3d")
    ax.scatter(arr[0], arr[1], arr[2], c=im)
    plt.show()


def preprocess(path):
    plt.rcParams["figure.figsize"] = (20, 12)
    img: np.ndarray = io.imread(path)
    if SHOW_PREIMAGE:
        labels = plt.axes(xticks=[], yticks=[])
        labels.imshow(img)
        plt.show()
    print("Image shape: ", img.shape)
    print("Image size: ", img.size)
    img_data = (img / 255).reshape(img.shape[0] * img.shape[1], img.shape[2])

    # display initial colorspace
    if SHOW_COLOR:
        colorshow(img_data)

    return img_data, img.shape


def cluster(img_data: np.ndarray, cluster_count: int = 160):
    arr = img_data.transpose()
    print("Clustering...")
    gmmR = GaussianMixture(n_components=cluster_count, verbose=1)
    gmmG = GaussianMixture(n_components=cluster_count, verbose=1)
    gmmB = GaussianMixture(n_components=cluster_count, verbose=1)
    print("Fitting model ...")
    print("Red...")
    gmmR.fit(arr[0].reshape(-1, 1))
    print("Green")
    gmmG.fit(arr[1].reshape(-1, 1))
    print("Blue")
    gmmB.fit(arr[2].reshape(-1, 1))
    print("Predicting data ...")
    labelsR = gmmR.predict(arr[0].reshape(-1, 1))
    labelsG = gmmG.predict(arr[1].reshape(-1, 1))
    labelsB = gmmB.predict(arr[2].reshape(-1, 1))

    return (gmmR, gmmB, gmmG), (labelsR, labelsB, labelsG)


def replace_with_cluster(img_data: np.ndarray, labels, shape):
    for i in range(labels[0].shape[0]):
        img_data[i][0] = labels[0][i]
        img_data[i][1] = labels[1][i]
        img_data[i][2] = labels[2][i]

    if SHOW_COLOR:
        colorshow(img_data)

    # Reformat array
    img = img_data.reshape(shape) * 255
    img = img.astype("uint8")
    if SHOW_POSTIMAGE:
        labels = plt.axes(xticks=[], yticks=[])
        labels.imshow(img)
        plt.show()
    return img


def color_cluster(fp, ccount):
    img_data, shape = preprocess(fp)
    models, labels_tup = cluster(img_data, ccount)
    centers = [x.means_ for x in models]
    ncolors = np.array([[centers[x][i] for i in labels_tup[x]] for x in [0, 1, 2]])
    img = replace_with_cluster(np.copy(img_data), ncolors, shape)
    return img, labels_tup, models, shape


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def psnr(original, compressed):
    mse = np.mean((original-compressed)**2)
    mpixel = 255
    return 10*math.log10(mpixel/math.sqrt(mse))


def compress(infp, outfp, ccount=160):
    img, labels_tup, models, shape = color_cluster(infp, ccount)
    Image.fromarray(img).save(outfp)
    original = io.imread(infp)
    compressed = io.imread(outfp)  # can never be too safe
    # original = cv2.cvtColor(original, cv2.COLOR_RGB2YCrCb)
    # compressed = cv2.cvtColor(compressed, cv2.COLOR_RGB2YCrCb)
    # ps = (6*psnr(original[0], compressed[0])+psnr(original[1], compressed[1])+psnr(original[2], compressed[2]))/8
    ps = cv2.PSNR(original, compressed)

    CENTER_COUNT_COUNT = math.ceil((len(bin(models[0].n_components))-2)/8)
    tCENTER_COUNT_COUNT = len(bin(models[0].n_components))-2
    CENTER_COUNT = models[0].n_components
    HEIGHT = shape[0]
    WIDTH = shape[1]

    headersize = 0
    headersize += 8  # center_count_count
    headersize += tCENTER_COUNT_COUNT  # center_count
    headersize += 32  # height and width
    headersize += (tCENTER_COUNT_COUNT + 3)*models[0].n_components
    theadersize = math.ceil(headersize/8)

    bodysize = HEIGHT*WIDTH*tCENTER_COUNT_COUNT
    tbodysize = math.ceil(bodysize/8)

    totalsize = headersize+bodysize
    ttotalsize = math.ceil(totalsize/8)
    print()
    print(f"Header Size: {sizeof_fmt(theadersize)}")
    print(f"\tCENTER_COUNT_COUNT: 1")
    print(f"\tCENTER_COUNT: {CENTER_COUNT_COUNT}")
    print(f"\tHEIGHT: 2")
    print(f"\tWIDTH: 2")
    print(f"\tCENTER ALIASING: {(CENTER_COUNT_COUNT + 3)*models[0].n_components}")
    print(f"\tONE CENTER PAGE: {CENTER_COUNT_COUNT+3}")
    print()
    print(f"Body Size: {sizeof_fmt(tbodysize)}")
    print(f"\tONE PIXEL: {tCENTER_COUNT_COUNT} bits")
    print(f"\tONE ROW: {tCENTER_COUNT_COUNT*WIDTH} bits")
    print(f"\tONE COLUMN: {tCENTER_COUNT_COUNT*HEIGHT} bits")
    print()
    print(f"Total Size: {sizeof_fmt(ttotalsize)}")
    print(f"Naive Size: {sizeof_fmt(HEIGHT*WIDTH*3)}")
    print(f"Compression Ratio: {round((HEIGHT*WIDTH*3)/ttotalsize,4)}x")
    print(f"Weighted PSNR: {ps}")
    print(f"BPS: {round(totalsize/(HEIGHT*WIDTH),4)}")


if __name__ == "__main__":
    compress("imgs/birb.png", "test.png", 160)
