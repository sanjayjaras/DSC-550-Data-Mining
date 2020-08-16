import cv2
import os
import imutils
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (256, 256)


def crop_image(image_path, IMG_SIZE=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, dsize=IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # add contour on the image
    img_cnt = cv2.drawContours(img.copy(), [c], -1, (0, 255, 255), 4)

    # add extreme points
    img_pnt = cv2.circle(img_cnt.copy(), extLeft, 8, (0, 0, 255), -1)
    img_pnt = cv2.circle(img_pnt, extRight, 8, (0, 255, 0), -1)
    img_pnt = cv2.circle(img_pnt, extTop, 8, (255, 0, 0), -1)
    img_pnt = cv2.circle(img_pnt, extBot, 8, (255, 255, 0), -1)

    # crop
    ADD_PIXELS = 0
    new_img = img[
        extTop[1] - ADD_PIXELS : extBot[1] + ADD_PIXELS,
        extLeft[0] - ADD_PIXELS : extRight[0] + ADD_PIXELS,
    ].copy()
    new_img = cv2.resize(new_img, dsize=IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
    return gray


def save_image(image_array, file_path, format="png"):
    fig, ax = plt.subplots()
    im = ax.imshow(image_array)
    plt.imsave(
        file_path, image_array, format=format,
    )
    plt.close(fig)


def augment_data(file_dir, n_generated_samples, save_to_dir):
    data_gen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        brightness_range=(0.3, 1.0),
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
    )

    for filename in os.listdir(file_dir):
        image = cv2.imread(file_dir + "/" + filename)
        # reshape the image
        image = image.reshape((1,) + image.shape)
        save_prefix = "aug_" + filename.split(".")[0]
        i = 0
        for batch in data_gen.flow(
            x=image,
            batch_size=1,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format="png",
        ):
            i += 1
            if i > n_generated_samples:
                break

