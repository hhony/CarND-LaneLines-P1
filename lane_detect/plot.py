from matplotlib.pyplot import imshow, show, imread, imsave


def show_image(image, gray=False):
    if gray:
        imshow(image, cmap='gray')
    else:
        imshow(image)
    show()


def image_read(filename):
    return imread(filename)


def image_save(filename, image):
    imsave(filename, image)