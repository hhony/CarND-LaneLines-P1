from matplotlib.pyplot import imshow, show, imread

def show_image(image, gray=False):
    if gray:
        imshow(image, cmap='gray')
    else:
        imshow(image)
    show()

def image_read(filename):
    return imread(filename)
