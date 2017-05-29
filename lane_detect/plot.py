from matplotlib.pyplot import imshow, show

def show_image(image, gray=False):
    if gray:
        imshow(image, cmap='gray')
    else:
        imshow(image)
    show()
