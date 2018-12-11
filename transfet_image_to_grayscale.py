from PIL import Image
im = Image.open('myplot.png')
im1 = im.convert('L')
im1.save('myplot1.png')
