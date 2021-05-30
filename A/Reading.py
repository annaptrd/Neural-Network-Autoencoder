
import struct as st
import numpy as np

def readMNIST(filename):


    imagesfile = open(filename, 'rb')

    # magic number
    imagesfile.seek(0)
    magic = st.unpack('>4B', imagesfile.read(4))

    print("Magic number: {0} ".format(magic))

    # dimensions
    numberOfImages = st.unpack('>I', imagesfile.read(4))[0]
    rows = st.unpack('>I', imagesfile.read(4))[0]
    columns = st.unpack('>I', imagesfile.read(4))[0]

    print("rows: {0} ".format(rows))
    print("cols: {0} ".format(columns))

    # store images
    nBytesTotal = numberOfImages*rows*columns*1
    images_collection = np.asarray(st.unpack('>'+'B'*nBytesTotal, imagesfile.read(nBytesTotal))).reshape((numberOfImages,rows, columns))
    images_collection = images_collection.reshape(images_collection.shape[0], 28, 28, 1)
    images_collection = images_collection.astype('float32')
    images_collection /= 255
    imagesfile.close()
    return images_collection

