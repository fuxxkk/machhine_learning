# -*- coding: utf-8 -*-

from PIL import Image
import struct
import os

imageFilePath = "/Applications/app/ml_data"
numOfImage = 1187
numOfLabel = 1187
rows = 60
columns = 90
imageUbyte = "train-images.idx3-ubyte"
labelUbyte = "train-labels.idx1-ubyte"
label = "label.txt"


def readImage():
    print()
    os.getcwd()
    file = open(imageUbyte, "wb")
    file.write(struct.pack('i', 50855936))
    file.write(struct.pack('i', numOfImage))
    file.write(struct.pack('i', rows))
    file.write(struct.pack('i', columns))
    for i in range(numOfImage):
        lene = Image.open(imageFilePath + str(i + 1000) + ".jpg")
        # lene=lene.convert("L")
        # if not os.path.exists()
        for j in range(rows):
            for k in range(columns):
                file.write(struct.pack('B', lene.getpixel((k, j))))
    # print lene.mode,lene.size,lene.format

    file.close()
    print("create " + imageUbyte + " success")


def readLabel():
    fileubyte = open(labelUbyte, "wb")
    filetxt = open(label, "r")

    fileubyte.write(struct.pack('i', 17301504))
    fileubyte.write(struct.pack('i', numOfLabel))

    for i in range(numOfLabel):
        initdata = filetxt.read(1)
        fileubyte.write(struct.pack('B', int(initdata)))
        filetxt.read(1)

    fileubyte.close()
    filetxt.close()

    print("create " + labelUbyte + " success")


if __name__ == '__main__':
    readImage()
    readLabel()
    print("create files finished")