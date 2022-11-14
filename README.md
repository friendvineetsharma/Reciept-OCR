# Reciept-OCR
Optical Character Recognition technique to read Receipts.

An application which can extract text from the receipt and presents a json output in a uniform structure.

Packages:
Numpy
OpenCV-Python
Matplotlib
sys
skimage
PIL
pytessarect
datefinder
spacy

Used various methods of image processing such as Gaussian Blur, image dilation, Canny edge filtering, Contour detection, Perspective transform (Important). 
To extract the text from the image we have used pytesseract.

TO RUN THE CODE TYPE COMMAND IN COMMAND PROMPT
python Receipt-OCR.py (Path of image)

After running the code you will get two files:
1. result.png which shows the preprocessed image.
2. output.txt which shows the information stored in the image.
