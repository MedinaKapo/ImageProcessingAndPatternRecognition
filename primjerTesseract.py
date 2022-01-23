import cv2
import pytesseract as p
from matplotlib import pyplot as plt
image_file = "test.png"
img = cv2.imread(image_file)

def prikazi(im_path):
    dpi = 80
    slika = plt.imread(im_path)
    visina,sirina = slika.shape[:2]
    figsize = sirina / float(dpi), visina / float(dpi)
    fig = plt.figure(figsize=[6.8,4.8])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(slika, cmap='gray')
    plt.show()

invertovana_slika = cv2.bitwise_not(img)
cv2.imwrite("invertovana.png", invertovana_slika)
ocr_result = p.image_to_string(invertovana_slika)
print("Tekst kod invertovane: ","\n")
print(ocr_result)
prikazi('invertovana.png')
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = grayscale(img)
cv2.imwrite("gray.jpg", gray_image)
ocr_result = p.image_to_string(gray_image)
print("Tekst kod grayscale slike: ","\n")
print(ocr_result)
prikazi('gray.jpg')
#binarizacija
thresh, im_bw = cv2.threshold(gray_image, 100, 300, cv2.THRESH_BINARY)
cv2.imwrite("bw_image.jpg", im_bw)
ocr_result = p.image_to_string(im_bw)
print("Tekst kod binarizirane: ","\n")
print(ocr_result)
prikazi('bw_image.jpg')
def ukloniSum(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)

no_noise = ukloniSum(im_bw)
cv2.imwrite("sumaBez.jpg", no_noise)
ocr_result = p.image_to_string(no_noise)
print("Tekst kod slike sa uklonjenim sumom: ","\n")
print(ocr_result)
prikazi('sumaBez.jpg')
def tankifont(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)
eroded_image = tankifont(no_noise)
cv2.imwrite("erodirana.jpg", eroded_image)
ocr_result = p.image_to_string(eroded_image)
print("Tekst kod erodirane slike: ","\n")
print(ocr_result)
prikazi('erodirana.jpg')
def debelifont(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)
dilated_image = debelifont(no_noise)
cv2.imwrite("dilatirana.jpg", dilated_image)
ocr_result = p.image_to_string(dilated_image)
print("Tekst kod dilatirane slike: ","\n")
print(ocr_result)
prikazi('dilatirana.jpg')
#https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df
import numpy as np
new = cv2.imread("rotate2.jpg")
#display("data/page_01_rotated.JPG")
def dajKosiUgao(cvImage) -> float:
    # Pripremite sliku, kopirajte, pretvorite u sivu skalu, zamućenje i prag
    novaSlika = cvImage.copy()
    gray = cv2.cvtColor(novaSlika, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


    # Primijenite dilate da spojite tekst u smislene redove/pasuse..
    # Ali koristite manje jezgro na Y osi za razdvajanje između različitih blokova teksta
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Pronađite sve konture
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        cv2.rectangle(novaSlika,(x,y),(y+w,x+h),(0,255,0),2)

    # Pronađite najveću konturu i okruženje u kutiji za minimalnu površinu
    largestContour = contours[0]
    print (len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    cv2.imwrite("boxes.jpg", novaSlika)
    # Odredite ugao. Pretvorite ga u vrijednost koja je izvorno korištena za dobivanje iskrivljene slike
    ugao = minAreaRect[-1]
    if ugao < -45:
        ugao = 90 + ugao
    return -1.0 * ugao
#Rotirajte sliku oko njenog centra
def rotirajSliku(cvImage, ugao: float):
    novaSlika = cvImage.copy()
    (h, w) = novaSlika.shape[:2]
    centar = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centar, ugao, 1.0)
    novaSlika = cv2.warpAffine(novaSlika, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return novaSlika

def ispravi(cvImage):
        ugao = dajKosiUgao(cvImage)
        return rotirajSliku(cvImage, -1.0 * ugao)

fixed = ispravi(new)
cv2.imwrite("rotated_fixed.jpg", fixed)
#prikazi("rotated_fixed.jpg")