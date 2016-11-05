import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='An fft-based descreen filter')
parser.add_argument('input')
parser.add_argument('output')
parser.add_argument('--thresh', '-t', default=92, type=int,
                    help='Threshold level for normalized magnitude spectrum')
parser.add_argument('--radius', '-r', default=6, type=int,
                    help='Radius to expand the area of mask pixels')
parser.add_argument('--middle', '-m', default=4, type=int,
                    help='Ratio for middle preservation')
args = parser.parse_args()

def normalize(img):
    h, w = img.shape
    x = np.floor(img % w)
    y = np.floor(img / w)
    cx = np.abs(x - w/2);
    cy = np.abs(y - h/2);
    energy = (cx ** 0.5 + cy ** 0.5);
    return np.maximum(energy*energy, 0.01)

def ellipse(w, h):
    offset = (w+h)/2./(w*h)
    y, x = np.ogrid[-h: h+1., -w: w+1.]
    return np.uint8((x/w)**2 + (y/h)**2 - offset <= 1)

img = np.float32(cv2.imread(args.input).transpose(2, 0, 1))
rows, cols = img.shape[-2:]
coefs = normalize(np.arange(rows * cols).reshape(rows, cols))
spectrum = np.empty_like(img)
fft = np.empty((3, rows, cols, 2))
mid = args.middle*2
rad = args.radius

for i in range(3):
    fft[i] = cv2.dft(img[i],flags = 18)
    fft[i] = np.fft.fftshift(fft[i])
    spectrum[i] = 20*np.log(cv2.magnitude(fft[i,:,:,0],fft[i,:,:,1]) * coefs)

spectrum = spectrum.transpose(1, 2, 0)
ret, thresh = cv2.threshold(cv2.cvtColor(spectrum, cv2.COLOR_BGR2GRAY), args.thresh, 255, cv2.THRESH_BINARY)
ew, eh = cols/mid, rows/mid
pw, ph = (cols-ew*2)/2, (rows-eh*2)/2
middle = np.pad(ellipse(ew, eh), ((ph,rows-ph-eh*2-1), (pw,cols-pw-ew*2-1)), 'constant')
thresh *= 1-middle
thresh = cv2.dilate(thresh, ellipse(rad,rad))
thresh = cv2.GaussianBlur(thresh, (0,0), rad/3., 0, 0, cv2.BORDER_REPLICATE)
thresh = 1 - thresh / 255

for i in range(3):
    img_back = fft[i] * np.repeat(thresh[...,None], 2, axis = 2)
    img_back = np.fft.ifftshift(img_back)
    img_back = cv2.idft(img_back)
    img[i] = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

cv2.imwrite(args.output, img.transpose(1, 2, 0))