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

def normalize(h, w):
    x = np.arange(w)
    y = np.arange(h)
    cx = np.abs(x - w/2) ** 0.5
    cy = np.abs(y - h/2) ** 0.5
    energy = sum(np.meshgrid(cx, cy))
    return np.maximum(energy*energy, 0.01)

def ellipse(w, h):
    offset = (w+h)/2./(w*h)
    y, x = np.ogrid[-h: h+1., -w: w+1.]
    return np.uint8((x/w)**2 + (y/h)**2 - offset <= 1)

img = np.float32(cv2.imread(args.input).transpose(2, 0, 1))
rows, cols = img.shape[-2:]
coefs = normalize(rows, cols)
mid = args.middle*2
rad = args.radius
ew, eh = cols/mid, rows/mid
pw, ph = (cols-ew*2)/2, (rows-eh*2)/2
middle = np.pad(ellipse(ew, eh), ((ph,rows-ph-eh*2-1), (pw,cols-pw-ew*2-1)), 'constant')

for i in range(3):
    fftimg = cv2.dft(img[i],flags = 18)
    fftimg = np.fft.fftshift(fftimg)
    spectrum = 20*np.log(cv2.magnitude(fftimg[:,:,0],fftimg[:,:,1]) * coefs)

    ret, thresh = cv2.threshold(np.float32(np.maximum(0, spectrum)), args.thresh, 255, cv2.THRESH_BINARY)
    thresh *= 1-middle
    thresh = cv2.dilate(thresh, ellipse(rad,rad))
    thresh = cv2.GaussianBlur(thresh, (0,0), rad/3., 0, 0, cv2.BORDER_REPLICATE)
    thresh = 1 - thresh / 255

    img_back = fftimg * np.repeat(thresh[...,None], 2, axis = 2)
    img_back = np.fft.ifftshift(img_back)
    img_back = cv2.idft(img_back)
    img[i] = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

cv2.imwrite(args.output, img.transpose(1, 2, 0))
