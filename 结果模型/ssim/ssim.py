import cv2
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_nrmse as nrmse
def match(imfil1,imfil2):
    img1=cv2.imread(imfil1)
    (h,w)=img1.shape[:2]
    img2=cv2.imread(imfil2)
    resized=cv2.resize(img2,(w,h))
    (h1,w1)=resized.shape[:2]
    img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2=cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return ssim(img1,img2)

image1=r"H:\Code\python\PRNet_PyTorch-master\TestImages\results\ssim\image00475_uv.jpg"
image2=r"H:\Code\python\PRNet_PyTorch-master\TestImages/results/ssim/uv_posmap.jpg"
a=match(image1,image2)
print(a)