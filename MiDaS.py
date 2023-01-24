# Import dependencies
import cv2
import torch
import matplotlib.pyplot as plt 

# Download the MiDaS
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cuda')
midas.eval()
# Input transformation pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform 

#Map output into distance
def remap(x):
    in_min = 0
    in_max = 640
    out_min = 0
    out_max = 30
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
# Hook into OpenCV
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while cap.isOpened(): 
    ret, frame = cap.read()
    distance=[]
    # Transform input for midas 
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cuda')

    # Make a prediction
    with torch.no_grad(): 
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size = img.shape[:2], 
            mode='bicubic', 
            align_corners=False
        ).squeeze()

        output = prediction.cpu().numpy()

    
    plt.imshow(output)
    cv2.imshow('CV2Frame', frame)
    plt.pause(0.00001)
    distance = list(map(remap,output.flatten()))
    print(output)
    if cv2.waitKey(10) & 0xFF == ord('q'): 
        cap.release()
        cv2.destroyAllWindows()
#neu muon co ket qua cua 1 pixel bat ky thi lay ket qua trong list distance, vi
#tri se la x(x1,y1): width.x1+y1+1
