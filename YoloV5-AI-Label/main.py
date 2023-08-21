# pyinstaller main.py --name AILabel --onefile --hidden-import Shapely --hidden-import yaml --hidden-import="PIL.E
# xifTags" --hidden-import seaborn
import os
import shutil
from PIL import Image
import cv2
import torch
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# If file path prompted an error, check on the terminal directory
# file path always starts on the terminal directory pointing

## yolo repo path
# yolov5_repo_path = 'ultralytics/yolov5'
yolov5_repo_path = r'C:\Users\eujseah\Desktop\Raw_yolov5\yolov5'
# model_path       = r'C:\Users\eujseah\Desktop\AI_assist_label\YoloV5-AI-Label\best.pt'
model = torch.hub.load(yolov5_repo_path, 'custom', path='YoloV5-AI-Label/best.pt', source='local', force_reload=True)

model_trained_working_size = 1280

CLASSNUMBER = 0
LABELS = []
CLASSES = []
# NextImage = False

# Extract out all variables might changed

# Rename this variable if there is a need to change the original folder file
raw_image_folder = "YoloV5-AI-Label/raw"

# The actual folder that AI will assist labeling on this
resized_image_folder = "YoloV5-AI-Label/images"

# Get classes from Classes.txt file
with open('YoloV5-AI-Label/Classes.txt') as f:
    lines = f.readlines()
    for line in lines:
        CLASSES.append(line.rstrip())


def resize_images_in_folder(input_folder, output_folder, size_image=1280):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    size=(size_image, size_image)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".jpg")

            with Image.open(input_path) as img:
                img = img.resize(size, Image.LANCZOS)
                img.save(output_path, "JPEG")



# Updates the slider and text at the top right of the screen
def updateclass(input):
    global CLASSNUMBER
    global LABELS
    CLASSNUMBER = input
    drawImage(LABELS)
    return


downcoords = None
upcoords = None


def mousefunction(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONDOWN:
        global LABELS
        Labels = LABELS
        image = cv2.imread(CURRENTIMAGE)
        NewLabels = []

        # Checking if the location right clicked was in or near one of the bounded boxes
        # If so, remove the label and redraw the boxes
        for label in Labels:
            point = Point(x, y)
            slabel = label.split(" ")
            height, width, channels = image.shape
            x_center, y_center, w, h = float(slabel[1]) * width, float(slabel[2]) * height, float(
                slabel[3]) * width, float(slabel[4]) * height
            x1 = round(x_center - w / 2)
            y1 = round(y_center - h / 2)
            x2 = round(x_center + w / 2)
            y2 = round(y_center + h / 2)

            polygon = Polygon([(x1, y2), (x2, y2), (x2, y1), (x1, y1)])
            if not polygon.buffer(20.0).contains(point):
                NewLabels.append(label)
            else:
                print("Removing Label")

        LABELS = NewLabels  # Update Labels List
        drawImage(NewLabels)  # ReDraw Image with new labels
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        global downcoords
        downcoords = [x, y] # Get pressed coordinates
    elif event == cv2.EVENT_LBUTTONUP:
        global upcoords
        upcoords = [x, y]   # Get released coordinates
        Labels = LABELS
        NewLabels = []
        image = cv2.imread(CURRENTIMAGE)
        inputX = image.shape[1]
        inputY = image.shape[0]
        x1 = downcoords[0] 
        y1 = downcoords[1]
        x2 = upcoords[0]
        y2 = upcoords[1]
        width = (abs(x2 - x1))
        height = (abs(y2 - y1))

        # Find relative portion and coordination of the pressed and released state
        yoloX = "{:.6f}".format(((x1 + x2) / 2) / inputX)
        yoloY = "{:.6f}".format(((y1 + y2) / 2) / inputY) 
        yoloWidth = "{:.6f}".format(width / inputX)
        yoloHeight = "{:.6f}".format(height / inputY)
        global CLASSNUMBER

        # Arrange into yolov5 label format
        label = str(CLASSNUMBER) + " " + yoloX + " " + yoloY + " " + yoloWidth + " " + yoloHeight
        Labels.append(label)
        LABELS = Labels  # Update labels list with added label

        drawImage(Labels)
        return


def drawImage(labels):
    global CURRENTIMAGE
    image = cv2.imread(CURRENTIMAGE)
    for label in labels:
        label = label.split(" ")
        height, width, channels = image.shape
        x_center, y_center, w, h = float(label[1])*width, float(label[2])*height, float(label[3])*width, float(label[4])*height
        x1 = round(x_center-w/2)
        y1 = round(y_center-h/2)
        x2 = round(x_center+w/2)
        y2 = round(y_center+h/2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.putText(
            img=image,
            text=CLASSES[int(label[0])],
            org=(x1, y1 - 10),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1.0,
            color=(0, 0, 0),
            thickness=3
        )

        cv2.putText(
            img=image,
            text=CLASSES[int(label[0])],
            org=(x1, y1 - 10),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1.0,
            color=(255, 255, 255),
            thickness=2
        )

    global CLASSNUMBER
    cv2.rectangle(image, (0, 0), (400, 40), (255, 255, 255), -1)
    im = cv2.putText(
        img=image,
        text=CLASSES[CLASSNUMBER],
        org=(5, 25),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=1.0,
        color=(0, 0, 0),
        thickness=2
    )

    cv2.imshow("AI Label", image)


def AIDetections(filename):
    directory = resized_image_folder
    im = cv2.imread(os.path.join(directory, filename))
    inputX = im.shape[1]
    inputY = im.shape[0]
    Labels = []
    results = model(os.path.join(directory, filename))
    detections = results.xyxy[0]
    for detection in detections:
        x1 = int(detection[0].item())
        y1 = int(detection[1].item())
        x2 = int(detection[2].item())
        y2 = int(detection[3].item())
        width = (abs(x2 - x1))
        height = (abs(y2 - y1))
        yoloX = "{:.6f}".format(((x1 + x2) / 2) / inputX)
        yoloY = "{:.6f}".format(((y1 + y2) / 2) / inputY)
        yoloWidth = "{:.6f}".format(width / inputX)
        yoloHeight = "{:.6f}".format(height / inputY)
        classNum = int(detection[5].item())

        label = str(classNum) + " " + yoloX + " " + yoloY + " " + yoloWidth + " " + yoloHeight
        Labels.append(label)

    return Labels, im


CURRENTIMAGE = ""
def main():
    # Remove YoloV5-AI-Label/labels file (if any) from the previous run
    # MAKE SURE THE DIRECTORY IS AT THE PARENT FOLDER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if os.path.isdir("YoloV5-AI-Label/labels"):
        shutil.rmtree("YoloV5-AI-Label/labels")

    # CHANGE THIS TO THE DIRECTORY OF YOUR PHOTOS
    FileList = []

    # Comment below code if you did not intent to resize your images
    # resize_images_in_folder(raw_image_folder, resized_image_folder, model_trained_working_size)

    #Any Preprocessing steps can be written here
    #
    #
    #
    #

    # Check whether the specified path exists or not
    isExist = os.path.exists(resized_image_folder)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(resized_image_folder)
        print("Images Directory Created")

    directory = resized_image_folder
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # put all the image in sequence to the FileList to be assisted
            FileList.append(filename)

    ## If Error somehow happen, may caused by cv version not correct, or maybe the opencv-headless messing something.
    ## (it is meant for server side scripting), uninstall headless version.
    cv2.namedWindow("AI Label", cv2.WINDOW_GUI_NORMAL)
    cv2.createTrackbar('r', 'AI Label', 0, len(CLASSES)-1, updateclass)

    for image in FileList:
        global CURRENTIMAGE
        CURRENTIMAGE = os.path.join(directory, image)
        Labels, im = AIDetections(image)
        drawImage(Labels)
        global LABELS
        LABELS = Labels
        cv2.setMouseCallback('AI Label', mousefunction, param=[LABELS])

        k = cv2.waitKey(0)
        if k == 49:  # 1 to Apply

            print("APPLY")
            filename = image
            Labels = LABELS

            path = 'YoloV5-AI-Label/labels'  # SAVES LABELS TO THE LABELS FOLDER
            # Check whether the specified path exists or not
            isExist = os.path.exists(path)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(path)
                print("Labels Directory Created")

            print("\nOUTPUT TO " + f'{filename[:-4]}.txt')
            with open(os.path.join('YoloV5-AI-Label/labels/', f'{filename[:-4]}.txt'), 'w+') as f:
                for label in Labels:
                    f.write(label + "\n")
                    print(label)
                f.flush()
                print('\n')

            os.rename(CURRENTIMAGE, os.path.join('YoloV5-AI-Label/labels', image))  # Move labled image to labels folder
            continue

        elif k == ord('q'):  # Press q to exit
            print("Exiting")
            shutil.copy('YoloV5-AI-Label/Classes.txt', 'YoloV5-AI-Label/labels')
            exit()

    #Copying Classes.txt into /labels file if it exist
    if os.path.exists("YoloV5-AI-Label/labels"):
        shutil.copy('YoloV5-AI-Label/Classes.txt', 'YoloV5-AI-Label/labels')


if __name__ == "__main__":
    main()
