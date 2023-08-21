# YoloV5-AI-Label
YoloV5 AI Assisted Labeling

Get yolov5 to label the image, then manually correction on the image

1. Enter your classes in the /Classes.txt file, starting with class 0
2. Replace the /best.pt YoloV5 model with your own YoloV5 model
3. Place your unlabled and raw images in the /raw folder
4. Check on the parameters in /main.py to confirm the value being pass
5. Run /main.py to start the program.
6. Use the slider to select the class you want to label
7. Click and drag on the image to draw label boxes
8. Right click on a box to remove it
9. Once everything in the frame is labeled, press 1 to apply (press q to quit the window safely)
10. Image and its label file will be moved to the /labels folder
11. The /Classes.txt will also be moved into the /labels folder. (Comment the shutil.copy() if not needed the Classes.txt file)

With these labels, you can train your model with [YoloV5](https://github.com/ultralytics). 
The /labels file can be directly being pulled or pushed into [Roboflow](https:app.roboflow.com) to generate dataset and do version control and split dataset easily.
Once the new model is trained, replace the /best.pt file from the new model.
Remove the files in /raw if needed
Start labeling again. As the model gets better, less manual labeling will be needed.


###### Requirements
- OpenCV
- Shapely
- PyTorch (Optional to download cuda and cudnn version to inference on gpu, but is recommended)
- [YoloV5 Requirements](https://github.com/ultralytics/yolov5/blob/master/requirements.txt)
