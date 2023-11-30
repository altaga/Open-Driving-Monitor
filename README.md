# Open Driving Monitor

Drowsiness, Blind Spot, and Emotions Monitor system for driving intended for conventional automobiles and and heavy machinery. It objective is to enhance safety through fatigue detection, blind spot awareness, and emotional state analysis, all powered by OpenCV.

<img src="https://i.ibb.co/ZBxgtS4/logo-1.png" width="1000">

# Fast Links:

### Video Demo:

Video Demo: Click on the image
[![Car](https://i.ibb.co/ZBxgtS4/logo-1.png)](pending...)


### Test Notebooks:

- Emotions: [CLICK HERE](./Emotions/test/Test_Emotions_DNN.ipynb)

- Drowsiness: [CLICK HERE](./Drowsiness/test/Test_Blink_DNN.ipynb)

- YoloV3: [CLICK HERE](./Yolo/test/LoadAndTestYolo.ipynb)

### WebPage:

- Open Driving Navigator: [CLICK HERE](https://open-driving-navigator.vercel.app/)

### Native App:

- Open Driving Emulator: [CLICK HERE](https://play.google.com/store/apps/details?id=com.altaga.ODS)

# Introduction:

Driving has evolved into a daily routine for humans, comparable to eating, brushing our teeth, or sleeping. However, it has transformed into a time-consuming activity that occupies a significant portion of our daily lives. Moreover, if specific safety protocols are neglected, driving can pose potential risks.

<img src="https://i.ibb.co/HKC3yBh/e4264624281d816222229deed61c8e32.webp" width="1000">

# Problem:

The Center for Disease Control and Prevention (CDC) says that 35% of American drivers sleep less than the recommended minimum of seven hours a day. It mainly affects attention when performing any task and in the long term, it can affect health permanently [[1]](https://medlineplus.gov/healthysleep.html).

<img src="https://i.ibb.co/1Xz9MWX/image.png" width="1000">

According to a report by the WHO (World Health Organization) [[2]](http://www.euro.who.int/__data/assets/pdf_file/0008/114101/E84683.pdf), falling asleep while driving is one of the leading causes of traffic accidents. Up to 24% of accidents are caused by falling asleep, and according to the DMV USA (Department of Motor Vehicles) [[3]](https://dmv.ny.gov/press-release/press-release-03-09-2018) and NHTSA (National Highway traffic safety administration) [[4]](https://www.nhtsa.gov/risky-driving/drowsy-driving), 20% of accidents are related to drowsiness, being at the same level as accidents due to alcohol consumption with sometimes even worse consequences than those.

<img src="https://media2.giphy.com/media/PtrhzZJhbEBm8/giphy.gif" width="1000">

Also, the NHTSA mentions that being angry or in an altered state of mind can lead to more dangerous and aggressive driving [[5]](https://www.nhtsa.gov/risky-driving/speeding), endangering the life of the driver due to these psychological disorders.

<img src="https://i.ibb.co/YcWYJNw/tenor-1.gif" width="1000">

# Solution:
We have previously developed this idea with a couple of iterations one of them was a small project that ran on Edge impulse, called Edge driving Monitor:

https://www.hackster.io/422087/edge-driving-monitor-c504a8

Regretfully, as you can see, we had quite a lot of limitations in this project.
The CV models used were very limited because we were just using an ESP32 to run them on site so the information provided had a great deal of error. In addition to that it was slow and the visual stimulus was very limited, at least if you were diving.

At at the time the limitations of entry level hardware and the training models we had at hand were several, the project is almost 3 years old by now.

We also did a second try with a Jetson Nano:

https://devpost.com/software/torch-drowsiness-monitor

We still had a miriad of problems.

At first we wanted to run Pytorch and do the whole CV application on a Raspberry Pi 3, which is much more available and an easier platform to use. It probably was too much processing for the Raspi3 as it wasn't able to run everything we demanded so we upgraded to a Jetson Nano, we found several problems with the thermals at that point.

Later we had a little problem of focus with certain cameras so we had to experiment with several webcams that we had available to find one that didn't require to focus.

What we needed was:

- A better platform to run the CV models
- Better Models (it was 3 years ago :( )
- A better User interface

### I think we can have all three now in the year 2023!


We built a prototype which is capable of performing these 3 monitoring reliably and in addition to being easy to install in any vehicle.

<img src="https://i.ibb.co/WsggbsV/20231124-151109.jpg">

This PoC uses a Raspberry Pi 4 as the main computer to maintain low consumption for continuous use in a vehicle.

## Materials:

Hardware:
- RaspberryPi 4 (4Gb) - x1.
https://www.raspberrypi.com/products/raspberry-pi-4-model-b/
- Power Inverter for car - x1.
https://www.amazon.com/s?k=power+inverter+truper&ref=nb_sb_noss_2
- HD webcam - x1.
https://www.logitech.com/en-eu/products/webcams.html
- LCD Screen - x1.
https://www.alibaba.com/product-detail/Original-3-5-7-10-1_1600479875551.html
- GY-NEO6MV2 (GPS module) - x1.
https://www.alibaba.com/product-detail/Merrillchip-GY-NEO6MV2-New-NEO-6M_1600953573665.html
- Mini Speaker - x1.
https://www.alibaba.com/product-detail/High-Quality-Wireless-Blue-Tooth-Speaker_1600990407880.html

Optional Hardware:
- Jetson Nano - x1.
https://developer.nvidia.com/embedded/jetson-nano-developer-kit
- Jetson AGX Xavier - x1.
https://www.nvidia.com/es-la/autonomous-machines/embedded-systems/jetson-agx-xavier/
- Smartphone - x1.
https://www.amazon.com/s?k=smartphone

Software:
- OpenCV:
https://opencv.org/
- TensorFlow:
https://www.tensorflow.org/
- Raspberry Pi OS:
https://www.raspberrypi.com/software/
- YOLOv3:
https://pjreddie.com/darknet/yolo/
- NextJS 14:
https://nextjs.org/
- Open Layers Maps: 
https://openlayers.org/

Optional Software:
- Jetson SDK Manager
https://developer.nvidia.com/sdk-manager
- React Native:
https://reactnative.dev/
  
Online Platforms:
- Google Colab:
https://colab.research.google.com/
- AWS IoT:
https://aws.amazon.com/es/iot/
- Vercel:
https://vercel.com/
- Open Driving Navigator:
https://open-driving-navigator.vercel.app/

## Connection Diagram:

<img src="https://i.ibb.co/XZdX6ZB/software-drawio.png" width="1000">

This general connection diagram shows how through a camera we can obtain images of the driver or those of the streets to later obtain relevant data on the driver's alertness, his state of mind and the objects around the car. All fed back by our internal screen and our online web map.

- Eye State Detection: Through preprocessing in OpenCV haarcascades, OpenCV DNN and a frozen graph inference model (Tensor Flow), we obtain the driver's state of attention and drowsiness.. [Details](#drowsiness-model-training)
  
- Emotions Identification: Through preprocessing in OpenCV haarcascades, OpenCV DNN and a frozen graph inference model (Tensor Flow), we obtain the driver's mood.[Details](#emotions-model-training)

- YoloV3: Using OpenCV DNN and the famous network [YoloV3 from Darknet](https://pjreddie.com/darknet/) We carry out the identification of vehicles and pedestrians in the blind spot of the car. [Details](#yolov3-model-testing)

- Open Driving Monitor: Using a board enabled with OpenCV DNN, we created a system that can run the 3 AI models and also provide vehicle GPS information at all times. The selected board will be shown later. [Details](#board-setup)

- Open Driving Navigator: Using the NextJS, Open Layers and Vercel framework, we create a map that allows us to display the cars that are on our platform in real time and their states.. [Details](#open-driving-navigator-webpage)

- Open Driving Emulator: Using the React Native and AWS IoT framework, we created a car emulator so you can confirm that the data correctly reaches our online map.. [Details](#open-driving-emulator-android-native-app)

## Hardware Diagram:

Our hardware system shows the correct connection of the hardware used, the selection of a raspberry pi as our final hardware was made by doing [benchmarks](#comparison-benchmarks) with other boards specialized in AI.

<img src="https://i.ibb.co/znDn53G/hardware-drawio.png" width="1000">

- Raspberry Pi 4 (4Gb): This board allows the neural networks corresponding to each module to be run using the OpenCV DNN module with an ideal efficiency for the POC.

- USB Cam: This camera allows the input of images to the neural networks, it can be replaced by a native Raspberry camera or a wireless camera.

- GY-GPS6MV2: This module allows you to obtain the geolocation data of the device since the Raspberry Pi, despite being a computer, does not contain native GPS. This sensor provides data to the raspberry via serial.

- Speaker: This speaker allows us to obtain an auditory alarm signal for the correct function of the drowsiness detector, in the same way you can choose to connect it directly to the car speakers.

- LCD Screen: Allows the visualization of the navigation system in our [Open Driving Navigator](#open-driving-navigator-webpage), in addition to providing Blind point information on the side of the car. 

# Online Train and Test:

The correct training and testing of neural networks is essential to enhance the efficiency of driver assistance systems, as proposed in this project. However, carrying out these processes effectively requires appropriate datasets and frameworks. In this section, we will provide all the necessary resources so that you can reproduce the neural networks presented in this project as well as test their efficiency.

<img src="https://i.ibb.co/q5Thrq9/image.png" width="1000">

**NOTE:** The only neural network that was not trained was the Darknet YoloV3 network because it is a network already trained and ready to use, so we will only show its implementation with OpenCV DNN.

## Online Training:

For the efficient training of this type of neural networks, the use of GPUs is usually required due to their exceptional efficiency compared to CPUs. However, this type of infrastructure can be somewhat expensive and difficult to maintain. However, thanks to Google and TensorFlow, it is possible to do this training for free thanks to Google Colab.

<img src="https://i.ibb.co/sFxVFTX/image.png" width="1000">

By having a Google account, it will give us free access to Jupyter Notebooks Online with GPU or TPU (with certain limitations). Which are enough for us to train, deploy the neural networks and share the notebooks that we will show below.

**NOTE:** Please note the layers supported by the OpenCV DNN module, some models with complex or very modern layers may not be supported yet.

https://docs.opencv.org/4.8.0/d6/d87/group__dnnLayerList.html

### Emotions Model Training:

Here is the link for the training notebook: [CLICK HERE](./Emotions/train/Train_Test_and_Deploy_Emotions.ipynb)

<img src="https://i.ibb.co/t4xrKxS/vlcsnap-2023-11-27-22h52m49s056.png" width="1000">

The neural network to detect emotions is a convolutional neural network designed specifically to recognize and classify emotions through images. To perform this task correctly we design the following neural network in tensorflow.

<img src="https://i.ibb.co/wBNZrw2/output-1.png" width="1000">

- Conv2d: This layer applies kernels to the image and obtains its main characteristics.
  
- Activation: This layer always comes after a convolutional layer to detect activations after the kernel.
  
- BatchNormalization: This layer normalizes the activations of a previous layer and accelerates the training of the neural network.
  
- MaxPooling2D: this layer reduces the number of parameters in the network and is added in order to prevent overfitting in training.
  
- Dropout: Randomly turns off a percentage of neurons during each training step, which improves model generalization.
  
- Flatten: this layer converts the output of the 3D layers into a one-dimensional vector that is finally passed to layers of fully connected neural networks, that is, it converts it into a format that this layer understands and can classify.
  
- Dense: in this layer each neuron is connected to all the neurons in the previous layer and has the purpose of performing the final classification.

The dataset we used in this training was [FER-2013](https://huggingface.co/spaces/mxz/emtion/resolve/c697775e0adc35a9cec32bd4d3484b5f5a263748/fer2013.csv) which is a dataset with more than 28k images of emotions already classified.

<img src="https://i.ibb.co/dtLZfhy/image.png" width="1000">

Already in the notebook we have detailed the entire process of importing the dataset, separating it into Test, Train and Validation subsets, you only have to open the notebook in colab and hit run there to create the model yourself.

<img src="https://i.ibb.co/D55MdD3/New-Project.png" width="1000">

The notebook for this neural network is: [CLICK HERE](./Emotions/train/Train_Test_and_Deploy_Emotions.ipynb)

**NOTE:** Also in the sale file folder [Train](./Emotions/train/) We added the requirements.txt file so you know exactly in which ENV and version of all the python modules the training was carried out.

Finally, after training we will be able to obtain a Frozen Graph which is an inference model that is already optimized for production, this will be the file that we will provide to the OpenCV DNN module.

<img src="https://i.ibb.co/NyLBGqF/image.png" width="1000">

To directly download the Frozen Graph: [CLICK HERE](./Emotions/model/emotions-v1.pb)

### Drowsiness Model Training:

Link to the training notebook: [CLICK HERE](./Drowsiness/train/Train_Test_and_Deploy_Blink.ipynb)

<img src="https://i.ibb.co/MVQfBq2/vlcsnap-2023-11-27-23h13m20s380.png" width="1000">

The neural network for detecting eye state is a convolutional neural network specifically designed to recognize a closed eye from an open eye. To perform this task correctly we design the following neural network in tensorflow.

<img src="https://i.ibb.co/9N93ttX/New-Project-3.png" width="1000">

- Conv2d: This layer applies kernels to the image and obtains its main characteristics.
  
- Activation: This layer always comes after a convolutional layer to detect activations after the kernel.
  
- BatchNormalization: This layer normalizes the activations of a previous layer and accelerates the training of the neural network.
  
- MaxPooling2D: this layer reduces the number of parameters in the network and is added in order to prevent overfitting in training.
  
- Dropout: Randomly turns off a percentage of neurons during each training step, which improves model generalization.
  
- Flatten: this layer converts the output of the 3D layers into a one-dimensional vector that is finally passed to layers of fully connected neural networks, that is, it converts it into a format that this layer understands and can classify.
  
- Dense: in this layer each neuron is connected to all the neurons in the previous layer and has the purpose of performing the final classification.

The dataset we used in this training was[B-eye](https://github.com/altaga/DBSE-monitor/raw/master/Drowsiness/train/dataset/dataset_B_Eye_Images.zip) which is a dataset with more than 4,800 images of open and closed eyes already classified.

<img src="https://i.ibb.co/R6Vg6HS/image.png" width="1000">

Already in the notebook we have detailed the entire process of importing the dataset, separating it into Test, Train and Validation subsets, you only have to open the notebook in colab and hit run there to create the model yourself.

<img src="https://i.ibb.co/VpmMsWr/New-Project-1.png" width="1000">

The neural network notebook is: [CLICK HERE](./Drowsiness/train/Train_Test_and_Deploy_Blink.ipynb)

**NOTE:** Also in the same file folder [Train](./Drowsiness/train/)We added the requirements.txt file so you know exactly in which ENV and version of all the python modules the training was carried out.

Finally, after training we will be able to obtain a Frozen Graph which is an inference model that is already optimized for production, this will be the file that we will provide to the OpenCV DNN module.

<img src="https://i.ibb.co/NyLBGqF/image.png" width="1000">

To download directly the Frozen Graph: [CLICK HERE](./Drowsiness/model/blink-v1.pb)

## Online Models Testing:

Once we have the models ready, it is necessary to move on to the Testing stage, which involves the exhaustive evaluation of the models with inputs completely outside the training dataset, the objective of this is to verify the accuracy and performance of the model.

<img src="https://i.ibb.co/xzcBvMZ/image.png" width="1000">

### Emotions Model Testing:

Here is a link for the testing training notebook: [CLICK HERE](./Emotions/test/Test_Emotions_DNN.ipynb)

We invite you to open the Notebook and perform the test yourself, the dataset we created was 28 images, 7 of each emotion, in order to verify the accuracy with this new data.

<img src="https://i.ibb.co/dgGjX2z/image.png" width="1000">

Finally, the model precision percentages show us the following.

<img src="https://i.ibb.co/sg256sk/image.png" width="1000">

We can notice that the emotion that has the most problems recognizing is disgust and fear. Which tells us that this model still has room for improvement.

### Drowsiness Model Testing:

Here is a link directly to the testing notebook: [CLICK HERE](./Drowsiness/test/Test_Blink_DNN.ipynb)

We invite you to open the Notebook and perform the test yourself, but when testing with a test dataset that we created, we reached the following results.

<img src="https://i.ibb.co/k8mJvrq/image.png" width="1000">

Finally, the model precision percentages show us the following.

<img src="https://i.ibb.co/MRSkRXB/image.png" width="1000">

We can see that the model is perfect, however we noticed during the in-field tests that when we closed our eyes a little or distracted them from the camera, the model resulted in closed eyes, which for practical purposes to detect drowsiness or distraction this is very useful to us. .

**NOTE:** in the demo video we demonstrate this function clearly, you can go watch it to see this project in operation!

### YoloV3 Model Testing:

Here is a link directly to the testing notebook: [CLICK HERE](./Yolo/test/LoadAndTestYolo.ipynb)

We invite you to open the Notebook and take the test yourself. Nevetheless here we share the test results with you.

<img src="https://i.ibb.co/2SsxxBm/image.png" width="1000">

The model used in the test is the Yolo-Tiny model, because it is the lightest that we can use in this project and its detections are not sufficient for the proper functioning of the project.

**NOTE:** In the next section you will see the comparison of the models in various HW, if you want to use the complete YoloV3 model we recommend using at least the Jetson Nano, since it does perform processing on GPU and allows a frame rate realistic to function.

# Board Setup:

The correct choice of hardware for these AI models is essential for correct operation, adjusting to the energy consumption of the vehicle and the budget to carry out this project in production.

<img src="https://i.ibb.co/bXrn5h9/New-Project-4.png" width="1000">

In all boards the AWS IoT configuration is the same, since it is done through certificates in the following code section.

[CODE](./RPi%20Deploy/iot-mqtt.py)

    EndPoint = "XXXXXXXXXX-ats.iot.us-east-1.amazonaws.com"
    caPath = "opencvDNN/certs/aws-iot-rootCA.crt"
    certPath = "opencvDNN/certs/aws-iot-device.pem"
    keyPath = "opencvDNN/certs/aws-iot-private.key"

The data frame that must be sent to the platform so that data begins to appear is the following.

    {
      "coordinates": [
        -99.4738495,
        19.3749642
      ],
      "color": "#808080",
      "data": "Emotion: Neutral\nState: Awake",
      "id": 98574584180
    }

And the GPS configuration also has no difference from one board to another since they all have the same 40-pin configuration.

[CODE](./RPi%20Deploy/Gps/gps.py)

The GPS module does not require any additional configuration, it is configured only when it is connected, when it is already working the LED on the board will flash every second.

<img src="https://i.ibb.co/gzfhDMr/image.png" width="600">

## Raspberry Pi 4:

The Raspberry Pi 4 (RPi4) is a board the size of a credit card that provides us with the minimum characteristics so that this project can be carried out.

<img src="https://i.ibb.co/MMST3xf/image.png" width="600">

- CPU: Broadcom BCM2711 Quad-core
- GPU: VideoCore VI graphics
- RAM: 4GB
- Storage: 32 GB (MicroSD)
- Audio: 3.5mm jack
- Screen Port: 2 × micro HDMI
- Network Interface: Wi-Fi 802.11b/g/n/ac 
- Board Price: [$55 - (11/28/2023, Seeedstudio)](https://www.seeedstudio.com/Raspberry-Pi-4-Computer-Model-B-4GB-p-4077.html)

In the case of the RPi4 we are fortunate that there are already compiled versions of OpenCV with the DNN module. The steps to carry out this installation are as follows:

- Update apt repository.
    
      sudo apt update

- Update pip, setuptools and wheel.

      pip install --upgrade pip setuptools wheel

- Install OpenCV requirements

      sudo apt-get install -y libhdf5-dev libhdf5-serial-dev python3-pyqt5 libatlas-base-dev libjasper-dev

- Select the correct version of OpenCV module and install it
 
  - **opencv-python**: The main distribution of the OpenCV library for Python, providing computer vision functionality.
  - **opencv-python-headless**: A lightweight version of the OpenCV library for Python without GUI dependencies, suitable for headless environments or server deployments.
  - **opencv-contrib-python**: An extended distribution of OpenCV for Python, including additional modules and features beyond the core library. **(USE THIS)**
  - **opencv-contrib-python-headless**: A headless version of the extended OpenCV library for Python, omitting GUI components, making it suitable for server environments or systems without graphical interfaces. 

- Install the latest version (v4.8.0 - 11/28/2023) of opencv-contrib-python.

      pip install opencv-contrib-python

Once this is done you will be able to use all the OpenCV modules on the RPi4 including the OpenCV DNN.

## Jetson Nano:

The Jetson Nano is an AI development board created by NVIDIA. This board provides us with a good cost benefit to carry out this project, in addition to allowing image processing by GPU.

<img src="https://i.ibb.co/f8ZntpX/image.png" width="600">

- CPU: Quad-core ARM Cortex-A57 MPCore
- GPU: 128-core Maxwell
- RAM: 4GB
- Storage: 32 GB (MicroSD)
- Audio: 3.5mm jack
- Screen Port: HDMI 2.0
- Network Interface: Gigabit Ethernet
- Board Price: [$149 - (11/28/2023, Seeedstudio)](https://www.seeedstudio.com/NVIDIA-Jetson-Nano-Development-Kit-B01-p-4437.html)

In the case of the Jetson Nano it already comes with a version of OpenCV, but it is not adapted to work with CUDA and cuDNN, which are the modules in the jetson that allow us to perform image processing on the GPU. During the setup process we will use a [script](./OpenCV%20Scripts/build_opencv_nano.sh) already designed to automatically configure OpenCV on our board.
Configuracion:

- Update apt repository.
    
      sudo apt update

- Download the installation script, feel free to open the script file and review it.

      wget https://raw.githubusercontent.com/altaga/Open-Driving-Monitor/main/OpenCV%20Scripts/build_opencv_nano.sh

- For the script to be successful, check that the OpenCV build parameters have the versions corresponding to our version of jetpack.

  <img src="https://i.ibb.co/93N1N5Y/New-Project-5.png" width="1000">

- In the script file change the CUDA_ARCH_BIN and CUDNN_VERSION numbers if they do not match on your jetson.
      ...
      -D CUDA_ARCH_BIN=5.3 # Cuda Arch BIN 
      -D CUDA_ARCH_PTX=
      -D CUDA_FAST_MATH=ON
      -D CUDNN_VERSION='8.2' # cuDNN
      ...
  
- In the Jetson Nano the build process can take 4 to 5 hours, we still recommend that it have a small fan that diffuses the heat generated by the board, otherwise you run the risk of the board turning off mid-process and having to do it from the beggining.

Once this is done you will be able to use all the OpenCV modules on the Jetson Nano including the OpenCV DNN and GPU processing.

## Jetson AGX Xavier:

The Jetson AGX Xavier (Jetson AGX) is an AI development board created by NVIDIA. This board provides us with the best performance in AI models due to advanced GPU processing, but with high cost hardware.

<img src="https://i.ibb.co/J74dZP5/61m739zmag-L-AC-SX679.jpg" width="600">

- CPU: 8-core ARMv8.2
- GPU: NVIDIA Volta architecture with 512
- RAM: 32 GB
- Audio: Integrated audio codec HDMI
- Screen Port: HDMI 2.0, eDP 1.4, DP 1.2
- Network Interface: Gigabit Ethernet
- Board Price: [$699 - (11/28/2023, Seeedstudio)](https://www.seeedstudio.com/NVIDIA-Jetson-AGX-Xavier-Development-Kit-p-4418.html)

In the case of the Jetson AGX it already comes with a version of OpenCV, but it is not adapted to work with CUDA and cuDNN, which are the modules in the jetson that allow us to perform image processing on the GPU. During the setup process we will use a [script](./OpenCV%20Scripts/build_opencv_agx.sh) already designed to automatically configure OpenCV on our board.

Configuration:

- Update apt repository.
    
      sudo apt update

- Download the installation script, feel free to open the script file and review it.

      wget https://raw.githubusercontent.com/altaga/Open-Driving-Monitor/main/OpenCV%20Scripts/build_opencv_agx.sh

- For the script to be successful, check that the OpenCV build parameters have the versions corresponding to our version of jetpack.

  <img src="https://i.ibb.co/xSg1mPS/New-Project-6.png" width="1000">

- In the script file change the CUDA_ARCH_BIN and CUDNN_VERSION numbers if they do not match on your jetson.

      ...
      -D CUDA_ARCH_BIN=7.2 # Cuda Arch BIN
      -D CUDA_ARCH_PTX=
      -D CUDA_FAST_MATH=ON
      -D CUDNN_VERSION='8.6' # cuDNN
      ...
  
- On the Jetson AGX the build process can take 1 to 2 hours, the board already has its own built-in fan, so you don't have to worry about the temperature at all.

Once this is done you will be able to use all the OpenCV modules on the Jetson AGX including the OpenCV DNN and GPU processing.

# Comparison Benchmarks:

We compare the performance of all the neural networks on each of the boards in order to obtain FPS data for each of the models.

### RPi4:

- Drowsiness:

  Video: Click on the image

  [<img src="https://i.ibb.co/ZBxgtS4/logo-1.png" width="300">](https://youtu.be/ALhJKYtyX7Q)  


- Emotions:

  Video: Click on the image

  [<img src="https://i.ibb.co/ZBxgtS4/logo-1.png" width="300">](https://youtu.be/37lpyZrZbPw)

- Yolo:

  Video: Click on the image

  [<img src="https://i.ibb.co/ZBxgtS4/logo-1.png" width="300">](https://youtu.be/b_tWIjI1-8c)


### Jetson Nano:

- Drowsiness:

  Video: Click on the image

  [<img src="https://i.ibb.co/ZBxgtS4/logo-1.png" width="300">](https://youtu.be/c6ioMZixa9U)


- Emotions:

  Video: Click on the image

  [<img src="https://i.ibb.co/ZBxgtS4/logo-1.png" width="300">](https://youtu.be/hM1KTele-LE)


- Yolo:

  Video: Click on the image

  [<img src="https://i.ibb.co/ZBxgtS4/logo-1.png" width="300">](https://youtu.be/-PLD04Vq0mI)


### Jetson AGX:

- Drowsiness:

  Video: Click on the image
  
  [<img src="https://i.ibb.co/ZBxgtS4/logo-1.png" width="300">](https://youtu.be/1xDIpZbogDo)


- Emotions:

  Video: Click on the image

  [<img src="https://i.ibb.co/ZBxgtS4/logo-1.png" width="300">](https://youtu.be/sdUzYwAY8LI)


- Yolo:

  Video: Click on the image

  [<img src="https://i.ibb.co/ZBxgtS4/logo-1.png" width="300">](https://youtu.be/bbcro-RTcR0)


## Benchmarks:

Once the tests are finished and taking the averages of the FPS of each test, we obtain the following results.

- The processing or inference time for each of the boards was as follows, this is just the time it takes for the image to go through the network and obtain a result.
  
  <img src="https://i.ibb.co/m4VbZBx/image.png" width="600">

- The real time it takes our program to carry out the pre-processing, inference and display of the image on each of the boards is.

  <img src="https://i.ibb.co/qF08658/image.png" width="600">

Finally we decided to use the RPi4 as the final board because the FPS it provides us is not enough for this project, however you can use the board that you consider best.

# Open Driving Navigator (WebPage):

Already having the vehicle data, we create a web platform that allows us to monitor in real time the cars that are sending data to the system. This is in order to prevent accidents and provide more information to other vehicles.

<img src="https://i.ibb.co/pnGjwNy/image.png" width="600">

URL: https://open-driving-navigator.vercel.app/

NOTE: the page requires location permissions so that when sending data to the platform we can see the cars appear in our location.

## NextJS:

For the web platform, the framework of [NextJS](https://nextjs.org/) was used in in its most recent version (11/28/2023) and the open source maps [Open Layers](https://openlayers.org/).

All the code of the website is open source and is in the following link.

[CODE](./Map%20WebPage/)

## AWS Iot:

Communication between the devices and the website is carried out through AWS IoT since it allows us to maintain a secure connection at all times and we use the MQTTS protocol.

<img src="https://i.ibb.co/Jq9W7GW/mqtts-drawio.png" width="600">

Configuring AWS IoT on the web page is done in the following code section.

[CODE](./Map%20WebPage/src/utils/aws-configuration.js)

    var awsConfiguration = {
      poolId: "us-east-1:xxxxxxxxxxxxxxxxxxxxxxxxxxxxx", // 'YourCognitoIdentityPoolId'
      host:"xxxxxxxxxxxxxxxxxxxxx.iot.us-east-1.amazonaws.com", // 'YourAwsIoTEndpoint', e.g. 'prefix.iot.us-east-1.amazonaws.com'
      region: "us-east-1" // 'YourAwsRegion', e.g. 'us-east-1'
    };
    module.exports = awsConfiguration;

## Vercel:

The deployment of the website to the Internet can be done easily and for free thanks to the Vercel platform, [Check Free Plan Limits](https://vercel.com/docs/limits/overview)

<img src="https://i.ibb.co/vwWjdtG/image.png" width="600">

It is only necessary to connect the main repository of the website with Vercel and it will automatically perform the deployment.

URL: https://open-driving-navigator.vercel.app/

# Open Driving Emulator (Android Native App):

We can simulate a car by sending data from the simulator to the platform.

<img src="https://i.ibb.co/pnGjwNy/image.png" width="600">

NOTE: the app requires location permissions so that when sending data to the platform we can see the simulated car appear in our location.

## React Native Setup:

To create the emulator, the framework of [React Native](https://reactnative.dev/) in its most recent version (11/28/2023).

All the code of the app is open source and is in the following link.

[CODE](./Emulator%20ReactNative/)

## AWS Iot:

The communication between the app and the website is carried out through AWS IoT since it allows us to maintain a secure connection at all times and we use the MQTTS protocol.

<img src="https://i.ibb.co/Jq9W7GW/mqtts-drawio.png" width="600">

The configuration of AWS IoT in the app is exactly the same as that of the website, since both work with javascript and this is done in the following section of code.

[CODE](./Emulator%20ReactNative/src/utils/aws-configuration.js)

    var awsConfiguration = {
      poolId: "us-east-1:xxxxxxxxxxxxxxxxxxxxxxxxxxxxx", // 'YourCognitoIdentityPoolId'
      host:"xxxxxxxxxxxxxxxxxxxxx.iot.us-east-1.amazonaws.com", // 'YourAwsIoTEndpoint', e.g. 'prefix.iot.us-east-1.amazonaws.com'
      region: "us-east-1" // 'YourAwsRegion', e.g. 'us-east-1'
    };
    module.exports = awsConfiguration;

## Google Play:

To make it easier for them to test the web platform. This is the link to the beta of our application, with it you can send information to our online map and simulate our system without additional hardware. AI models can be tested from the application in future versions using [OpenCV.JS](https://docs.opencv.org/3.4/d5/d10/tutorial_js_root.html)

<img src="https://i.ibb.co/sQfN4y7/image.png" width="1000">

URL: https://play.google.com/store/apps/details?id=com.altaga.ODS

## How use it:

To use the app, you will only have to open the application, accept the location permissions and finally press "Start Emulation"

<img src="https://i.ibb.co/nwrjCmW/vlcsnap-2023-11-28-16h57m14s456.png" width="32%"> <img src="https://i.ibb.co/vsLQChD/vlcsnap-2023-11-28-16h57m10s157.png" width="32%"> <img src="https://i.ibb.co/nw24BLp/vlcsnap-2023-11-28-16h57m33s176.png" width="32%">

# The Final Product:

### Complete System:

<img src="https://i.ibb.co/WsggbsV/20231124-151109.jpg">

### In-car system:

<img src="https://i.ibb.co/Z6nF54M/20231127-132608.png" width="49%"> <img src="https://i.ibb.co/PQm4mGk/20231127-125635.png" width="49%">

# EPIC DEMO:

Video: Click on the image
[![Car](https://i.ibb.co/ZBxgtS4/logo-1.png)](pending...)


# Commentary and final words:

With this new hardware and the new models that we have in the year 2023 our vision for this project has finally been achieved. It now works without any delay and the models are instantaneous. Probably the next steps we will take for the project is just on the industrial side to try and make it a consumer product that can be in concept installed on every non-smart car. Thank you for reading and hopefully you like the project.



# References:

Links:

(1) https://medlineplus.gov/healthysleep.html

(2) http://www.euro.who.int/__data/assets/pdf_file/0008/114101/E84683.pdf

(3) https://dmv.ny.gov/press-release/press-release-03-09-2018

(4) https://www.nhtsa.gov/risky-driving/drowsy-driving

(5) https://www.nhtsa.gov/risky-driving/speeding

# Table of contents

- [Open Driving Monitor](#open-driving-monitor)
- [Fast Links:](#fast-links)
    - [Video Demo:](#video-demo)
    - [Test Notebooks:](#test-notebooks)
    - [WebPage:](#webpage)
    - [Native App:](#native-app)
- [Introduction:](#introduction)
- [Problem:](#problem)
- [Solution:](#solution)
  - [Materials:](#materials)
  - [Connection Diagram:](#connection-diagram)
  - [Hardware Diagram:](#hardware-diagram)
- [Online Train and Test:](#online-train-and-test)
  - [Online Training:](#online-training)
    - [Emotions Model Training:](#emotions-model-training)
    - [Drowsiness Model Training:](#drowsiness-model-training)
  - [Online Models Testing:](#online-models-testing)
    - [Emotions Model Testing:](#emotions-model-testing)
    - [Drowsiness Model Testing:](#drowsiness-model-testing)
    - [YoloV3 Model Testing:](#yolov3-model-testing)
- [Board Setup:](#board-setup)
  - [Raspberry Pi 4:](#raspberry-pi-4)
  - [Jetson Nano:](#jetson-nano)
  - [Jetson AGX Xavier:](#jetson-agx-xavier)
- [Comparison Benchmarks:](#comparison-benchmarks)
    - [RPi4:](#rpi4)
    - [Jetson Nano:](#jetson-nano-1)
    - [Jetson AGX:](#jetson-agx)
  - [Benchmarks:](#benchmarks)
- [Open Driving Navigator (WebPage):](#open-driving-navigator-webpage)
  - [NextJS:](#nextjs)
  - [AWS Iot:](#aws-iot)
  - [Vercel:](#vercel)
- [Open Driving Emulator (Android Native App):](#open-driving-emulator-android-native-app)
  - [React Native Setup:](#react-native-setup)
  - [AWS Iot:](#aws-iot-1)
  - [Google Play:](#google-play)
  - [How use it:](#how-use-it)
- [The Final Product:](#the-final-product)
    - [Complete System:](#complete-system)
    - [In-car system:](#in-car-system)
- [EPIC DEMO:](#epic-demo)
- [Commentary:](#commentary)
- [References:](#references)
- [Table of contents](#table-of-contents)
