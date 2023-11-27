# Open Driving Monitor

Drowsiness, Blind Spot, and Emotion Monitor system for driving and heavy machinery, enhancing safety through fatigue detection, blind spot awareness, and emotional state analysis powered by OpenCV.

<img src="https://i.ibb.co/ZBxgtS4/logo-1.png" width="1000">

# Introduction:



# Solution:



# Materials:

Hardware:
- RaspberryPi 4 - x1.
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
  
Online Platforms:
- Google Colab:
https://colab.research.google.com/
- AWS IoT:
https://aws.amazon.com/es/iot/
- Vercel:
https://vercel.com/
- Open Driving Map:
https://open-driving-navigator.vercel.app/

# Connection Diagram:

<img src="https://i.ibb.co/XZdX6ZB/software-drawio.png" width="1000">

Este esquema general de conexiones muestra como a traves de una camara podemos obtener las imaganes del conductor o las de las calles para posteriormente obtener datos relevantes del estado de alerta del conductor, su estado de animo y los objetos alrededor del auto. Todo retroalimentado por nuestra pantalla interna y nuestro web map online.

- Eye State Detection: Mediante un preprocesamiento en OpenCV haarcascades, OpenCV DNN y un modelo de inferencia frozen graph (Tensor Flow), obtenemos el estado de atencion y somnolencia del conductor. [Details](#drowsiness-model-training)
  
- Emotion Identification: Mediante un preprocesamiento en OpenCV haarcascades, OpenCV DNN y un modelo de inferencia frozen graph (Tensor Flow), obtenemos el estado de animo del conductor. [Details](#emotion-model-training)

- YoloV3: Mediante OpenCV DNN y la famosa red [YoloV3 from Darknet](https://pjreddie.com/darknet/) realizamos la identificacion de vehiculos y peatones en el punto ciego del auto. [Details](#yolov3-model-testing)

- Open Driving Monitor: Mediante una board habilitada con OpenCV DNN, realizamos un sistema que puede correr los 3 modelos de AI y ademas proveer informacion del GPS del vehiculo en todo momento. La board seleccionada se mostrara mas adelante. [Details](#board-setup)

- Open Driving Navigator: Mediante el framework NextJS, Open Layers y Vercel, realizamos un mapa que nos permite desplegar los automoviles que esten en nuestra plataforma en tiempo real y sus estados. [Details](#open-driving-map-webpage)

- Open Driving Emulator: Mediante el framework de React Native y AWS IoT, realizamos un emulador de automovil para que puedas confirmar que los datos llegan correctamente a nuestro mapa online. [Details](#open-driving-emulator-android-native-app)

# Hardware Diagram:

<img src="" width="1000">



# Online Train and Test:



## Online Training:



### Emotion Model Training:



### Drowsiness Model Training:



## Online Models Testing:



### Emotion Model Testing:



### Drowsiness Model Testing:



### YoloV3 Model Testing:



# Board Setup:



## RPi:



## Jetson Nano:



## Jetson AGX Xavier:



# Comparison Benchmarks:



# Open Driving Map (WebPage):



## NextJS:



## AWS Iot:



## Vercel:



# Open Driving Emulator (Android Native App):



## React Native Setup:



## AWS Iot:



## Google Play:

Este es el enlace de la beta de nuestra aplicacion, con ella podras mandar informacion a nuestro mapa online y simular nuestro sistema sin hardware adicional. Los modelos de AI podran ser probados desde la aplicacion en versiones futuras mediante [OpenCV.JS](https://docs.opencv.org/3.4/d5/d10/tutorial_js_root.html)

<img src="https://i.ibb.co/sQfN4y7/image.png" width="1000">

https://play.google.com/store/apps/details?id=com.altaga.ODS

# The Final Product:



# EPIC DEMO:

Video: Click on the image
[![Car](https://i.ibb.co/ZBxgtS4/logo-1.png)](https://youtu.be/rNhcBHKiGik)

Sorry github does not allow embed videos.

# Commentary:

I would consider the product finished as we only need a little of additional touches in the industrial engineering side of things for it to be a commercial product. Well and also a bit on the Electrical engineering perhaps to use only the components we need. That being said this functions as an upgrade from a project that a couple friends and myself are developing and It was ideal for me to use as a springboard and develop the idea much more. This one has the potential of becoming a commercially available option regarding Smart cities as the transition to autonomous or even smart vehicles will take a while in most cities.

That middle ground between the Analog, primarily mechanical-based private transports to a more "Smart" vehicle is a huge opportunity as the transition will take several years and most people are not able to afford it. Thank you for reading.

# References:

Links:

(1) https://medlineplus.gov/healthysleep.html

(2) http://www.euro.who.int/__data/assets/pdf_file/0008/114101/E84683.pdf

(3) https://dmv.ny.gov/press-release/press-release-03-09-2018

(4) https://www.nhtsa.gov/risky-driving/drowsy-driving

(5) https://www.nhtsa.gov/risky-driving/speeding

# Table of contents

- [Open Driving Monitor](#open-driving-monitor)
- [Introduction:](#introduction)
- [Solution:](#solution)
- [Materials:](#materials)
- [Connection Diagram:](#connection-diagram)
- [Hardware Diagram:](#hardware-diagram)
- [Online Train and Test:](#online-train-and-test)
  - [Online Training:](#online-training)
    - [Emotion Model Training:](#emotion-model-training)
    - [Drowsiness Model Training:](#drowsiness-model-training)
  - [Online Models Testing:](#online-models-testing)
    - [Emotion Model Testing:](#emotion-model-testing)
    - [Drowsiness Model Testing:](#drowsiness-model-testing)
    - [YoloV3 Model Testing:](#yolov3-model-testing)
- [Board Setup:](#board-setup)
  - [RPi:](#rpi)
  - [Jetson Nano:](#jetson-nano)
  - [Jetson AGX Xavier:](#jetson-agx-xavier)
- [Comparison Benchmarks:](#comparison-benchmarks)
- [Open Driving Map (WebPage):](#open-driving-map-webpage)
  - [NextJS:](#nextjs)
  - [AWS Iot:](#aws-iot)
  - [Vercel:](#vercel)
- [Open Driving Emulator (Android Native App):](#open-driving-emulator-android-native-app)
  - [React Native Setup:](#react-native-setup)
  - [AWS Iot:](#aws-iot-1)
  - [Google Play:](#google-play)
- [The Final Product:](#the-final-product)
- [EPIC DEMO:](#epic-demo)
- [Commentary:](#commentary)
- [References:](#references)
- [Table of contents](#table-of-contents)
