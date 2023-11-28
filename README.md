# Open Driving Monitor

Drowsiness, Blind Spot, and Emotions Monitor system for driving and heavy machinery, enhancing safety through fatigue detection, blind spot awareness, and emotional state analysis powered by OpenCV.

<img src="https://i.ibb.co/ZBxgtS4/logo-1.png" width="1000">

# Fast Links:

### Video Demo:

Video Demo: Click on the image
[![Car](https://i.ibb.co/ZBxgtS4/logo-1.png)](pending...)

Sorry github does not allow embed videos.

### Test Notebooks:

- Emotions: [CLICK HERE](./Emotions/test/Test_Emotions_DNN.ipynb)

- Drowsiness: [CLICK HERE](./Drowsiness/test/Test_Blink_DNN.ipynb)

- YoloV3: [CLICK HERE](./Yolo/test/LoadAndTestYolo.ipynb)

### WebPage:

- Open Driving Navigator: [CLICK HERE](https://open-driving-navigator.vercel.app/)

### Native App:

- Open Driving Emulator: [CLICK HERE](https://play.google.com/store/apps/details?id=com.altaga.ODS)

# Introduction:



# Solution:



# Materials:

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

# Connection Diagram:

<img src="https://i.ibb.co/XZdX6ZB/software-drawio.png" width="1000">

Este esquema general de conexiones muestra como a traves de una camara podemos obtener las imaganes del conductor o las de las calles para posteriormente obtener datos relevantes del estado de alerta del conductor, su estado de animo y los objetos alrededor del auto. Todo retroalimentado por nuestra pantalla interna y nuestro web map online.

- Eye State Detection: Mediante un preprocesamiento en OpenCV haarcascades, OpenCV DNN y un modelo de inferencia frozen graph (Tensor Flow), obtenemos el estado de atencion y somnolencia del conductor. [Details](#drowsiness-model-training)
  
- Emotions Identification: Mediante un preprocesamiento en OpenCV haarcascades, OpenCV DNN y un modelo de inferencia frozen graph (Tensor Flow), obtenemos el estado de animo del conductor. [Details](#emotions-model-training)

- YoloV3: Mediante OpenCV DNN y la famosa red [YoloV3 from Darknet](https://pjreddie.com/darknet/) realizamos la identificacion de vehiculos y peatones en el punto ciego del auto. [Details](#yolov3-model-testing)

- Open Driving Monitor: Mediante una board habilitada con OpenCV DNN, realizamos un sistema que puede correr los 3 modelos de AI y ademas proveer informacion del GPS del vehiculo en todo momento. La board seleccionada se mostrara mas adelante. [Details](#board-setup)

- Open Driving Navigator: Mediante el framework NextJS, Open Layers y Vercel, realizamos un mapa que nos permite desplegar los automoviles que esten en nuestra plataforma en tiempo real y sus estados. [Details](#open-driving-navigator-webpage)

- Open Driving Emulator: Mediante el framework de React Native y AWS IoT, realizamos un emulador de automovil para que puedas confirmar que los datos llegan correctamente a nuestro mapa online. [Details](#open-driving-emulator-android-native-app)

# Hardware Diagram:

Nuestro sistema de hardware muestra la correcta conexion del hardware utilizado, la seleccion de una raspberry pi como nuestro hardware final se realizo haciendo [benchmarks](#comparison-benchmarks) con otras boards especializadas en AI.

<img src="https://i.ibb.co/znDn53G/hardware-drawio.png" width="1000">

- Raspberry Pi 4 (4Gb): esta board permite correr mediante el modulo OpenCV DNN las redes nuronales correspondientes a cada modulo con una eficiencia ideal para el POC.

- USB Cam: esta camara permite el input de imagenes a las redes neuronales, esta puede ser sustituida por una camara nativa de raspberry o una camara inalambrica.

- GY-GPS6MV2: este modulo permite obtener los datos de geolocalizacion del dispositivo ya que la raspberry pi apesar de ser un computador no contiene GPS nativo. Este sensor provee mediante serial lo datos a la raspberry.

- Speaker: este speaker nos permite obtener una señal de alarma auditiva para la correcta funcion del drowsiness detector, de igual manera de puede optar por conectarlo directamente a los speakers del automovil.

- LCD Screen: Permite la visualizacion del sistema de navegacion en nuestro [Open Driving Navigator](#open-driving-navigator-webpage), ademas de proveer informacion del Blind point al lateral del auto.

# Online Train and Test:

El correcto entrenamiento y pruebas de las redes neuronales son fundamentales para potenciar la eficiencia de los sistemas de asistencia al conductor, tal como se proponen en este proyecto. No obstante, llevar a cabo estos procesos de manera efectiva requiere de datasets y frameworks apropiados. En esta sección, proporcionaremos todos los recursos necesarios para que puedas reproducir las redes neuronales presentadas en este proyecto asi como probar su eficiencia.

<img src="https://i.ibb.co/q5Thrq9/image.png" width="1000">

**NOTA:** la unica red neuronal que no se entreno fue la red YoloV3 de Darknet debido a que es una red ya entrenada y lista para usar, asi que solo mostraremos su implementacion con OpenCV DNN.

## Online Training:

Para el eficiente entrenamiento de este tipo de redes nuronales suele requerirse el uso de GPUs debido a su eficiencia excepcional en comparación con las CPU. Sin embargo este tipo de infraestructura puede ser algo cara y dificil de mantener. Si embargo gracias a Google y TensorFlow posible realizar este entrenamiento de forma gratuita gracias a Google Colab.

<img src="https://i.ibb.co/sFxVFTX/image.png" width="1000">

Al tener una cuenta de google, este nos dara acceso gratuito a Jupyter Notebooks Online con GPU o TPU (con ciertas limitaciones). Los cuales nos son suficientes para entrenar, desplegar las redes neuronales y compartirles los notebooks que mostraremos a continuacion.

**NOTA:** Tenga en cuenta las layers compatibles con OpenCV DNN module, algunos modelos con layers complejas o muy modernas tal vez no son compatibles aun.

https://docs.opencv.org/4.8.0/d6/d87/group__dnnLayerList.html

### Emotions Model Training:

Aqui un link directamente al notebook de entrenamient: [CLICK HERE](./Emotions/train/Train_Test_and_Deploy_Emotions.ipynb)

<img src="https://i.ibb.co/t4xrKxS/vlcsnap-2023-11-27-22h52m49s056.png" width="1000">

La red neuronal para detectar emociones es una red neuronal convolucional diseñada específicamente para reconocer y clasificar emociones a travez de imagenes. Para relizar esta tarea correctamente diseñamos la siguiente red neuronal en tensorflow.

<img src="https://i.ibb.co/wBNZrw2/output-1.png" width="1000">

- Conv2d: Esta capa aplica kernels a la imagen y obtiene sus caracteristicas principales.
  
- Activation: Esta capa siempre va despues de una capa convolucional para detectar las activaciones despues del kernel.
  
- BatchNormalization: Esta capa normaliza las activaciones de una capa anterior y acelera el entrenamiento de la red neuronal.
  
- MaxPooling2D: esta capa reduce el número de parámetros en la red y se agrega con el fin de prevenir el overfitting en el entrenamiento.
  
- Dropout: apaga aleatoriamente un porcentaje de neuronas durante cada paso de entrenamiento, lo que mejora la generalización del modelo.
  
- Flatten: esta capa convierte la salida de las capas 3D en un vector unidimensional que finalmente pasan a capas de redes neuronales totalmente conectadas, osea la convierte en un formato que esta capa entiende y puede clasificar.
  
- Dense: en esta capa cada neurona está conectada a todas las neuronas de la capa anterior y tiene el fin de realizar la clasificacion final.

El dataset que utilizamos en este entrenamiento fue [FER-2013](https://huggingface.co/spaces/mxz/emtion/resolve/c697775e0adc35a9cec32bd4d3484b5f5a263748/fer2013.csv) el cual es un dataset con mas de 28k de imnagenes de emociones ya clasificadas.

<img src="https://i.ibb.co/dtLZfhy/image.png" width="1000">

Ya en el notebook tenemos detallado todo el proceso de importar el dataset, separarlo en Test, Train y Validation subsets, unicamente tienes que abir el notebook en colab y darle run all para crear el modelo por ti mismo.

<img src="https://i.ibb.co/D55MdD3/New-Project.png" width="1000">

El notebook de esta red neuronal es: [CLICK HERE](./Emotions/train/Train_Test_and_Deploy_Emotions.ipynb)

**NOTA:** Ademas en la misma carpeta de [Train](./Emotions/train/) agregamos el archivo requirements.txt para que sepas exactamente en que ENV y version de todos los modulos de python se realizo el entrenamiento.

Finalmente despues del entrenamiento podremos obtener un Frozen Graph el cual es un modelo de inferencia que ya esta optimizado para produccion, este sera el archivo que propocionaremos al modulo OpenCV DNN.

<img src="https://i.ibb.co/NyLBGqF/image.png" width="1000">

Si quieres descargar directamente el Frozen Graph: [CLICK HERE](./Emotions/model/emotions-v1.pb)

### Drowsiness Model Training:

Aqui un link directamente al notebook de entrenamiento: [CLICK HERE](./Drowsiness/train/Train_Test_and_Deploy_Blink.ipynb)

<img src="https://i.ibb.co/MVQfBq2/vlcsnap-2023-11-27-23h13m20s380.png" width="1000">

La red neuronal para detectar el estado del ojo es una red neuronal convolucional diseñada específicamente para reconocer un ojo cerrado de uno abierto. Para relizar esta tarea correctamente diseñamos la siguiente red neuronal en tensorflow.

<img src="https://i.ibb.co/9N93ttX/New-Project-3.png" width="1000">

- Conv2d: Esta capa aplica kernels a la imagen y obtiene sus caracteristicas principales.
  
- Activation: Esta capa siempre va despues de una capa convolucional para detectar las activaciones despues del kernel.
  
- BatchNormalization: Esta capa normaliza las activaciones de una capa anterior y acelera el entrenamiento de la red neuronal.
  
- MaxPooling2D: esta capa reduce el número de parámetros en la red y se agrega con el fin de prevenir el overfitting en el entrenamiento.
  
- Dropout: apaga aleatoriamente un porcentaje de neuronas durante cada paso de entrenamiento, lo que mejora la generalización del modelo.
  
- Flatten: esta capa convierte la salida de las capas 3D en un vector unidimensional que finalmente pasan a capas de redes neuronales totalmente conectadas, osea la convierte en un formato que esta capa entiende y puede clasificar.
  
- Dense: en esta capa cada neurona está conectada a todas las neuronas de la capa anterior y tiene el fin de realizar la clasificacion final.

El dataset que utilizamos en este entrenamiento fue [B-eye](https://github.com/altaga/DBSE-monitor/raw/master/Drowsiness/train/dataset/dataset_B_Eye_Images.zip) el cual es un dataset con mas de 4,800 imagenes de ojos abiertos y cerrados ya clasificados.

<img src="https://i.ibb.co/R6Vg6HS/image.png" width="1000">

Ya en el notebook tenemos detallado todo el proceso de importar el dataset, separarlo en Test, Train y Validation subsets, unicamente tienes que abir el notebook en colab y darle run all para crear el modelo por ti mismo.

<img src="https://i.ibb.co/VpmMsWr/New-Project-1.png" width="1000">

El notebook de esta red neuronal es: [CLICK HERE](./Drowsiness/train/Train_Test_and_Deploy_Blink.ipynb)

**NOTA:** Ademas en la misma carpeta de [Train](./Drowsiness/train/) agregamos el archivo requirements.txt para que sepas exactamente en que ENV y version de todos los modulos de python se realizo el entrenamiento.

Finalmente despues del entrenamiento podremos obtener un Frozen Graph el cual es un modelo de inferencia que ya esta optimizado para produccion, este sera el archivo que propocionaremos al modulo OpenCV DNN.

<img src="https://i.ibb.co/NyLBGqF/image.png" width="1000">

Si quieres descargar directamente el Frozen Graph: [CLICK HERE](./Drowsiness/model/blink-v1.pb)

## Online Models Testing:

Ya que tenemos los modelos listos, es necesario pasar a la etapa de Testing, lo que implica la evaluación exhaustiva de los modelos con inputs totalmente fuera del dataset de entrenamiento, el objetivo de esto es verificar la precisión y el rendimiento del modelo.

<img src="https://i.ibb.co/xzcBvMZ/image.png" width="1000">

### Emotions Model Testing:

Aqui un link directamente al notebook de testing: [CLICK HERE](./Emotions/test/Test_Emotions_DNN.ipynb)

Te invitamos a abrir el Notebook y realizar el test tu mismo, el dataset que realizamos fueron 28 imagenes, 7 de cada emocion, con el fin de verificar la presicion con estos datos nuevos.

<img src="https://i.ibb.co/dgGjX2z/image.png" width="1000">

Finalmente los porcentajes de presicion del modelo nos arrojaron lo siguiente.

<img src="https://i.ibb.co/sg256sk/image.png" width="1000">

Podemos notar que la emocion que mas tiene problemas al reconocer es el disgusto y el miedo. Lo que nos indica que este modelo aun tiene un rango de mejora.

### Drowsiness Model Testing:

Aqui un link directamente al notebook de testing: [CLICK HERE](./Drowsiness/test/Test_Blink_DNN.ipynb)

Te invitamos a abrir el Notebook y realizar el test tu mismo, pero al realizar las pruebas con un dataset de prueba que realizamos, llegamos a los siguientes resultados.

<img src="https://i.ibb.co/k8mJvrq/image.png" width="1000">

Finalmente los porcentajes de presicion del modelo nos arrojaron lo siguiente.

<img src="https://i.ibb.co/MRSkRXB/image.png" width="1000">

Podemos notar el modelo es perfecto, sin embargo notamos durante las pruebas in field que al cerrar un poco los ojos o distraerlos de la camara el modelo daba como resultado cerrado, que por fines practicos para detectar la somnolencia o la distraccion esto nos es muy util.

**NOTA:** en el video demo demostramos esta funcion claramente, puedes ir a verlo para ver este proyecto en funcionamiento!

### YoloV3 Model Testing:

Aqui un link directamente al notebook de testing: [CLICK HERE](./Yolo/test/LoadAndTestYolo.ipynb)

Te invitamos a abrir el Notebook y realizar el test tu mismo. Sin embargo te compartimos los resultados del test.

<img src="https://i.ibb.co/2SsxxBm/image.png" width="1000">

El modelo utilizado en la prueba es el modelo Yolo-Tiny, debido a que a que es el mas ligero que podemos usar en este proyecto y sus detecciones nos son suficientes para el buen funcionamiento del proyecto. 

**NOTA:** En la siguiente seccion veras la comparativa de los modelos en varios HW, si deseas utilizar el modelo YoloV3 completo recomendamos utilizar por lo menos la Jetson Nano, ya que esta si realiza procesamiento en GPU y permite una tasa de Frames realista para funcionar.

# Board Setup:

La elección correcta del hardware para estos modelos de AI es escencial para un correcto funcionamiento, ajustarse al consumo energetico del vehiculo y el presupuesto para realizar este proyecto en produccion.

<img src="https://i.ibb.co/bXrn5h9/New-Project-4.png" width="1000">

En todas las boards la configuracion de AWS IoT es la misma, ya que se hace a travez de certificados en la siguiente seccion de codigo.

[CODE](./RPi%20Deploy/iot-mqtt.py)

    EndPoint = "XXXXXXXXXX-ats.iot.us-east-1.amazonaws.com"
    caPath = "opencvDNN/certs/aws-iot-rootCA.crt"
    certPath = "opencvDNN/certs/aws-iot-device.pem"
    keyPath = "opencvDNN/certs/aws-iot-private.key"

El data frame que hay que mandar a la plataforma para que empiecen a aparecer datos es el siguiente.

    {
      "coordinates": [
        -99.4738495,
        19.3749642
      ],
      "color": "#808080",
      "data": "Emotion: Neutral\nState: Awake",
      "id": 98574584180
    }

Y la configuracion del GPS de igual forma no tiene diferencia entre una board a otra ya que todas tienen la misma configuracion de 40 pines.

[CODE](./RPi%20Deploy/Gps/gps.py)

El modulo GPS no requiere ninguna configuracion adicional, este se confugura solo al tenerlo conectado, cuando este ya este funcionando parpadeara el led en la board cada segundo.

<img src="https://i.ibb.co/gzfhDMr/image.png" width="600">

## Raspberry Pi 4:

El Raspberry Pi 4 (RPi4) es una board del tamaño de una tarjeta de credito que nos provee las caracteristicas minimas para que este proyecto se pueda realizar.

<img src="https://i.ibb.co/MMST3xf/image.png" width="600">

- CPU: Broadcom BCM2711 Quad-core
- GPU: VideoCore VI graphics
- RAM: 4GB
- Storage: 32 GB (MicroSD)
- Audio: 3.5mm jack
- Screen Port: 2 × micro HDMI
- Network Interface: Wi-Fi 802.11b/g/n/ac 
- Board Price: [$55 - (11/28/2023, Seeedstudio)](https://www.seeedstudio.com/Raspberry-Pi-4-Computer-Model-B-4GB-p-4077.html)

En el caso de la RPi4 tenemos la fortuna que existen ya versiones compiladas de OpenCV con el modulo DNN. Los pasos para realizar esta instalacion son los siguientes:

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

Una vez hecho esto y podras usar todos los modulos de OpenCV en la RPi4 incluyendo el OpenCV DNN.

## Jetson Nano:

La Jetson Nano es una placa de desarrollo de AI creada por NVIDIA. Esta board nos provee un buen costo beneficio para realizar este proyecto, ademas que permite el procesamiento de imagenes por GPU.

<img src="https://i.ibb.co/f8ZntpX/image.png" width="600">

- CPU: Quad-core ARM Cortex-A57 MPCore
- GPU: 128-core Maxwell
- RAM: 4GB
- Storage: 32 GB (MicroSD)
- Audio: 3.5mm jack
- Screen Port: HDMI 2.0
- Network Interface: Gigabit Ethernet
- Board Price: [$149 - (11/28/2023, Seeedstudio)](https://www.seeedstudio.com/NVIDIA-Jetson-Nano-Development-Kit-B01-p-4437.html)

En el caso de la Jetson Nano ya viene con una version de OpenCV, pero esta no esta adaptada para funcionar con CUDA y cuDNN, que son los modulos en la jetson que nos permiten relizar el procesamiento de imagenes en GPU. Durante el proceso de setup usaremos un [script](./OpenCV%20Scripts/build_opencv_nano.sh) ya diseñado para configurar automaticamente OpenCV en nuestra board.

Configuracion:

- Update apt repository.
    
      sudo apt update

- Descarga el script de instalacion, sientete libre de abrir el achivo de script y revisarlo.

      wget https://raw.githubusercontent.com/altaga/Open-Driving-Monitor/main/OpenCV%20Scripts/build_opencv_nano.sh

- Para que el script tenga exito revisa que los parametros del build de OpenCV tengan las versiones correspondientes a nuestra version de jetpack.

  <img src="https://i.ibb.co/93N1N5Y/New-Project-5.png" width="1000">

- En el archivo de srcipt cambia los numeros de CUDA_ARCH_BIN y CUDNN_VERSION si no coinciden en tu jetson.

      ...
      -D CUDA_ARCH_BIN=5.3 # Cuda Arch BIN 
      -D CUDA_ARCH_PTX=
      -D CUDA_FAST_MATH=ON
      -D CUDNN_VERSION='8.2' # cuDNN
      ...
  
- En la Jetson Nano el proceso de build puede tardar de 4 a 5 horas, igual recomendamos que tenga un ventilador pequeño que dicipe el calor que genere la board, sino corres el riesgo de que se apague la board a medio proceso y tengas que realizarlo desde el inicio.

Una vez hecho esto y podras usar todos los modulos de OpenCV en la Jetson Nano incluyendo el OpenCV DNN y procesamiento en GPU.

## Jetson AGX Xavier:

La Jetson AGX Xavier (Jetson AGX) es una placa de desarrollo de AI creada por NVIDIA. Esta board nos provee el mejor rendimiento en modelos de AI por el procesamiento en GPU avanzado, pero siendo un hardware con un alto costo.

<img src="https://i.ibb.co/J74dZP5/61m739zmag-L-AC-SX679.jpg" width="600">

- CPU: 8-core ARMv8.2
- GPU: NVIDIA Volta architecture with 512
- RAM: 32 GB
- Audio: Integrated audio codec HDMI
- Screen Port: HDMI 2.0, eDP 1.4, DP 1.2
- Network Interface: Gigabit Ethernet
- Board Price: [$699 - (11/28/2023, Seeedstudio)](https://www.seeedstudio.com/NVIDIA-Jetson-AGX-Xavier-Development-Kit-p-4418.html)

En el caso de la Jetson AGX ya viene con una version de OpenCV, pero esta no esta adaptada para funcionar con CUDA y cuDNN, que son los modulos en la jetson que nos permiten relizar el procesamiento de imagenes en GPU. Durante el proceso de setup usaremos un [script](./OpenCV%20Scripts/build_opencv_agx.sh) ya diseñado para configurar automaticamente OpenCV en nuestra board.

Configuracion:

- Update apt repository.
    
      sudo apt update

- Descarga el script de instalacion, sientete libre de abrir el achivo de script y revisarlo.

      wget https://raw.githubusercontent.com/altaga/Open-Driving-Monitor/main/OpenCV%20Scripts/build_opencv_agx.sh

- Para que el script tenga exito revisa que los parametros del build de OpenCV tengan las versiones correspondientes a nuestra version de jetpack.

  <img src="https://i.ibb.co/xSg1mPS/New-Project-6.png" width="1000">

- En el archivo de srcipt cambia los numeros de CUDA_ARCH_BIN y CUDNN_VERSION si no coinciden en tu jetson.

      ...
      -D CUDA_ARCH_BIN=7.2 # Cuda Arch BIN
      -D CUDA_ARCH_PTX=
      -D CUDA_FAST_MATH=ON
      -D CUDNN_VERSION='8.6' # cuDNN
      ...
  
- En la Jetson AGX el proceso de build puede tardar de 1 a 2 horas, la board ya tiene su propio ventilador integrado, entonces no tienes que preocuparte por la temperatura en absoluto.

Una vez hecho esto y podras usar todos los modulos de OpenCV en la Jetson AGX incluyendo el OpenCV DNN y procesamiento en GPU.

# Comparison Benchmarks:

Comparamos el funcionamiento de todas las redes neuronales en cada una de las boards con el fin de obtener datos de FPS de cada uno de los modelos.

### RPi4:

- Drowsiness:

  Video: Click on the image

  [<img src="https://i.ibb.co/ZBxgtS4/logo-1.png" width="300">](https://youtu.be/ALhJKYtyX7Q)  

  Sorry github does not allow embed videos.

- Emotions:

  Video: Click on the image

  [<img src="https://i.ibb.co/ZBxgtS4/logo-1.png" width="300">](https://youtu.be/37lpyZrZbPw)

  Sorry github does not allow embed videos.

- Yolo:

  Video: Click on the image

  [<img src="https://i.ibb.co/ZBxgtS4/logo-1.png" width="300">](https://youtu.be/b_tWIjI1-8c)

  Sorry github does not allow embed videos.

### Jetson Nano:

- Drowsiness:

  Video: Click on the image

  [<img src="https://i.ibb.co/ZBxgtS4/logo-1.png" width="300">](https://youtu.be/c6ioMZixa9U)

  Sorry github does not allow embed videos.

- Emotions:

  Video: Click on the image

  [<img src="https://i.ibb.co/ZBxgtS4/logo-1.png" width="300">](https://youtu.be/hM1KTele-LE)

  Sorry github does not allow embed videos.

- Yolo:

  Video: Click on the image

  [<img src="https://i.ibb.co/ZBxgtS4/logo-1.png" width="300">](https://youtu.be/-PLD04Vq0mI)

  Sorry github does not allow embed videos.

### Jetson AGX:

- Drowsiness:

  Video: Click on the image
  
  [<img src="https://i.ibb.co/ZBxgtS4/logo-1.png" width="300">](https://youtu.be/1xDIpZbogDo)

  Sorry github does not allow embed videos.

- Emotions:

  Video: Click on the image

  [<img src="https://i.ibb.co/ZBxgtS4/logo-1.png" width="300">](https://youtu.be/sdUzYwAY8LI)

  Sorry github does not allow embed videos.

- Yolo:

  Video: Click on the image

  [<img src="https://i.ibb.co/ZBxgtS4/logo-1.png" width="300">](https://youtu.be/bbcro-RTcR0)

  Sorry github does not allow embed videos.

## Benchmarks:

Una vez terminadas las pruebas y sacando los promedios de los FPS de cada prueba, obtenemos los siguientes resultados.

- El tiempo de procesamiento o inferencia de cada una de las boards fue el siguiente, esto es solo el tiempo que le toma a la imagen pasar por la red y obtener un resultado.

  <img src="https://i.ibb.co/m4VbZBx/image.png" width="600">

- El tiempo real que toma nuestro programa en realizar el pre procesamiento, inferencia y despliegue de la imagen en cada una de las boards es.

  <img src="https://i.ibb.co/qF08658/image.png" width="600">

Finalmente decidimos utilizar la RPi4 como board final debido a que los FPS que nos provee nos es suficiente para este proyecto, sin embargo puedes usar la board que consideres mejor.

# Open Driving Navigator (WebPage):

Teniendo ya la data del vehiculo realizamos una plataforma web que nos permite monitorizar en tiempo real los autos que esten mandando data al sistema. Esto con el fin de prevenir accidentes y dar mayor informacion a los demas vehiculos.

<img src="https://i.ibb.co/pnGjwNy/image.png" width="600">

URL: https://open-driving-navigator.vercel.app/

NOTA: la pagina requiere los permisos de localizacion para que al mandar datos a la plataforma podamos ver los autos aprecer en nuestra localizacion.

## NextJS:

Para la plataforma web se utilizo el framework de [NextJS](https://nextjs.org/) en su version mas recente (11/28/2023) y los mapas open source [Open Layers](https://openlayers.org/).

Todo el codigo de la plagina web es open source y esta en el siguiente link.

[CODE](./Map%20WebPage/)

## AWS Iot:

La comunicacion entre los devices y la pagina web se realiza mediante AWS IoT ya que nos permite manetener una conexion segura en todo momento y que usamos el protocolo MQTTS.

<img src="https://i.ibb.co/Jq9W7GW/mqtts-drawio.png" width="600">

La configuracion de AWS IoT en la pagina web se realiza en la siguiente seccion de codigo.

[CODE](./Map%20WebPage/src/utils/aws-configuration.js)

    var awsConfiguration = {
      poolId: "us-east-1:xxxxxxxxxxxxxxxxxxxxxxxxxxxxx", // 'YourCognitoIdentityPoolId'
      host:"xxxxxxxxxxxxxxxxxxxxx.iot.us-east-1.amazonaws.com", // 'YourAwsIoTEndpoint', e.g. 'prefix.iot.us-east-1.amazonaws.com'
      region: "us-east-1" // 'YourAwsRegion', e.g. 'us-east-1'
    };
    module.exports = awsConfiguration;

## Vercel:

El despliegue de la pagina web a internet es posible hacerlo facilmente y de forma gratuita gracias a la plataforma de Vercel, [Check Free Plan Limits](https://vercel.com/docs/limits/overview)

<img src="https://i.ibb.co/vwWjdtG/image.png" width="600">

Solo es necesario conectar el repositorio principal de la pagina web con Vercel y automaticamente realizara el deployment.

URL: https://open-driving-navigator.vercel.app/

# Open Driving Emulator (Android Native App):

Podemos realizar la simulacion de un autommovil mandando datos desde el simulador hacia la plataforma.

<img src="https://i.ibb.co/pnGjwNy/image.png" width="600">

NOTA: la app requiere los permisos de localizacion para que al mandar datos a la plataforma podamos ver el auto simulado aprecer en nuestra localizacion.

## React Native Setup:

Para realizar el emulador se utilizo el framework de [React Native](https://reactnative.dev/) en su version mas recente (11/28/2023).

Todo el codigo de la app es open source y esta en el siguiente link.

[CODE](./Emulator%20ReactNative/)

## AWS Iot:

La comunicacion entre la app y la pagina web se realiza mediante AWS IoT ya que nos permite manetener una conexion segura en todo momento y que usamos el protocolo MQTTS.

<img src="https://i.ibb.co/Jq9W7GW/mqtts-drawio.png" width="600">

La configuracion de AWS IoT en la app es exactamente el mismo que el de la pagina web, ya que ambos funcionan con javascript y este se realiza en la siguiente seccion de codigo.

[CODE](./Emulator%20ReactNative/src/utils/aws-configuration.js)

    var awsConfiguration = {
      poolId: "us-east-1:xxxxxxxxxxxxxxxxxxxxxxxxxxxxx", // 'YourCognitoIdentityPoolId'
      host:"xxxxxxxxxxxxxxxxxxxxx.iot.us-east-1.amazonaws.com", // 'YourAwsIoTEndpoint', e.g. 'prefix.iot.us-east-1.amazonaws.com'
      region: "us-east-1" // 'YourAwsRegion', e.g. 'us-east-1'
    };
    module.exports = awsConfiguration;

## Google Play:

Para facilitar el que puedan probarla plataforma web. Este es el enlace de la beta de nuestra aplicacion, con ella podras mandar informacion a nuestro mapa online y simular nuestro sistema sin hardware adicional. Los modelos de AI podran ser probados desde la aplicacion en versiones futuras mediante [OpenCV.JS](https://docs.opencv.org/3.4/d5/d10/tutorial_js_root.html)

<img src="https://i.ibb.co/sQfN4y7/image.png" width="1000">

URL: https://play.google.com/store/apps/details?id=com.altaga.ODS

## How use it:

Para utilizar la app, solo tendras que abrir la aplicacion, aceptar los permisos de localizacion y finalmente presional "Start Emulation"

<img src="https://i.ibb.co/nwrjCmW/vlcsnap-2023-11-28-16h57m14s456.png" width="32%"> <img src="https://i.ibb.co/vsLQChD/vlcsnap-2023-11-28-16h57m10s157.png" width="32%"> <img src="https://i.ibb.co/nw24BLp/vlcsnap-2023-11-28-16h57m33s176.png" width="32%">

# The Final Product:

### Complete System:

<img src="https://i.ibb.co/WsggbsV/20231124-151109.jpg">

### In-car system:

<img src="https://i.ibb.co/Z6nF54M/20231127-132608.png" width="49%"> <img src="https://i.ibb.co/PQm4mGk/20231127-125635.png" width="49%">

# EPIC DEMO:

Video: Click on the image
[![Car](https://i.ibb.co/ZBxgtS4/logo-1.png)](pending...)

Sorry github does not allow embed videos.

# Commentary:



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
