# Object detection app

## Download models 

* [SSD mobilenet](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API#use-existing-config-file-for-your-model)

* [Yolo V3](https://pjreddie.com/darknet/yolo/)

## Install

### Raspberry

* Make sure to have `git`:

```
sudo apt install git
```

* Install OpenCV 4 fast and optimized for Raspberry Pi:

```
git clone https://github.com/dlime/Faster_OpenCV_4_Raspberry_Pi.git
cd Faster_OpenCV_4_Raspberry_Pi/debs
sudo dpkg -i OpenCV*.deb
sudo ldconfig
```

* Install dependencies. I prefer to use `.deb` files in Raspberry Pi instead of `pip` because it doesn't have to compile sources. For installing pandas takes more than 1 hour using `pip`.

```
sudo apt install python3-dotenv python3-pandas python3-pandas python3-picamera python3-flask python3-celery python3-redis
```

* Install object-detection

```
git clone https://github.com/cristianpb/object-detection.git
cd object-detection/
sudo apt install vim
```
