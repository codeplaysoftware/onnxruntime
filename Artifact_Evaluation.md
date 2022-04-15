## Intel CPU and GPU experiments
Download the docker image from - https://www.dropbox.com/s/t6klvu0iuri5c2s/ort-intelocl0.2.tar?dl=0 \
Run the following commands
```code
$ sudo docker load -i ort-intelocl0.2.tar
$ sudo docker images
REPOSITORY   TAG       IMAGE ID       CREATED         SIZE
<none>       <none>    735233dbc905   6 minutes ago   2.96GB
ubuntu       18.04     f5cbed4244ba   9 days ago      63.2MB
```
Copy the Image ID (735233dbc905) for the next command
```code
$ sudo docker run --device /dev/dri:/dev/dri -it <ImageID>
```
To run the network on CPU devices
```code
$ ./testNetwork --model models/vgg16_nhwc.onnx --image models/10.jpg.txt --device cpu
$ ./testNetwork --model models/resnet50_nhwc.onnx --image models/10.jpg.txt --device cpu
```
To run the network on GPU devices
```code
$ ./testNetwork --model models/vgg16_nhwc.onnx --image models/10.jpg.txt --device gpu
$ ./testNetwork --model models/resnet50_nhwc.onnx --image models/10.jpg.txt --device gpu
```

## Risc-V experiments
Download the docker image from -https://www.dropbox.com/s/b9e4pzy02f6l214/ort-riscv0.3.tar?dl=0 \
Run the following commands
```code
$ sudo docker load -i ort-riscv0.3.tar
$ sudo docker images
REPOSITORY   TAG       IMAGE ID       CREATED         SIZE
<none>       <none>    735233dbc905   6 minutes ago   1.96GB
ubuntu       18.04     f5cbed4244ba   9 days ago      63.2MB
```
Copy the Image ID (735233dbc905) for the next command
```code
$ sudo docker run --device /dev/dri:/dev/dri -it <ImageID>
```
To run the network on RISC-V devices
```code
$ ./testNetwork --model models/vgg16_nhwc.onnx --image models/10.jpg.txt 
$ ./testNetwork --model models/resnet50_nhwc.onnx --image models/10.jpg.txt 
```

## Nvidia GPU experiments
First check if host system has CUDA Version >=11.4.1 (for instance through these :)
```code
$ nvidia-smi
$ nvcc --version
```
The docker image is available here - https://github.com/codeplaysoftware/onnxruntime/blob/sycl_nhwc/dockerfile_ort_nvidia_gpu.sycl \
Building and running the docker image :
```
$ sudo docker build -t ort_dpcpp_nvidia_gpu -f dockerfile_nvidia_gpu.sycl
$ sudo docker run --rm --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it ort_dpcpp_nvidia_gpu
```
To run the network on RISC-V devices
```code
$ ./testNetwork --model models/vgg16_nhwc.onnx --image models/10.jpg.txt 
$ ./testNetwork --model models/resnet50_nhwc.onnx --image models/10.jpg.txt 
```
