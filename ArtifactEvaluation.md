# ONNXRuntime with SYCL EP and CUDA EP : Instructions to build & run performance Demos

## Requirements 
In order to build and run the docker image from the docker file provided *( dockerFile.sycl )*, the host machine still needs to have CUDA installed with a version **no prior to v11.4**.   

## Building the docker image 
*Following instructions are run within the base directory of ORT.*
```
$ docker build -t ort_sycl_cuda -f dockerfile_ort_nvidia_gpu.sycl .
```

## Running a docker container
```
sudo docker run --rm --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it ort_sycl_cuda
```

## Running VGG-16 and Resnet-50 Demos
*Following instructions are run within the docker container.*
### Using SYCL EP
```
cd /home/networks/buildSYCL

./testNetwork --ep SYCL --image ../../imagenet-examples/10.jpg.txt --model ../models/vgg16_nhwc.onnx --device GPU --vendor Nvidia 

./testNetwork --ep SYCL --image ../../imagenet-examples/10.jpg.txt --model ../models/resnet_nhwc.onnx --device GPU --vendor Nvidia 

```

### Using CUDA EP
```
cd /home/networks/buildCUDA

./testNetwork --ep CUDA --image ../../imagenet-examples/10nchw.jpg.txt --model ../models/vgg16-12.onnx

./testNetwork --ep CUDA --image ../../imagenet-examples/10nchw.jpg.txt --model ../models/resnet50-v1-12.onnx
```