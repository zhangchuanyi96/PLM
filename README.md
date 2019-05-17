Introduction
------------
This is the source code for our paper **Peer-learning Network for Fine-Grained Recognition**

Installation
------------
After creating a virtual environment of python 3.5, run `pip install -r requirements.txt` to install all dependencies

How to use
------------
The code is currently tested only on GPU
* Data Preparation

    Download data into PML root directory and uncompress them using
    ```
    wget https://web-fg-data.oss-cn-hongkong.aliyuncs.com/CUB200-WEB100.tar.gz
    wget https://web-fg-data.oss-cn-hongkong.aliyuncs.com/CUB200-WEB300.tar.gz
    tar -xvf CUB200-WEB100.tar.gz
    tar -xvf CUB200-WEB300.tar.gz
    ```
    Training images in **CUB200-WEB100** and **CUB200-WEB100** are all crawled from the Internet while their test images are the test data in CUB200-2011.
    **CUB200-WEB100** has 20000 training images in total while **CUB200-WEB100** has 58433 training images in total.

* Demo

    - If you just want to do a quick test on the model and check the final fine-grained recognition performance, please follow subsequent steps

      - Download one of the following trained model into `model/` using
          ```
          wget https://web-fg-data.oss-cn-hongkong.aliyuncs.com/Model/WEB100-demo-79.12.pth
          wget https://web-fg-data.oss-cn-hongkong.aliyuncs.com/Model/WEB300-demo-89.11.pth
          ```
          | Model                 | Description                                | Performance(%) |
          | --------------------- | ------------------------------------------ | -------------- |
          | WEB100-demo-79.12.pth | leveraged 100 web images for each category | 79.12          |
          | WEB300-demo-89.11.pth | leveraged 300 web images for each category | 89.11          |
      - Create a soft link for data by `ln -s CUB200-WEB100 cub200`
      - Activate virtual environment (e.g. conda)
      - Modify `CUDA_VISIBLE_DEVICES` to proper cuda device id in `cub200_demo.sh` 
      - Modify the model name in `cub200_demo.sh` according to the model downloaded.
      - Run demo using `bash cub200_demo.sh`

    - If you want to de a quick test on the model of cifar10 / cifar100, please follow subsequent steps
      - Download one of the following trained model into `cifar/model/` using
          ```
          wget https://web-fg-data.oss-cn-hongkong.aliyuncs.com/Model/cifar10_demo-77.40.pth
          wget https://web-fg-data.oss-cn-hongkong.aliyuncs.com/Model/cifar100_demo-34.46.pth
          ```
          | Model                   | Description                                      | Performance(%) |
          | ----------------------- | ------------------------------------------------ | -------------- |
          | cifar10_demo-77.40.pth  | trained on cifar10 dataset with noise rate 0.45  | 77.40          |
          | cifar100_demo-34.46.pth | trained on cifar100 dataset with noise rate 0.45 | 34.46          |
      - Activate virtual environment (e.g. conda)
      - Go into `cifar` directory
      - Modify `CUDA_VISIBLE_DEVICES` to proper cuda device id in `cifar10_demo.sh` or `cifar100_demo.sh`, 
      - Modify the model name in `cifar10_demo.sh` or `cifar100_demo.sh` according to the model downloaded.
      - Run demo using `cifar10_demo.sh` or `cifar100_demo.sh` accordingly

* Source Code

    - If you want to train the whole network from begining using source code on the fine-grained dataset, please follow subsequent steps
    
      - Create soft link to dataset by `ln -s CUB200-WEB100 cub200`
      - Modify `CUDA_VISIBLE_DEVICES` to proper cuda device id in `cub200_run.sh`
      - Activate virtual environment(e.g. conda) and then run the script
          ```
          bash cub200_run.sh
          ```

    - If you want to train on the cifar10 / cifar100 dataset, please follow subsequent steps
    
      - Go into `cifar` directory
      - Modify `CUDA_VISIBLE_DEVICES` to proper cuda device id in `cifar10_run.sh` or `cifar100_run.sh`
      - Activate virtual environment(e.g. conda) and then run the script
          ```
          bash cifar10_run.sh
          ```
          or 
          ```
          bash cifar100_run.sh
          ```