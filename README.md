# Plus-Ultra
Plus Ultra is a tool to enhance digital media. Currently it supports upscaling and jpeg artifact removal from 2D illustrations or images resembling anime style art and manga.

## Disclaimer
I DONOT own the rights to any of the screenshots , images or art presented in this repo. The rights are held by their original owners and makers. I am using it in my repo for demonstration purposes only !
If you are the original owner of the images used here and want them to be removed , please contact me and i will make sure to remove them !

## Comparison between an input image upscaled by 4x using Bicubic upscaling and 2D_Art_0 Model
![Violet evergarden comparison](https://raw.githubusercontent.com/AtharvBhat/Plus-Ultra/master/results/VioletEvergarden_comparison.png)
![Persona 5 comparison](https://raw.githubusercontent.com/AtharvBhat/Plus-Ultra/master/results/Persona5_comparison.png)
![Persona 5 waifu2x comparison](https://raw.githubusercontent.com/AtharvBhat/Plus-Ultra/master/results/persona5-waifu2x-comparison.png)

### What it is :
This is an open source tool which uses various techniques to enhance digital media and currently only supports 2D illustrations and anime style images.
It includes three models which all provide a different result. You can try them all and use the one which best suits your needs.

I recommend you start off with **2D_Art_0** as your default model. The rest of the models tend to oversharpen most images, but i have still included them as they work better than the default model in certain scenarios like the **Ghost in a Shell 1995** movie screenshot. You can check all the test cases in the `test_inputs` and `test_outputs` folders.

### What it isn't :
The provided 2D Art models were trained on frames extracted from dozens of anime videos scraped from youtube.
The 2D Art model may not provide good results on realistic scenes with high-frequency texture detail since animes generally lack those details.

It isnt magic. It will not magically come up with new information about the image and make it look as good as you imagine. It infers things about the image and tries to approximate a better looking picture.  I will go into more technical details later.


## Installation and Requirements


* The tool is written in **python** programming language and uses the **fastai** library which runs on top of **pytorch** framework. Hence you will need to install the latest version of **pytorch** along with **fastai** library.

* Nvidia GPU is required

* **I recommend using anaconda for installing pytorch and fastai**

### Step 1.
Install Anaconda with python verion 3.x for your appropriate OS
https://www.anaconda.com/products/individual

### Step 2.
Open Anaconda terminal and create a new environment if needed. ( Optional but recommended to avoid conflicts )
Install Pytorch framework by running the following command in the anaconda terminal https://pytorch.org/

`conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`

### Step 3.
Install the fastai library by running this command in anaconda terminal

`pip install fastai`

#### Check installation
Open the python interpreter by typing `python` in your anaconda terminal and copy and run the following code
```
import fastai.utils
fastai.utils.show_install(1)
```

If you get an output that looks something like this , you have successfully installed all libraries
```
=== Software ===
python        : 3.8.2
fastai        : 1.0.61
fastprogress  : 0.2.3
torch         : 1.5.0
nvidia driver : 446.14
torch cuda    : 10.2 / is available
torch cudnn   : 7604 / is enabled

=== Hardware ===
nvidia gpus   : 1
torch devices : 1
  - gpu0      : 8192MB | GeForce GTX 1070 Ti
```

If your output does not match this , you can find troubleshooting steps here : https://docs.fast.ai/troubleshoot.html

## Usage
* To use this tool , first download the repo.
* Open anaconda terminal and navigate to the downloaded folder using the `cd` command followed by the path of the folder
```
cd name-of-subfolder/sub-subfolder/
```
* Activate the environment where you installed fastai library ( Optional )

### Using upscale.py
```
usage: upscale.py [-h] [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR] [--model MODEL] [--fp_16]
                  [--scale_factor SCALE_FACTOR] [--keep_size] [--tta]

optional arguments:
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        String containing path of folder containing input images
  --output_dir OUTPUT_DIR
                        String containing path of output directory
  --model MODEL         File name of the model to use for upscaling with extention
  --fp_16               pass this argument to use mixed precision inference
  --scale_factor SCALE_FACTOR
                        scale by 2x or 4x
  --keep_size           Keep input size
  --tta                 test time augmentation. take 4 transformations of the image and average their results
```
To get started with the provided test inputs , run the following line of code
```
python upscale.py --input_dir "test_inputs" --output_dir "test_outputs" --model "2D_Art_0.pkl" --scale_factor 4 --fp_16
```

### Training your own model

To Train your own model , you can use the `train.ipynb` jupyter notebook.
* Set Paths to your low resolution and high resolution images by edditing the following lines of code
```
low_res_images_path = ""
high_res_images_path = ""
```
* Your low resolution images should be about scaled by 0.5 times the original image
* Set model Hyperparameters according to your input image size and GPU
* NOTE : FP16 training is not supported by older GPU's. In newer GPU's it will allow you to train with larger batchsize and can speedup training
* Use the Learning Rate Finder to select an appropriate learning rate. (If you are now sure how to interpret the graph , just use the default of 1e-4, it works fine)
* Edit the following line of code and set a name for your model. Your model will be stored inside a "models" folder in your low resolution image directory
```
learn.save("NAME OF MODEL")
```
* After you are done Training, use the final export block to export your model as a ".pkl" file. (Note : Loading the model does not require the extention but make sure to provide the .pkl extention in the export command)

You have now trained your own model ! Copy paste it in the models directory of the repo and use the upscale.py file to see how it performs on the test inputs !

## Technical tidbits:
* It uses the SR-Resnet model first shown in the Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network https://arxiv.org/abs/1609.04802 paper but combines it with feature loss described in Perceptual Losses for Real-Time Style Transfer and Super-Resolution https://arxiv.org/abs/1603.08155.
* `twtygqyy`'s implementation of Srresnet https://github.com/twtygqyy/pytorch-SRResNet is modified to do 2x instead of 4x upscaling. For 4x upscaling the code simplay passes the same image through the model twice.
* Jeremy howard's implementation of Feature loss is was used as the loss function for training.

## References
1. Srresnet paper : Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network https://arxiv.org/abs/1609.04802
2.  `twtygqyy`'s implementation of Srresnet https://github.com/twtygqyy/pytorch-SRResNet
3. Perceptual Losses for Real-Time Style Transfer and Super-Resolution https://arxiv.org/abs/1603.08155
4. Fastai coursev3 Lesson 7 implementation of Feature loss : https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson7-superres.ipynb

