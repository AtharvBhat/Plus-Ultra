#import libs
import fastai
from fastai.vision import *
from srresnet import srresnet
import time
import os
from argparse import ArgumentParser
from tqdm import tqdm
import warnings

#turn off pytorch warnings
warnings.filterwarnings('ignore')


#define arguments
parser = ArgumentParser()
parser.add_argument("--input_dir", help="String containing path of folder containing input images")
parser.add_argument("--output_dir", help="String containing path of output directory")
parser.add_argument("--model", help="File name of the model to use for upscaling with extention")
parser.add_argument("--fp_16", help="pass this argument to use mixed precision inference", action="store_true")
parser.add_argument("--scale_factor", help="scale by 2x or 4x")
parser.add_argument("--keep_size", help="Keep input size", action="store_true")
parser.add_argument("--tta", help="test time augmentation. take 4 transformations of the image and average their results", action="store_true")
args = parser.parse_args()

#parse args
input_dir = str(args.input_dir)
output_dir = str(args.output_dir)
model = str(args.model)
fp_16 = args.fp_16
scale_factor = int(args.scale_factor)
keep_size = args.keep_size
tta = args.tta

#read input images and store them in a list
input_images = os.listdir(input_dir)
print(f"Found {len(input_images)} images in input folder !")

#check if output folder exists. if it doesnt , create one
if not os.path.exists(output_dir):
    print("Output directory not found")
    os.makedirs(output_dir)
    print("Created output directory with specified path")

#load model
print("Loading model")
upscaler = load_learner("models/", model)
upscaler.model.eval()

#convert model for mixed precision
if fp_16 is not None:
    print("Using mixed precision inference")
    upscaler.to_fp16()

start_time = time.time()

#run inference on images
for image in tqdm(input_images):
    #load image as tensor
    img = open_image(input_dir + "/" + image)
    img_height, img_width = img.size

    if tta :
        #image transformations
        im1 = PIL.Image.fromarray(image2np(img.data*255).astype(np.uint8)).convert('RGB')
        im2 = PIL.ImageOps.flip(im1)
        im3 = PIL.ImageOps.mirror(im1)
        im4 = PIL.ImageOps.flip(im3)

        #convert images back to tensors
        im1 = Image(pil2tensor(im1, np.float32).div_(255))
        im2 = Image(pil2tensor(im2, np.float32).div_(255))
        im3 = Image(pil2tensor(im3, np.float32).div_(255))
        im4 = Image(pil2tensor(im4, np.float32).div_(255))

        #get prdiction of image 1
        _,pred1,_ = upscaler.predict(im1)
        pred1 = torch.clamp(pred1,0,1)
        gc.collect()
        torch.cuda.empty_cache()

        #pass same image again for 4x upscaling
        if scale_factor == 4:
            _,pred1,_ = upscaler.predict(pred1)
            pred1 = torch.clamp(pred1,0,1)
            gc.collect()
            torch.cuda.empty_cache()

        #get prdiction of image 2
        _,pred2,_ = upscaler.predict(im2)
        pred2 = torch.clamp(pred2,0,1)
        gc.collect()
        torch.cuda.empty_cache()

        #pass same image again for 4x upscaling
        if scale_factor == 4:
            _,pred2,_ = upscaler.predict(pred2)
            pred2 = torch.clamp(pred2,0,1)
            gc.collect()
            torch.cuda.empty_cache()

        #get prdiction of image 3
        _,pred3,_ = upscaler.predict(im3)
        pred3 = torch.clamp(pred3,0,1)
        gc.collect()
        torch.cuda.empty_cache()

        #pass same image again for 4x upscaling
        if scale_factor == 4:
            _,pred3,_ = upscaler.predict(pred3)
            pred3 = torch.clamp(pred3,0,1)
            gc.collect()
            torch.cuda.empty_cache()

        #get prdiction of image 4
        _,pred4,_ = upscaler.predict(im4)
        pred4 = torch.clamp(pred4,0,1)
        gc.collect()
        torch.cuda.empty_cache()

        #pass same image again for 4x upscaling
        if scale_factor == 4:
            _,pred4,_ = upscaler.predict(pred4)
            pred4 = torch.clamp(pred4,0,1)
            gc.collect()
            torch.cuda.empty_cache()

        #convert all tensors back to images
        im1 = PIL.Image.fromarray(image2np(pred1.data*255).astype(np.uint8)).convert('RGB')
        im2 = PIL.Image.fromarray(image2np(pred2.data*255).astype(np.uint8)).convert('RGB')
        im3 = PIL.Image.fromarray(image2np(pred3.data*255).astype(np.uint8)).convert('RGB')
        im4 = PIL.Image.fromarray(image2np(pred4.data*255).astype(np.uint8)).convert('RGB')

        #inverse transforms
        im1 = np.array(im1).astype(np.float32)
        im2 = np.array(PIL.ImageOps.flip(im2)).astype(np.float32)
        im3 = np.array(PIL.ImageOps.mirror(im3)).astype(np.float32)
        im4 = np.array(PIL.ImageOps.mirror(PIL.ImageOps.flip(im4))).astype(np.float32)

        #take average of all images
        img = (im1+im2+im3+im4)/4
        img = PIL.Image.fromarray(img.astype(np.uint8)).convert('RGB')
        pred = Image(pil2tensor(img, np.float32).div_(255))


    else:
        #get predictions
        _,pred,_ = upscaler.predict(img)
        pred = torch.clamp(pred,0,1)

        #pass same image again for 4x upscaling
        if scale_factor == 4:
            _,pred,_ = upscaler.predict(pred)
            pred = torch.clamp(pred,0,1)
            gc.collect()
            torch.cuda.empty_cache()

    if keep_size :
        x = PIL.Image.fromarray(image2np(pred.data*255).astype(np.uint8)).convert('RGB')
        x = x.resize((img_width, img_height))
        pred = Image(pil2tensor(x, np.float32).div_(255))
        

    #save image
    Image.save(pred, output_dir + "/" +image)

print(f"Time taken : {(time.time() - start_time)} seconds")