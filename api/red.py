import os
from os import path, listdir, makedirs
from glob import glob

import random
import time
import copy

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import tensorboard

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

 
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from django.http import HttpResponse
from . import cartesianRemap as remap

cfg = dict(
    seed = 2023,
    train_dir = "./data/train/",
    val_dir = "./data/validation/",
    test_dir = "./data/test/",
    num_images = 496,
    image_size = 128,
    
    num_epochs = 501,
    batch_size = 128,       # paper original usa instance normalization  
    lr = 2e-4,
    display_step = 5,
    adversarial_criterion = nn.BCEWithLogitsLoss(), # vanilla loss 
    recon_criterion = nn.L1Loss(), 
    lambda_recon = 100,   
)
# Seteamos una semillas

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def colors_to_classes(label):
    '''
    Given a labelling with colors, return a labelling with class numbers
    '''
    # Convierto los colores en las etiquetas de all_classes
    mask = np.zeros((label.shape[0],label.shape[1]),dtype=np.dtype('i'))
    # (pelvis - grey)
    mask[(label[:,:,0] == 151) * (label[:,:,1]==151) * (label[:,:,2]==147)]=9
    # (spleen - pink)
    mask[(label[:,:,0] == 255) * (label[:,:,1]==0) * (label[:,:,2]==255)]=8
    # (liver - purple)
    mask[(label[:,:,0] == 100) * (label[:,:,1]==0) * (label[:,:,2]==100)]=7
    # (surrenalGland - cyan)
    mask[(label[:,:,0] == 0) * (label[:,:,1]==255) * (label[:,:,2]==255)]=6
    # (kidney - yellow)
    mask[(label[:,:,0] == 255) * (label[:,:,1]==255) * (label[:,:,2]==0)]=5
    # (gallbladder - green)
    mask[(label[:,:,0] == 0) * (label[:,:,1]==255) * (label[:,:,2]==0)]=4
    # (pancreas - blue)
    mask[(label[:,:,0] == 0) * (label[:,:,1]==0) * (label[:,:,2]==255)]=3
    # (artery - red)
    mask[(label[:,:,0] == 255) * (label[:,:,1]==0) * (label[:,:,2]==0)]=2
    # (bones - white)
    mask[(label[:,:,0] == 255) * (label[:,:,1]==255) * (label[:,:,2]==255)]=1

    return mask

def segmentation_to_colors(real_predictions):
    
    red_channel = np.zeros(real_predictions.shape)
    green_channel = np.zeros(real_predictions.shape)
    blue_channel = np.zeros(real_predictions.shape)
    
    # (bones - white)
    red_channel[real_predictions==1] = 255
    green_channel[real_predictions==1] = 255
    blue_channel[real_predictions==1] = 255
    # (artery - red)
    red_channel[real_predictions==2] = 255
    green_channel[real_predictions==2] = 0
    blue_channel[real_predictions==2] = 0
    # (pancreas - blue)
    red_channel[real_predictions==3] = 0
    green_channel[real_predictions==3] = 0
    blue_channel[real_predictions==3] = 255
    # (gallbladder - green)
    red_channel[real_predictions==4] = 0
    green_channel[real_predictions==4] = 255
    blue_channel[real_predictions==4] = 0
    # (kidney - yellow)
    red_channel[real_predictions==5] = 255
    green_channel[real_predictions==5] = 255
    blue_channel[real_predictions==5] = 0
    # (surrenalGland - cyan)
    red_channel[real_predictions==6] = 0
    green_channel[real_predictions==6] = 255
    blue_channel[real_predictions==6] = 255
    # (liver - purple)
    red_channel[real_predictions==7] = 100
    green_channel[real_predictions==7] = 0
    blue_channel[real_predictions==7] = 100
    # (spleen - pink)
    red_channel[real_predictions==8] = 255
    green_channel[real_predictions==8] = 0
    blue_channel[real_predictions==8] = 255
    # (pelvis - grey)
    red_channel[real_predictions==9] = 151
    green_channel[real_predictions==9] = 151
    blue_channel[real_predictions==9] = 147

    predictions_rgb = np.stack((red_channel, green_channel, blue_channel), axis=0)

    return predictions_rgb

def reformat_label(label):
    new_label = colors_to_classes(label)
    return new_label


#esto no se si va aca o como definicion
train_transform = A.Compose(
[
    A.Resize(width=128, height=128), # default INTER_LINEAR interpolation
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    A.ToFloat(),
    ToTensorV2(),
],
#additional_targets={ "other_image" : "image"},

)
test_transform = A.Compose(
    [
        A.Resize(width=128, height=128), # default INTER_LINEAR interpolation
        ToTensorV2(),
    ],
)



class SimecoDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.base_dir = base_dir
        self.list_files = os.listdir(os.path.join(self.base_dir, 'images'))
        self.transform = transform

    def __len__(self):
        return len(self.list_files)
    
    def load_image_PIL(self, filename):
        '''
        Read an image file and return as PIL format 
        '''
        image = Image.open(filename)
        # Convert PIL image to numpy array
        image = np.array(image)

        return image
    
    def load_image(self, filename):
        '''
        Read an image file with OpenCV for Albumentation 
        '''
        image = cv2.imread(filename)
        # By default OpenCV uses BGR color space, we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image

    def __getitem__(self, index):
        filename = self.list_files[index]
        # Load images
        simeco = self.load_image(path.join(path.join(self.base_dir, 'images'), filename))  
        label = self.load_image(path.join(path.join(self.base_dir, 'labels'), filename))
        label = reformat_label(label)
                
        # transform
        if self.transform is not None:
            transformed = self.transform(image=simeco, mask=label)
            simeco = transformed['image']  # output
            label = transformed['mask'] #input
              
        return simeco, label

def visualize_augmentations(dataset, samples=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    figure, ax = plt.subplots(nrows=samples, ncols=2, figsize=(10, 24))
    for idx in range(samples):
        image, mask = dataset[idx]
        ax[idx, 0].imshow(image)
        ax[idx, 1].imshow(mask, interpolation="nearest")
        ax[idx, 0].set_title("Augmented image")
        ax[idx, 1].set_title("Augmented mask")
        ax[idx, 0].set_axis_off()
        ax[idx, 1].set_axis_off()
    plt.tight_layout()
    plt.show()



class UpSampleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=4, strides=2, padding=1, activation=True, batchnorm=True, dropout=False):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.ReLU(True)

        if dropout:
            self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.deconv(x)
        if self.batchnorm:
            x = self.bn(x)

        if self.dropout:
            x = self.drop(x)
        return x
    
class DownSampleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=4, strides=2, padding=1, activation=True, batchnorm=True):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x
    

class UNetGenerator(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Paper details:
        - Encoder: C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        - Decoder: CD512-CD1024-CD1024-C1024-C1024-C512 -C256-C128
        """
        super().__init__()

        # encoder/donwsample convs
        self.encoders = [
            DownSampleConv(in_channels, 64, batchnorm=False),  # bs x 64 x 128 x 128
            DownSampleConv(64, 128),  # bs x 128 x 64 x 64
            DownSampleConv(128, 256),  # bs x 256 x 32 x 32
            DownSampleConv(256, 512),  # bs x 512 x 16 x 16
            DownSampleConv(512, 512),  # bs x 512 x 8 x 8
            DownSampleConv(512, 512),  # bs x 512 x 4 x 4
            DownSampleConv(512, 512),  # bs x 512 x 2 x 2
            DownSampleConv(512, 512, batchnorm=False),  # bs x 512 x 1 x 1
        ]

        # decoder/upsample convs
        self.decoders = [
            UpSampleConv(512, 512, dropout=True),  # bs x 512 x 2 x 2
            UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 4 x 4
            UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 8 x 8
            UpSampleConv(1024, 512),  # bs x 512 x 16 x 16
            UpSampleConv(1024, 256),  # bs x 256 x 32 x 32
            UpSampleConv(512, 128),  # bs x 128 x 64 x 64
            UpSampleConv(256, 64),  # bs x 64 x 128 x 128
        ]
        self.decoder_channels = [512, 512, 512, 512, 256, 128, 64]
        self.final_conv = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x):
        skips_cons = []
        for encoder in self.encoders:
            x = encoder(x)

            skips_cons.append(x)

        skips_cons = list(reversed(skips_cons[:-1]))
        decoders = self.decoders[:-1]

        for decoder, skip in zip(decoders, skips_cons):
            x = decoder(x)
            # print(x.shape, skip.shape)
            x = torch.cat((x, skip), axis=1)

        x = self.decoders[-1](x)
        # print(x.shape)
        x = self.final_conv(x)
        return self.tanh(x)
    
#define UpsampleConvolLayer, replace ConvTranspose2d
class UpsampleConvolLayer(torch.nn.Module):
          def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
              super(UpsampleConvolLayer, self).__init__()
              self.upsample = upsample
              if upsample:
                  self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
              reflection_padding = kernel_size // 2
              self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
              self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

          def forward(self, x):
              x_in = x
              if self.upsample:
                  x_in = self.upsample_layer(x_in)
              out = self.reflection_pad(x_in)
              out = self.conv2d(out)
              return out

class Unet_SkipConnectionBlock(nn.Module):
    """ submodulo Unet con skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construccion de submodulo Unet con skip connections.
        Parameters:
            outer_nc (int) -- num de filtros en la capa mas externa (outer) conv_layer
            inner_nc (int) -- num de filtros en la capa mas interna (inner) conv_layer
            input_nc (int) -- num de canales en la imagen de entrada
            submodule -- submodulo previo
            outermost (bool)    -- Si es el modulo de afuera (ultimo)
            innermost (bool)    -- si es el modulo de adentro (primero)
            norm_layer          -- capa de normalización
            use_dropout (bool)  -- Si usamos dropout
        """
        super(Unet_SkipConnectionBlock, self).__init__()
        self.outermost = outermost # Para saber si el submodulo es el externo para no agregar la skip conection en el forward
        
        use_bias = norm_layer == nn.InstanceNorm2d # para agregar el bias a la conv dependiendo si usamos Batch o instance Norm
        if input_nc is None:
            input_nc = outer_nc

        # estructura del submodulo
        
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = UpsampleConvolLayer(inner_nc * 2, outer_nc, kernel_size=3, stride=1, upsample=2) 
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = UpsampleConvolLayer(inner_nc, outer_nc, kernel_size=3, stride=1, upsample=2) 
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = UpsampleConvolLayer(inner_nc * 2, outer_nc, kernel_size=3, stride=1, upsample=2) 
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class UnetGenerator_v2(nn.Module):
    """ U-net backbone -  Se construye desde el centro hacia afuera en forma de "cebolla" 
                                          (down - submodulo - up) """
    
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator_v2, self).__init__()
        # Primero se construye cuello de botella. 
        unet_block = Unet_SkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        # Las capas mas profundas dependiendo el tamaño de la imagen
        for i in range(num_downs - 5):
            unet_block = Unet_SkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # las primeras capas que aumentan el num de filtros.
        unet_block = Unet_SkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = Unet_SkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = Unet_SkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = Unet_SkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

class PatchGANDiscriminator(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.d1 = DownSampleConv(input_channels, 64, batchnorm=False)
        self.d2 = DownSampleConv(64, 128)
        self.d3 = DownSampleConv(128, 256)
        self.d4 = DownSampleConv(256, 512)
        self.final = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.d1(x)
        x1 = self.d2(x0)
        x2 = self.d3(x1)
        x3 = self.d4(x2)
        xn = self.final(x3)
        return xn
    
# https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

        
def display_progress(cond, real, fake, current_epoch, path, figsize=(10,5)):
    """
    Save cond, real (original) and generated (fake)
    images in one panel 
    """
    cond = (cond.detach().cpu().permute(1, 2, 0)+1)/2.0  
    real = (real.detach().cpu().permute(1, 2, 0)+1)/2.0 
    fake = (fake.detach().cpu().permute(1, 2, 0)+1)/2.0 
    images = [cond, real, fake]
    titles = ['input','real','generated']
    print(f'Epoch: {current_epoch}')
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    for idx,img in enumerate(images):
        ax[idx].imshow(img)
        ax[idx].axis("off")
    for idx, title in enumerate(titles):    
        ax[idx].set_title('{}'.format(title))
    plt.savefig(path)
    plt.show()

def save_image(npimg, filename):
    image = Image.fromarray(npimg)
    image.save(filename)    

def load_image(filename):
    '''
    Read an image file with OpenCV for Albumentation 
    '''
    image = cv2.imread(filename)
    # By default OpenCV uses BGR color space, we need to convert the image to RGB color space.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image


class Pix2Pix(pl.LightningModule):
    
    def __init__(self, in_channels, out_channels, learning_rate=0.0002, lambda_recon=100, display_step=10):

        super().__init__()
        self.save_hyperparameters()
        
        self.display_step = display_step
        #self.gen = UNetGenerator(in_channels, out_channels)
        self.gen = UnetGenerator_v2(in_channels, out_channels, 7)
        self.disc = PatchGANDiscriminator(in_channels + out_channels)

        # intializing weights
        self.gen = self.gen.apply(_weights_init)
        self.disc = self.disc.apply(_weights_init)

        self.adversarial_criterion = cfg['adversarial_criterion']   #nn.BCEWithLogitsLoss()
        self.recon_criterion = cfg['recon_criterion']               #nn.L1Loss()

        self.automatic_optimization = False
        
       
    def _gen_step(self, real_images, conditioned_images):
        # Pix2Pix has adversarial and a reconstruction loss
        # adversarial loss
        fake_images = self.gen(conditioned_images)
        disc_logits = self.disc(fake_images, conditioned_images)
        adversarial_loss = self.adversarial_criterion(disc_logits, torch.ones_like(disc_logits))

        # reconstruction loss
        recon_loss = self.recon_criterion(fake_images, real_images)
        lambda_recon = self.hparams.lambda_recon
        
        self.log_dict({"adv_loss":adversarial_loss,"recon_loss": recon_loss})
        
        g_loss = adversarial_loss + lambda_recon * recon_loss
        self.log('Generator Loss', g_loss)
        
        return g_loss

    def _disc_step(self, real_images, conditioned_images):
        fake_images = self.gen(conditioned_images).detach()
        fake_logits = self.disc(fake_images, conditioned_images)

        real_logits = self.disc(real_images, conditioned_images)

        fake_loss = self.adversarial_criterion(fake_logits, torch.zeros_like(fake_logits))
        real_loss = self.adversarial_criterion(real_logits, torch.ones_like(real_logits))
                
        return (real_loss + fake_loss) / 2

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        disc_opt = torch.optim.Adam(self.disc.parameters(), lr=lr)
        return disc_opt, gen_opt

    
    def training_step(self, batch, batch_idx):
        real, condition = batch
        condition = torch.unsqueeze(condition.float(),1) # convierto los labels a float y el canal de profundida = 1 
        # Manual optimization 
        # To access your optimizers
        d_opt, g_opt = self.optimizers() 

        # Discriminator
        d_loss = self._disc_step(real, condition) # To compute the loss
        d_opt.zero_grad() # To clear the gradients from the previous training step
        self.manual_backward(d_loss) # Instead of loss.backward() 
        d_opt.step() # To update your model parameters
        self.log('PatchGAN Loss', d_loss)
        
        
        # Generator
        g_loss = self._gen_step(real, condition)
        g_opt.zero_grad()
        self.manual_backward(g_loss)
        g_opt.step()
        

        if self.current_epoch%self.display_step==0 and batch_idx==0:
            fake = self.gen(condition).detach()
            display_progress(condition[0], real[0], fake[0], self.current_epoch, 
                             path="./working/img_{}".format(self.current_epoch))

    def scale_image(self,image, width=256, height=256):
    
        scaledImage = cv2.resize(image,(width,height), interpolation=cv2.INTER_AREA)
        
        return scaledImage 


    indice = 0



    
    def short_valid_step(self,input_path,output_path):
        
       
        # Genero las imagenes simuladas 
        self.gen.eval() # modelo en modo eval
        filenames = listdir(input_path)
        label = load_image(path.join(input_path, filenames[self.indice]))
        self.indice+=1
        if self.indice>=len(filenames):
            self.indice=0
        mask = reformat_label(label)
        # TODO: Ver porque poronga tengo que hacer la transformación de una imagen (label-RGB) si o si y no puedo mandar solo la mask
        transformed = test_transform(image=label, mask=mask)
        mask = transformed['mask'] #input
        mask = torch.unsqueeze(torch.unsqueeze(mask.float(),0),0)# convierto los labels a float y el canal de profundida = 1 
        fake_image = self.gen(mask)
        # Save image
        fake_image = fake_image[0,:,:,:].detach().cpu().numpy()
        fake_image = (((np.transpose(fake_image, (1, 2, 0)) + 1) / 2.0)*255).astype(np.uint8) 
        return fake_image 


    def short_valid_step_256(self,input_path,output_path):
        self.indice=0
        input_path= "./results/imagenesTomadasDeVTK/"
        # Genero las imagenes simuladas 
        self.gen.eval() # modelo en modo eval
        filenames = listdir(input_path)
        label = load_image(path.join(input_path, filenames[self.indice]))
        self.indice+=1
        if self.indice>=len(filenames):
            self.indice=0
        
        #label = self.scale_image(label) #cambio de 128 a 256 la imagen antes de generada
        
        mask = reformat_label(label)
        

        # TODO: Ver porque poronga tengo que hacer la transformación de una imagen (label-RGB) si o si y no puedo mandar solo la mask
        transformed = test_transform(image=label, mask=mask)
        #aca el transforme me rompe todo y me la vuelve a bajar a 128
        
        mask = transformed['mask'] #input
        
        mask = torch.unsqueeze(torch.unsqueeze(mask.float(),0),0)# convierto los labels a float y el canal de profundida = 1 

        #print(mask.shape)
        
        fake_image = self.gen(mask)
        # Save image
 
        fake_image = fake_image[0,:,:,:].detach().cpu().numpy()
        
        
        
        fake_image = (((np.transpose(fake_image, (1, 2, 0)) + 1) / 2.0)*255).astype(np.uint8) 
        #print(fake_image.shape)
        fake_image = self.scale_image(fake_image)
        #print(fake_image.shape)

        #fake_image = remap.acomodarFOV(img=fake_image)
        
        

        return fake_image 


    def valid_step256_fromImage(self,img):
        
        input_path = "./data/validation/labels"
        filenames = listdir(input_path)
        output_path = "./results/imagenesTomadasDeVTK/"
        label2 = load_image(path.join(input_path, filenames[0]))


        # Genero las imagenes simuladas 
        self.gen.eval() # modelo en modo eval
        label = img
        self.indice+=1
        
        label = self.scale_image(label) #cambio de 128 a 256 la imagen despues de generada
        same_shape = label2.shape == label.shape
        same_dtype = label2.dtype == label.dtype

        if same_shape and same_dtype:
            print("Las imágenes tienen el mismo formato.")
        else:
            print("Las imágenes NO tienen el mismo formato.")
        mask = reformat_label(label)
        save_image(img, path.join(output_path,filenames[0] + '.png')) 

        # TODO: Ver porque poronga tengo que hacer la transformación de una imagen (label-RGB) si o si y no puedo mandar solo la mask
        transformed = test_transform(image=label, mask=mask)
        #aca el transforme me rompe todo y me la vuelve a bajar a 128
        
        mask = transformed['mask'] #input
        
        mask = torch.unsqueeze(torch.unsqueeze(mask.float(),0),0)# convierto los labels a float y el canal de profundida = 1 

        #print(mask.shape)
        
        fake_image = self.gen(mask)
        # Save image
 
        fake_image = fake_image[0,:,:,:].detach().cpu().numpy()
        
        
        
        fake_image = (((np.transpose(fake_image, (1, 2, 0)) + 1) / 2.0)*255).astype(np.uint8) 
        #print(fake_image.shape)
        fake_image = self.scale_image(fake_image)
        #print(fake_image.shape)

        #fake_image = remap.acomodarFOV(img=fake_image)
        
        

        return fake_image 

    
    def validation_step(self,input_path, output_path):
        makedirs(output_path, exist_ok=True)
        # Genero las imagenes simuladas 
        self.gen.eval() # modelo en modo eval
        filenames = listdir(input_path)
        for filename in filenames:
            label = load_image(path.join(input_path, filename))
            mask = reformat_label(label)
            # TODO: Ver porque poronga tengo que hacer la transformación de una imagen (label-RGB) si o si y no puedo mandar solo la mask
            transformed = test_transform(image=label, mask=mask)
            mask = transformed['mask'] #input
            mask = torch.unsqueeze(torch.unsqueeze(mask.float(),0),0)# convierto los labels a float y el canal de profundida = 1 
            fake_image = self.gen(mask)
            # Save image
            fake_image = fake_image[0,:,:,:].detach().cpu().numpy()
            fake_image = (((np.transpose(fake_image, (1, 2, 0)) + 1) / 2.0)*255).astype(np.uint8) 
            fake_image = self.scale_image(fake_image)
            save_image(fake_image, path.join(output_path,filename + '.png')) 









  

def main(request):
    print(f'PyTorch version: {torch.__version__}')
    print(f'Pytorch Lightning: {pl.__version__}')
    print("Torch version:",torch.__version__)

    print("Is CUDA enabled?",torch.cuda.is_available())

    seed_everything(cfg['seed'])
    print(cfg['seed'])


    train_dataset = SimecoDataset(cfg['train_dir'], transform=train_transform)
    #visualize_augmentations(train_dataset)


    model_path = "./epoch=500-step=4008.ckpt"
    model = Pix2Pix.load_from_checkpoint(model_path)


    input_path = "./data/validation/labels"
    output_path = "./results/" 
    makedirs(output_path, exist_ok=True)

    # Genero las imagenes simuladas 
    model.eval() # modelo en modo eval


    model.short_valid_step(input_path,output_path)

    #model.validation_step(input_path,output_path)

    return model
    #return HttpResponse("<h1>Hello</h1>")




"""#if(data.type === 'connection_established'){
                chatSocket.send(JSON.stringify({
                    'message':'message'
                }))
            }"""