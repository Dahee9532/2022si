from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
import os


if __name__ == '__main__':
    parser = get_arguments()
    #add_argument : 입력받을 인자값 등록하는 것
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True) #required=True <- 필수지정옵션
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args() # <- 입력받을 인자값은 opt(namespace)에 저장
    #출력은, opt.input_dir 이렇게 
    #namespace format : Namespace(Dsteps=3, Gsteps=3, alpha=10, beta1=0.5, gamma=0.1, input_dir='Input/Images', input_name='ad', ker_size=3, lambda_grad=0.1, lr_d=0.0005, lr_g=0.0005, manualSeed=None, max_size=250, min_nfc=32, min_size=25, mode='train', nc_im=3, nc_z=3, netD='', netG='', nfc=32, niter=2000, noise_amp=0.1, not_cuda=0, num_layer=5, out='Output', padd_size=0, scale_factor=0.75, stride=1)
    opt = functions.post_config(opt) #opt setting
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)


    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        functions.adjust_scales2image(real, opt)
        train(opt, Gs, Zs, reals, NoiseAmp) #training.py
        SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt) #manipulate.py
