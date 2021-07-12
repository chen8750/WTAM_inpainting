# Generative Image Inpainting Based on Wavelet Transform Attention Model

[ISCAS 2020 Paper](https://doi.org/10.1109/ISCAS45731.2020.9180927) | [BibTex](#citing)

**Update (Jun, 2021)**:
1. The tech report of our new image inpainting system MuFA-Net is released, please checkout branch [v2.0.0](https://github.com/ChenWang8750/MuFA-Net) 
2. WTAM is trained and mainly works on rectangular masks, while MuFA-Net can generate high quality inpainting results for variable masks.

# Example inpainting results
<table style="float:center">
 <tr>
  <th><B>Input</B></th> <th><B> Ours(U-net) </B></th> <th><B> Ours(UNet++)</B> <th><B>Ground-truth</B></th>
 </tr>
<tr>
  <td>
   <img src='./imgs/11_real_A.png' >
  </td>
  <td>
  <img src='./imgs/11_fake_B_unet.png'>
  </td>
  <td>
  <img src='./imgs/11_fake_B_unetplus.png'>
  </td>
  <td>
   <img src='./imgs/11_real_B.png'>
  </td>
 
 </tr>
 
 <tr>
     <td>
   <img src='./imgs/159_real_A.png' >
  </td>
  <td>
  <img src='./imgs/159_fake_B_unet.png'>
  </td>
  <td>
  <img src='./imgs/159_fake_B_unetplus.png'>
  </td>
  <td>
   <img src='./imgs/159_real_B.png'>
  </td>

 </tr>
 
  </table> 

# Architecutre
<img src="https://github.com/ChenWang8750/WTAM/blob/master/imgs/architecutre.png" width="1000"/> 

# Wavelet transform attention model
<img src="https://github.com/ChenWang8750/WTAM/blob/master/imgs/wavelet_attention.png" width="800"/> 


## Run

0. Requirements:
    * Install python3.
    * Install [PyTorch] (http://pytorch.org/) (tested on Release>=0.4.0).
    * Install python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate).
1. Training:
    * Prepare training image datasets.
    * Modify [base_options.py](https://github.com/ChenWang8750/WTAM/blob/master/options/base_options.py) to set parameters.
    * Run `python train.py`.
2. Testing:
    * Prepare testing image datasets.
    * Modify [test_options.py](https://github.com/ChenWang8750/WTAM/blob/master/options/test_options.py) to set parameters.
    * Run `python test.py`.


## Pretrained models

[Paris StreetView](https://drive.google.com/drive/folders/) | [CelebA-HQ](https://drive.google.com/drive/folders/)

Rename `face_center_mask.pth` to `30_net_G.pth`, and put it in the folder `./log/face_center_mask`(if not existed, create it)

```bash
# CelebAMask-HQ 256x256 input
python test.py --which_model_netG='WTAM' --model='WTAM' --name='face_center_mask' --which_epoch=30 --dataroot='./datasets/test' `.
```

**Note:** For models trained with extra irregular masks, make sure `--offline_loading_mask=1 --testing_mask_folder='masks' `.

## Visdom

To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097. The checkpoints will be saved in `./log` by default.


## Citing
```
@inproceedings{wang2020generative,
  title={Generative Image Inpainting Based on Wavelet Transform Attention Model},
  author={Wang, Chen and Wang, Jin and Zhu, Qing and Yin, Baocai},
  booktitle={ISCAS},
  pages={1--5},
  year={2020},
  organization={IEEE}
}

```

## Acknowledgments
We benefit a lot from [Shift-Net_pytorch](https://github.com/Zhaoyi-Yan/Shift-Net_pytorch)
