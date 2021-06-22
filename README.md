# π-GAN


https://user-images.githubusercontent.com/9628319/122865841-e2d1c080-d2db-11eb-9621-1e176db59352.mp4

## Training a Model

The main training script can be found in train.py.

##### Relevant Flags:

Set the output directory:
`--output_dir=[output directory]`

Set the model loading directory:
`--load_dir=[load directory]`

Set the current training curriculum:
`--curriculum=[curriculum]`

Set the port for distributed training:
`--port=[port]`


##### To start training:

On one GPU:
`CUDA_VISIBLE_DEVICES=0 python3 train.py --flag=value --flag=value ...`

On multiple GPUs, simply list cuda visible devices in a comma-separated list:
`CUDA_VISIBLE_DEVICES=1,3 python3 train.py --flag=value --flag=value ...`

To continue training from another runspecify the `--load_dir=path/to/directory` flag. 

## Model Results and Evaluation

#### Evaluation Metrics
`python eval_metrics.py path/to/generator.pth --real_image_dir path/to/real_images/directory --num_images 8000`

#### Rendering Images
`python render_multiview_images.py path/to/generator.pth --seeds 0 1 2 3`

For best visual results, load the EMA parameters, use truncation, increase the resolution (e.g. to 512 x 512) and increase the number of depth samples (e.g. to 24 or 36).

#### Rendering Videos
`python render_video.py path/to/generator.pth --seeds 0 1 2 3`

You can pass the flag `--lock_view_dependence` to remove view dependent effects. This can help mitigate distracting visual artifacts such as shifting eyebrows. However, locking view dependence may lower the visual quality of images (edges may be blurrier etc.)

#### Rendering Videos Interpolating between faces
`python render_video_interpolation.py path/to/generator.pth --seeds 0 1 2 3`

#### Extracting 3D Shapes

`python3 shape_extraction.py path/to/generator.pth --seed 0`

## Pretrained Models
Here are pretrained models for CelebA, Cats, and CARLA models

CelebA: 
Cats: https://drive.google.com/file/d/1WBA-WI8DA7FqXn7__0TdBO0eO08C_EhG/view?usp=sharing
CARLA: https://drive.google.com/file/d/1n4eXijbSD48oJVAbAV4hgdcTbT3Yv4xO/view?usp=sharing

All zipped model files contain a generator.pth, ema.pth, and ema2.pth files. ema.pth used a decay of 0.999 and ema2.pth used a decay of 0.9999.

#### Changes/additions since original implementation

1. Added experimental pose identity loss. Controlled by pos_lambda in the curriculum, helps ensure generated scenes share the same canonical pose. Empirically, it seems to improve 3D models, but may introduce a minor decrease in image quality scores.

2. Added script for latent code interpolation in W-space.

3. Added options for truncation, following implementation in StyleGAN.

4. Tweaks to hyperparmeters, e.g. learning rate and initialization. Should result in improved evaluation metrics.


### Training Tips

If you have the resources, increasing the number of samples (steps) per ray will dramatically increase the quality of your 3D shapes. If you're looking for good shapes, e.g. for CelebA, try increasing num_steps and moving the back plane (ray_end) to allow the model to move the background back and capture the full head.

Training has been tested to work well on either two RTX 6000's or one RTX 8000. Training with smaller GPU's and batch sizes generally works fine, but it's also possible you'll encounter instability, especially at higher resolutions. Bubbles and artifacts that suddenly appear, or blurring in the tilted angles, are signs that training destabilized. This can usually be mitigated by training with a larger batch size or by reducing the learning rate.
