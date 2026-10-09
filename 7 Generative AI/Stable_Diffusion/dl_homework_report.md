# Stable Diffusion Homework Report

## Student Information

- Name: `[Your Name]`
- Course: `[Course Name / Number]`
- Assignment: Deep Learning Homework, Conditional Image Generation with PyTorch
- Date: `[Submission Date]`

## 1. Objective

The goal of this homework was to build a **conditional color image generator** using **PyTorch only**, based on the diffusion ideas from the provided Stable Diffusion code and the class notebook. The assignment required training the system **from scratch**, generating **conditional RGB images**, and avoiding external diffusion frameworks such as TensorFlow, Keras, Hugging Face, and FastAI.

To satisfy those requirements, I implemented a **latent diffusion pipeline**. The model first trains a small variational autoencoder (VAE) and then trains a **class-conditioned diffusion model in latent space**. The final generator produces color food images conditioned on class labels.

## 2. Dataset

I used the **Food-101** dataset from `torchvision`, but instead of training on all 101 classes, I selected a smaller balanced subset of **10 food classes** so the notebook would train faster while still using real food images.

- Dataset source: `torchvision.datasets.Food101`
- Image type: RGB color food photographs
- Resized image size: `64 x 64`
- Number of selected classes: 10
- Conditioning signal: numeric class label (`0` to `9`)
- Samples per class: `500`
- Total training examples: `5000`

The selected classes are:

1. `0 -> apple_pie`
2. `1 -> baby_back_ribs`
3. `2 -> hamburger`
4. `3 -> caesar_salad`
5. `4 -> cheesecake`
6. `5 -> fried_rice`
7. `6 -> ice_cream`
8. `7 -> pizza`
9. `8 -> steak`
10. `9 -> sushi`

This design keeps the assignment faithful to the original food-image idea while reducing training time significantly compared with using all 101 classes.

Implementation note: in the current `torchvision.datasets.Food101` implementation, dataset targets are returned as **integer labels**, so the notebook remaps the original Food-101 numeric labels into a new compact `0..9` label space for the selected classes.

In the saved notebook output, the dataset loader reports:

```text
training examples: 5000
sample batch shape: torch.Size([64, 3, 64, 64])
sample labels: [1, 5, 5, 9, 5, 2, 1, 0]
sample classes: ['baby_back_ribs', 'fried_rice', 'fried_rice', 'sushi', 'fried_rice', 'hamburger', 'baby_back_ribs', 'apple_pie']
```

## 3. Model Design

The final system has two stages.

### 3.1 Variational Autoencoder (VAE)

The first stage is a small VAE trained from scratch on the selected Food-101 subset. Its purpose is to compress RGB food images into a lower-dimensional latent representation before diffusion.

- Input: `3 x 64 x 64`
- Latent channels: `4`
- Latent spatial size: `16 x 16`
- Output: reconstructed RGB image

The VAE is trained with:

- reconstruction loss (`L1`)
- KL-divergence regularization

This stage is important because it allows the diffusion model to work in latent space instead of pixel space, which follows the core idea behind Stable Diffusion.

### 3.2 Conditional Latent Diffusion Model

After the VAE is trained, images are encoded into latent tensors. A conditional diffusion model is then trained to denoise those latent tensors.

The diffusion model includes:

- Gaussian Fourier time embeddings
- dense layers for time conditioning
- a U-Net style encoder-decoder structure
- spatial transformer blocks
- cross-attention using class embeddings

The condition is the **numeric food class label**, embedded with `nn.Embedding` and injected through cross-attention. This allows the model to produce different latent distributions for different food categories. The class names are only used as a readable lookup table for interpreting the numeric labels.

## 4. Adaptation from the Class Notebook

The notebook `StableDiffusion2026.ipynb` included conditional score-based diffusion components. I adapted the following ideas into the homework notebook:

- Gaussian Fourier projection for time embeddings
- dense time-conditioning layers
- cross-attention
- transformer block
- spatial transformer
- conditional score-matching loss
- Euler-Maruyama sampling

The original short homework notebook used a simpler **pixel-space DDPM**. I replaced that with a **latent-space conditional diffusion model** to better match the Stable Diffusion style while remaining fully in PyTorch.

## 5. Training Setup

The notebook uses the following main hyperparameters:

```python
IMAGE_SIZE = 64
LATENT_CHANNELS = 4
LATENT_SIZE = 16
BATCH_SIZE = 64
AUTOENCODER_EPOCHS = 15
DIFFUSION_EPOCHS = 40
LR_AUTOENCODER = 2e-4
LR_DIFFUSION = 2e-4
KL_WEIGHT = 1e-4
SIGMA = 25.0
NUM_DIFFUSION_STEPS = 300
MAX_SAMPLES_PER_CLASS = 500
```

Training is split into two phases:

1. Train the VAE on the 10-class Food-101 subset.
2. Encode the images into latent space and train the conditional diffusion model on those latents.

Sampling is done with an **Euler-Maruyama sampler**, and the generated latent tensor is decoded back into an RGB image using the VAE decoder.

The notebook also saves separate checkpoints for this run:

- VAE checkpoint: `food101_10class_500perclass_vae.pt`
- Diffusion checkpoint: `food101_10class_500perclass_latent_diffusion.pt`

## 6. Why This Satisfies the Homework Requirements

This solution satisfies the assignment constraints in the following ways:

- **PyTorch only**: all code is written using PyTorch and torchvision.
- **Conditional image generation**: generation is conditioned on numeric class labels.
- **Color images**: Food-101 images are RGB.
- **Training from scratch**: both the VAE and the diffusion model are initialized and trained from scratch.
- **No fine-tuning**: no pretrained diffusion model is used.
- **No external frameworks**: no TensorFlow, Keras, Hugging Face, FastAI, or similar libraries are used.

## 7. Results

The updated notebook was trained on the 10-class Food-101 subset with 500 images per class, for a total of 5,000 training examples. Both the VAE and the latent diffusion model were trained successfully from scratch.

### 7.1 VAE Training Results

The VAE loss decreased steadily across 15 epochs:

- Epoch 1 loss: `0.2370`
- Epoch 5 loss: `0.0921`
- Epoch 10 loss: `0.0745`
- Epoch 15 loss: `0.0694`

Final VAE metrics from the saved notebook:

- Final total VAE loss: `0.0694`
- Final reconstruction loss: `0.0689`
- Final KL loss: `4.2140`

This indicates that the autoencoder learned to reconstruct the food images reasonably well while maintaining a structured latent space.

### 7.2 Latent Diffusion Training Results

The latent diffusion training loss also decreased substantially over 40 epochs:

- Epoch 1 loss: `596.7848`
- Epoch 10 loss: `235.4779`
- Epoch 20 loss: `214.6748`
- Epoch 30 loss: `206.4949`
- Epoch 39 loss: `204.1971`
- Epoch 40 loss: `207.2346`

The final diffusion loss is much lower than the starting loss, which shows that the model learned a significantly better denoising function over time.

The notebook is currently set to generate final conditional samples for:

- label `0` -> `apple_pie`
- label `2` -> `hamburger`
- label `7` -> `pizza`
- label `9` -> `sushi`

These four classes are good examples because they are visually distinct and easy to compare qualitatively.

## 7.3 Screenshot Placement

Add screenshots in the report at the following spots.

### Screenshot A: Food-101 Training Samples

Insert this directly after **Section 2. Dataset**.

What to capture:

- the figure titled `Food-101 10-class training samples`

Suggested caption:

`Figure 1. Example RGB food images from the 10-class Food-101 training subset.`

### Screenshot B: VAE Reconstructions

Insert this directly after **Section 3.1 Variational Autoencoder (VAE)**.

What to capture:

- the figure titled `Original food images`
- the figure titled `VAE food reconstructions`

Suggested caption:

`Figure 2. Original food images and their reconstructions produced by the trained VAE.`

### Screenshot C: Conditional Diffusion Samples

Insert this directly after **Section 7. Results**.

What to capture:

- the four output figures titled:
  - `Conditional food samples for label 0: apple_pie`
  - `Conditional food samples for label 2: hamburger`
  - `Conditional food samples for label 7: pizza`
  - `Conditional food samples for label 9: sushi`

Suggested caption:

`Figure 3. Class-conditioned latent diffusion samples for apple pie, hamburger, pizza, and sushi.`

### Optional Screenshot D: Diffusion Training Log

Insert this near the end of **Section 5. Training Setup** or inside **Section 7.2 Latent Diffusion Training Results**.

What to capture:

- the printed diffusion training log showing loss by epoch

Suggested caption:

`Figure 4. Latent diffusion training log for the 10-class Food-101 subset.`

## 8. Discussion

Using a reduced Food-101 subset creates a tradeoff between **faster training** and **sample quality**.

Advantages of this setup:

- it keeps real food photographs instead of switching to simpler object datasets
- it still uses class conditioning
- it is much faster than training on all 101 food classes
- it stays closer to the original homework idea of food-image generation

Limitations of this setup:

- only 500 images per class are used
- the subset is small for training a diffusion model from scratch
- image quality may be noisy or blurry
- some classes may overlap visually, such as `hamburger` versus `pizza` backgrounds or plating

Even with those limitations, this setup is a better compromise if the goal is to preserve food-image generation while reducing runtime.

## 9. Limitations and Future Improvements

Possible improvements include:

1. Increasing the number of images per food class.
2. Expanding from 10 food classes to more categories.
3. Training for more epochs.
4. Using a stronger autoencoder architecture.
5. Adding classifier-free guidance for stronger conditional control.

If more compute time were available, the first improvements I would try would be:

- more images per class
- more training epochs
- sampling more than four final food categories for evaluation

## 10. Conclusion

In this homework, I built a **class-conditioned latent diffusion model** in **pure PyTorch** for generating **color food images**. The model uses a **10-class Food-101 subset**, a VAE trained from scratch, and a conditional diffusion model operating in latent space.

The final notebook used `500` images per selected class for a total of `5000` training examples. The VAE loss decreased to `0.0694`, and the latent diffusion loss decreased from `596.7848` at epoch 1 to about `207.2346` at epoch 40. This shows that the full two-stage pipeline trained successfully and produced class-conditioned food-image generation with numeric label conditioning.

## 11. Files Submitted

- `dl_homework.ipynb`
- `dl_homework_report.md`
