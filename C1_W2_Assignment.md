# Deep Convolutional GAN (DCGAN)

### Goal
In this notebook, you're going to create another GAN using the MNIST dataset. You will implement a Deep Convolutional GAN (DCGAN), a very successful and influential GAN model developed in 2015.

*Note: [here](https://arxiv.org/pdf/1511.06434v1.pdf) is the paper if you are interested! It might look dense now, but soon you'll be able to understand many parts of it :)*

### Learning Objectives
1.   Get hands-on experience making a widely used GAN: Deep Convolutional GAN (DCGAN).
2.   Train a powerful generative model.


![Generator architecture](dcgan-gen.png)

Figure: Architectural drawing of a generator from DCGAN from [Radford et al (2016)](https://arxiv.org/pdf/1511.06434v1.pdf).

## Getting Started

#### DCGAN
Here are the main features of DCGAN (don't worry about memorizing these, you will be guided through the implementation!): 

<!-- ```
Architecture guidelines for stable Deep Convolutional GANs
• Replace any pooling layers with strided convolutions (discriminator) and fractional-strided
convolutions (generator).
• Use BatchNorm in both the generator and the discriminator.
• Remove fully connected hidden layers for deeper architectures.
• Use ReLU activation in generator for all layers except for the output, which uses Tanh.
• Use LeakyReLU activation in the discriminator for all layers.
``` -->


*   Use convolutions without any pooling layers
*   Use batchnorm in both the generator and the discriminator
*   Don't use fully connected hidden layers
*   Use ReLU activation in the generator for all layers except for the output, which uses a Tanh activation.
*   Use LeakyReLU activation in the discriminator for all layers except for the output, which does not use an activation

You will begin by importing some useful packages and data that will help you create your GAN. You are also provided a visualizer function to help see the images your GAN will create.


```python
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0) # Set for testing purposes, please do not change!


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
```

## Generator
The first component you will make is the generator. You may notice that instead of passing in the image dimension, you will pass the number of image channels to the generator. This is because with DCGAN, you use convolutions which don’t depend on the number of pixels on an image. However, the number of channels is important to determine the size of the filters.

You will build a generator using 4 layers (3 hidden layers + 1 output layer). As before, you will need to write a function to create a single block for the generator's neural network.
<!-- From the paper, we know to "[u]se batchnorm in both the generator and the discriminator" and "[u]se ReLU activation in generator for all layers except for the output, which uses Tanh." --> 
Since in DCGAN the activation function will be different for the output layer, you will need to check what layer is being created. You are supplied with some tests following the code cell so you can see if you're on the right track!

At the end of the generator class, you are given a forward pass function that takes in a noise vector and generates an image of the output dimension using your neural network. You are also given a function to create a noise vector. These functions are the same as the ones from the last assignment.

<details>
<summary>
<font size="3" color="green">
<b>Optional hint for <code><font size="4">make_gen_block</font></code></b>
</font>
</summary>

1. You'll find [nn.ConvTranspose2d](https://pytorch.org/docs/master/generated/torch.nn.ConvTranspose2d.html) and [nn.BatchNorm2d](https://pytorch.org/docs/master/generated/torch.nn.BatchNorm2d.html) useful!
</details>


```python
# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: Generator
class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN, 
        corresponding to a transposed convolution, a batchnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''

        #     Steps:
        #       1) Do a transposed convolution using the given parameters.
        #       2) Do a batchnorm, except for the last layer.
        #       3) Follow each batchnorm with a ReLU activation.
        #       4) If its the final layer, use a Tanh activation after the deconvolution.

        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                #### START CODE HERE ####
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU()
                #### END CODE HERE ####
            )
        else: # Final Layer
            return nn.Sequential(
                #### START CODE HERE ####
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh()
                #### END CODE HERE ####
            )

    def unsqueeze_noise(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns a copy of that noise with width and height = 1 and channels = z_dim.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        x = self.unsqueeze_noise(noise)
        return self.gen(x)

def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    return torch.randn(n_samples, z_dim, device=device)
```


```python
# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
'''
Test your make_gen_block() function
'''
gen = Generator()
num_test = 100

# Test the hidden block
test_hidden_noise = get_noise(num_test, gen.z_dim)
test_hidden_block = gen.make_gen_block(10, 20, kernel_size=4, stride=1)
test_uns_noise = gen.unsqueeze_noise(test_hidden_noise)
hidden_output = test_hidden_block(test_uns_noise)

# Check that it works with other strides
test_hidden_block_stride = gen.make_gen_block(20, 20, kernel_size=4, stride=2)

test_final_noise = get_noise(num_test, gen.z_dim) * 20
test_final_block = gen.make_gen_block(10, 20, final_layer=True)
test_final_uns_noise = gen.unsqueeze_noise(test_final_noise)
final_output = test_final_block(test_final_uns_noise)

# Test the whole thing:
test_gen_noise = get_noise(num_test, gen.z_dim)
test_uns_gen_noise = gen.unsqueeze_noise(test_gen_noise)
gen_output = gen(test_uns_gen_noise)
```

Here's the test for your generator block:


```python
# UNIT TESTS
assert tuple(hidden_output.shape) == (num_test, 20, 4, 4)
assert hidden_output.max() > 1
assert hidden_output.min() == 0
assert hidden_output.std() > 0.2
assert hidden_output.std() < 1
assert hidden_output.std() > 0.5

assert tuple(test_hidden_block_stride(hidden_output).shape) == (num_test, 20, 10, 10)

assert final_output.max().item() == 1
assert final_output.min().item() == -1

assert tuple(gen_output.shape) == (num_test, 1, 28, 28)
assert gen_output.std() > 0.5
assert gen_output.std() < 0.8
print("Success!")
```

    Success!


## Discriminator
The second component you need to create is the discriminator.

You will use 3 layers in your discriminator's neural network. Like with the generator, you will need create the function to create a single neural network block for the discriminator.
<!-- From the paper, we know that we need to "[u]se LeakyReLU activation in the discriminator for all layers." And for the LeakyReLUs, "the slope of the leak was set to 0.2" in DCGAN. -->
There are also tests at the end for you to use.
<details>
<summary>
<font size="3" color="green">
<b>Optional hint for <code><font size="4">make_disc_block</font></code></b>
</font>
</summary>

1. You'll find [nn.Conv2d](https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html), [nn.BatchNorm2d](https://pytorch.org/docs/master/generated/torch.nn.BatchNorm2d.html), and [nn.LeakyReLU](https://pytorch.org/docs/master/generated/torch.nn.LeakyReLU.html) useful!
</details>


```python
# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: Discriminator
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
    hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_chan=1, hidden_dim=16):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a discriminator block of DCGAN, 
        corresponding to a convolution, a batchnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        #     Steps:
        #       1) Add a convolutional layer using the given parameters.
        #       2) Do a batchnorm, except for the last layer.
        #       3) Follow each batchnorm with a LeakyReLU activation with slope 0.2.
        
        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                #### START CODE HERE #### #
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(negative_slope=0.2)
                
                #### END CODE HERE ####
            )
        else: # Final Layer
            return nn.Sequential(
                #### START CODE HERE #### #
                nn.Conv2d(input_channels, output_channels, kernel_size, stride)
                #### END CODE HERE ####
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        '''
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)
```


```python
# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
'''
Test your make_disc_block() function
'''
num_test = 100

gen = Generator()
disc = Discriminator()
test_images = gen(get_noise(num_test, gen.z_dim))

# Test the hidden block
test_hidden_block = disc.make_disc_block(1, 5, kernel_size=6, stride=3)
hidden_output = test_hidden_block(test_images)

# Test the final block
test_final_block = disc.make_disc_block(1, 10, kernel_size=2, stride=5, final_layer=True)
final_output = test_final_block(test_images)

# Test the whole thing:
disc_output = disc(test_images)
```

Here's a test for your discriminator block:


```python
# Test the hidden block
assert tuple(hidden_output.shape) == (num_test, 5, 8, 8)
# Because of the LeakyReLU slope
assert -hidden_output.min() / hidden_output.max() > 0.15
assert -hidden_output.min() / hidden_output.max() < 0.25
assert hidden_output.std() > 0.5
assert hidden_output.std() < 1

# Test the final block

assert tuple(final_output.shape) == (num_test, 10, 6, 6)
assert final_output.max() > 1.0
assert final_output.min() < -1.0
assert final_output.std() > 0.3
assert final_output.std() < 0.6

# Test the whole thing:

assert tuple(disc_output.shape) == (num_test, 1)
assert disc_output.std() > 0.25
assert disc_output.std() < 0.5
print("Success!")
```

    Success!


## Training
Now you can put it all together!
Remember that these are your parameters:
  *   criterion: the loss function
  *   n_epochs: the number of times you iterate through the entire dataset when training
  *   z_dim: the dimension of the noise vector
  *   display_step: how often to display/visualize the images
  *   batch_size: the number of images per forward/backward pass
  *   lr: the learning rate
  *   beta_1, beta_2: the momentum term
  *   device: the device type

<!-- In addition, be warned that **this runs very slowly on the default CPU**. One way to run this more quickly is to download the .ipynb and upload it to Google Drive, then open it with Google Colab, click on `Runtime -> Change runtime type` and set hardware accelerator to GPU and replace
`device = "cpu"`
with
`device = "cuda"`. The code should then run without any more changes, over 1,000 times faster.  -->



```python
criterion = nn.BCEWithLogitsLoss()
z_dim = 64
display_step = 500
batch_size = 128
# A learning rate of 0.0002 works well on DCGAN
lr = 0.0002

# These parameters control the optimizer's momentum, which you can read more about here:
# https://distill.pub/2017/momentum/ but you don’t need to worry about it for this course!
beta_1 = 0.5 
beta_2 = 0.999
device = 'cuda'

# You can tranform the image values to be between -1 and 1 (the range of the tanh activation)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataloader = DataLoader(
    MNIST('.', download=False, transform=transform),
    batch_size=batch_size,
    shuffle=True)
```

Then, you can initialize your generator, discriminator, and optimizers.


```python
gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
disc = Discriminator().to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))

# You initialize the weights to the normal distribution
# with mean 0 and standard deviation 0.02
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
gen = gen.apply(weights_init)
disc = disc.apply(weights_init)
```

Finally, you can train your GAN!
For each epoch, you will process the entire dataset in batches. For every batch, you will update the discriminator and generator. Then, you can see DCGAN's results!

Here's roughly the progression you should be expecting. On GPU this takes about 30 seconds per thousand steps. On CPU, this can take about 8 hours per thousand steps. You might notice that in the image of Step 5000, the generator is disproprotionately producing things that look like ones. If the discriminator didn't learn to detect this imbalance quickly enough, then the generator could just produce more ones. As a result, it may have ended up tricking the discriminator so well that there would be no more improvement, known as mode collapse: 
![MNIST Digits Progression](MNIST_DCGAN_Progression.png)



```python
n_epochs = 50
cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
for epoch in range(n_epochs):
    # Dataloader returns the batches
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.to(device)

        ## Update discriminator ##
        disc_opt.zero_grad()
        fake_noise = get_noise(cur_batch_size, z_dim, device=device)
        fake = gen(fake_noise)
        disc_fake_pred = disc(fake.detach())
        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_pred = disc(real)
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / display_step
        # Update gradients
        disc_loss.backward(retain_graph=True)
        # Update optimizer
        disc_opt.step()

        ## Update generator ##
        gen_opt.zero_grad()
        fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
        fake_2 = gen(fake_noise_2)
        disc_fake_pred = disc(fake_2)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        gen_loss.backward()
        gen_opt.step()

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / display_step

        ## Visualization code ##
        if cur_step % display_step == 0 and cur_step > 0:
            print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            show_tensor_images(fake)
            show_tensor_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1

```


    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 500: Generator loss: 0.9352222303152082, discriminator loss: 0.5145211278796199



![png](output_20_4.png)



![png](output_20_5.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 1000: Generator loss: 2.4388038697242695, discriminator loss: 0.13375114747881892



![png](output_20_9.png)



![png](output_20_10.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 1500: Generator loss: 2.778523303240538, discriminator loss: 0.18442334569618102



![png](output_20_14.png)



![png](output_20_15.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 2000: Generator loss: 1.5564042309969655, discriminator loss: 0.411481518417597



![png](output_20_19.png)



![png](output_20_20.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 2500: Generator loss: 1.2204699741601934, discriminator loss: 0.4755505216121675



![png](output_20_24.png)



![png](output_20_25.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 3000: Generator loss: 1.0237537466287625, discriminator loss: 0.5508515179753305



![png](output_20_29.png)



![png](output_20_30.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 3500: Generator loss: 0.9337309087514867, discriminator loss: 0.5833885197639472



![png](output_20_34.png)



![png](output_20_35.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 4000: Generator loss: 0.8840442376136782, discriminator loss: 0.6092064551115036



![png](output_20_39.png)



![png](output_20_40.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 4500: Generator loss: 0.8287717036008844, discriminator loss: 0.6340877972841261



![png](output_20_44.png)



![png](output_20_45.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 5000: Generator loss: 0.7987173094749446, discriminator loss: 0.6529265396594998



![png](output_20_49.png)



![png](output_20_50.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 5500: Generator loss: 0.7908751960992807, discriminator loss: 0.6584804737567898



![png](output_20_54.png)



![png](output_20_55.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 6000: Generator loss: 0.7847455229759213, discriminator loss: 0.6658640645742417



![png](output_20_59.png)



![png](output_20_60.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 6500: Generator loss: 0.7691314390897758, discriminator loss: 0.673721488237381



![png](output_20_64.png)



![png](output_20_65.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 7000: Generator loss: 0.770869798839092, discriminator loss: 0.6776986620426181



![png](output_20_69.png)



![png](output_20_70.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 7500: Generator loss: 0.7684103722572324, discriminator loss: 0.6763846687078481



![png](output_20_74.png)



![png](output_20_75.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 8000: Generator loss: 0.7618033061027528, discriminator loss: 0.6792053034305581



![png](output_20_81.png)



![png](output_20_82.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 8500: Generator loss: 0.7562337343692784, discriminator loss: 0.6791251318454741



![png](output_20_86.png)



![png](output_20_87.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 9000: Generator loss: 0.7508334827423091, discriminator loss: 0.6797513142824172



![png](output_20_91.png)



![png](output_20_92.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 9500: Generator loss: 0.7520952570438375, discriminator loss: 0.6832506746053695



![png](output_20_96.png)



![png](output_20_97.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 10000: Generator loss: 0.7427340077757839, discriminator loss: 0.6865545665025714



![png](output_20_101.png)



![png](output_20_102.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 10500: Generator loss: 0.7417237596511838, discriminator loss: 0.6885304559469212



![png](output_20_106.png)



![png](output_20_107.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 11000: Generator loss: 0.7328983456492422, discriminator loss: 0.6914695100784302



![png](output_20_111.png)



![png](output_20_112.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 11500: Generator loss: 0.7289622874855994, discriminator loss: 0.6928530309200285



![png](output_20_116.png)



![png](output_20_117.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 12000: Generator loss: 0.7311369514465331, discriminator loss: 0.6928883051872242



![png](output_20_121.png)



![png](output_20_122.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 12500: Generator loss: 0.7186608016490932, discriminator loss: 0.7009185068607325



![png](output_20_126.png)



![png](output_20_127.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 13000: Generator loss: 0.7134759659767155, discriminator loss: 0.6978426716327661



![png](output_20_131.png)



![png](output_20_132.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 13500: Generator loss: 0.7192825925946239, discriminator loss: 0.6991316813230515



![png](output_20_136.png)



![png](output_20_137.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 14000: Generator loss: 0.7155776389241228, discriminator loss: 0.6986595014333722



![png](output_20_141.png)



![png](output_20_142.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 14500: Generator loss: 0.7132554928064342, discriminator loss: 0.6983388394117354



![png](output_20_146.png)



![png](output_20_147.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 15000: Generator loss: 0.7143500047922136, discriminator loss: 0.6993052992820744



![png](output_20_151.png)



![png](output_20_152.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 15500: Generator loss: 0.7081128971576691, discriminator loss: 0.6991398068666455



![png](output_20_158.png)



![png](output_20_159.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 16000: Generator loss: 0.704739709973336, discriminator loss: 0.7008727791309352



![png](output_20_163.png)



![png](output_20_164.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 16500: Generator loss: 0.7056849538683889, discriminator loss: 0.6999587173461924



![png](output_20_168.png)



![png](output_20_169.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 17000: Generator loss: 0.7071947321295741, discriminator loss: 0.6989173229932785



![png](output_20_173.png)



![png](output_20_174.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 17500: Generator loss: 0.7053896815776821, discriminator loss: 0.6990692837238311



![png](output_20_178.png)



![png](output_20_179.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 18000: Generator loss: 0.7031506487727169, discriminator loss: 0.6993944189548494



![png](output_20_183.png)



![png](output_20_184.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 18500: Generator loss: 0.7049044753909122, discriminator loss: 0.6989638904333116



![png](output_20_188.png)



![png](output_20_189.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 19000: Generator loss: 0.7032143213748939, discriminator loss: 0.6982788884639739



![png](output_20_193.png)



![png](output_20_194.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 19500: Generator loss: 0.704091481685639, discriminator loss: 0.698294734120369



![png](output_20_198.png)



![png](output_20_199.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 20000: Generator loss: 0.7033814581036568, discriminator loss: 0.697908317327499



![png](output_20_203.png)



![png](output_20_204.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 20500: Generator loss: 0.7019732355475422, discriminator loss: 0.6979859193563464



![png](output_20_208.png)



![png](output_20_209.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 21000: Generator loss: 0.6996736483573912, discriminator loss: 0.6988371458053585



![png](output_20_213.png)



![png](output_20_214.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 21500: Generator loss: 0.7008826977014543, discriminator loss: 0.6984157093763355



![png](output_20_218.png)



![png](output_20_219.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 22000: Generator loss: 0.7006542595028876, discriminator loss: 0.6972620655298232



![png](output_20_223.png)



![png](output_20_224.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 22500: Generator loss: 0.6996267964839937, discriminator loss: 0.6974654963016501



![png](output_20_228.png)



![png](output_20_229.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 23000: Generator loss: 0.6973336805105212, discriminator loss: 0.6982578172683712



![png](output_20_235.png)



![png](output_20_236.png)


    



```python

```
