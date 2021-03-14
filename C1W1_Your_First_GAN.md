# Your First GAN

### Goal
In this notebook, you're going to create your first generative adversarial network (GAN) for this course! Specifically, you will build and train a GAN that can generate hand-written images of digits (0-9). You will be using PyTorch in this specialization, so if you're not familiar with this framework, you may find the [PyTorch documentation](https://pytorch.org/docs/stable/index.html) useful. The hints will also often include links to relevant documentation.

### Learning Objectives
1.   Build the generator and discriminator components of a GAN from scratch.
2.   Create generator and discriminator loss functions.
3.   Train your GAN and visualize the generated images.


## Getting Started
You will begin by importing some useful packages and the dataset you will use to build and train your GAN. You are also provided with a visualizer function to help you investigate the images your GAN will create.



```python
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST # Training dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0) # Set for testing purposes, please do not change!

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
```

#### MNIST Dataset
The training images your discriminator will be using is from a dataset called [MNIST](http://yann.lecun.com/exdb/mnist/). It contains 60,000 images of handwritten digits, from 0 to 9, like these:

![MNIST Digits](MnistExamples.png)

You may notice that the images are quite pixelated -- this is because they are all only 28 x 28! The small size of its images makes MNIST ideal for simple training. Additionally, these images are also in black-and-white so only one dimension, or "color channel", is needed to represent them (more on this later in the course).

#### Tensor
You will represent the data using [tensors](https://pytorch.org/docs/stable/tensors.html). Tensors are a generalization of matrices: for example, a stack of three matrices with the amounts of red, green, and blue at different locations in a 64 x 64 pixel image is a tensor with the shape 3 x 64 x 64.

Tensors are easy to manipulate and supported by [PyTorch](https://pytorch.org/), the machine learning library you will be using. Feel free to explore them more, but you can imagine these as multi-dimensional matrices or vectors!

#### Batches
While you could train your model after generating one image, it is extremely inefficient and leads to less stable training. In GANs, and in machine learning in general, you will process multiple images per training step. These are called batches.

This means that your generator will generate an entire batch of images and receive the discriminator's feedback on each before updating the model. The same goes for the discriminator, it will calculate its loss on the entire batch of generated images as well as on the reals before the model is updated.

## Generator
The first step is to build the generator component.

You will start by creating a function to make a single layer/block for the generator's neural network. Each block should include a [linear transformation](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) to map to another shape, a [batch normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html) for stabilization, and finally a non-linear activation function (you use a [ReLU here](https://pytorch.org/docs/master/generated/torch.nn.ReLU.html)) so the output can be transformed in complex ways. You will learn more about activations and batch normalization later in the course.


```python
# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_generator_block
def get_generator_block(input_dim, output_dim):
    '''
    Function for returning a block of the generator's neural network
    given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a generator neural network layer, with a linear transformation 
          followed by a batch normalization and then a relu activation
    '''
    return nn.Sequential(
        # Hint: Replace all of the "None" with the appropriate dimensions.
        # The documentation may be useful if you're less familiar with PyTorch:
        # https://pytorch.org/docs/stable/nn.html.
        #### START CODE HERE ####
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True),
        #### END CODE HERE ####
    )
```


```python
# Verify the generator block function
def test_gen_block(in_features, out_features, num_test=1000):
    block = get_generator_block(in_features, out_features)

    # Check the three parts
    assert len(block) == 3
    assert type(block[0]) == nn.Linear
    assert type(block[1]) == nn.BatchNorm1d
    assert type(block[2]) == nn.ReLU
    
    # Check the output shape
    test_input = torch.randn(num_test, in_features)
    test_output = block(test_input)
    assert tuple(test_output.shape) == (num_test, out_features)
    assert test_output.std() > 0.55
    assert test_output.std() < 0.65

test_gen_block(25, 12)
test_gen_block(15, 28)
print("Success!")
```

    Success!


Now you can build the generator class. It will take 3 values:

*   The noise vector dimension
*   The image dimension
*   The initial hidden dimension

Using these values, the generator will build a neural network with 5 layers/blocks. Beginning with the noise vector, the generator will apply non-linear transformations via the block function until the tensor is mapped to the size of the image to be outputted (the same size as the real images from MNIST). You will need to fill in the code for final layer since it is different than the others. The final layer does not need a normalization or activation function, but does need to be scaled with a [sigmoid function](https://pytorch.org/docs/master/generated/torch.nn.Sigmoid.html). 

Finally, you are given a forward pass function that takes in a noise vector and generates an image of the output dimension using your neural network.

<details>

<summary>
<font size="3" color="green">
<b>Optional hints for <code><font size="4">Generator</font></code></b>
</font>
</summary>

1. The output size of the final linear transformation should be im_dim, but remember you need to scale the outputs between 0 and 1 using the sigmoid function.
2. [nn.Linear](https://pytorch.org/docs/master/generated/torch.nn.Linear.html) and [nn.Sigmoid](https://pytorch.org/docs/master/generated/torch.nn.Sigmoid.html) will be useful here. 
</details>



```python
# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: Generator
class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
          (MNIST images are 28 x 28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        # Build the neural network
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            # There is a dropdown with hints if you need them! 
            #### START CODE HERE ####
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
            
            #### END CODE HERE ####
        )
    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.gen(noise)
    
    # Needed for grading
    def get_gen(self):
        '''
        Returns:
            the sequential model
        '''
        return self.gen
```


```python
# Verify the generator class
def test_generator(z_dim, im_dim, hidden_dim, num_test=10000):
    gen = Generator(z_dim, im_dim, hidden_dim).get_gen()
    
    # Check there are six modules in the sequential part
    assert len(gen) == 6
    test_input = torch.randn(num_test, z_dim)
    test_output = gen(test_input)

    # Check that the output shape is correct
    assert tuple(test_output.shape) == (num_test, im_dim)
    assert test_output.max() < 1, "Make sure to use a sigmoid"
    assert test_output.min() > 0, "Make sure to use a sigmoid"
    assert test_output.min() < 0.5, "Don't use a block in your solution"
    assert test_output.std() > 0.05, "Don't use batchnorm here"
    assert test_output.std() < 0.15, "Don't use batchnorm here"

test_generator(5, 10, 20)
test_generator(20, 8, 24)
print("Success!")
```

    Success!


## Noise
To be able to use your generator, you will need to be able to create noise vectors. The noise vector z has the important role of making sure the images generated from the same class don't all look the same -- think of it as a random seed. You will generate it randomly using PyTorch by sampling random numbers from the normal distribution. Since multiple images will be processed per pass, you will generate all the noise vectors at once.

Note that whenever you create a new tensor using torch.ones, torch.zeros, or torch.randn, you either need to create it on the target device, e.g. `torch.ones(3, 3, device=device)`, or move it onto the target device using `torch.ones(3, 3).to(device)`. You do not need to do this if you're creating a tensor by manipulating another tensor or by using a variation that defaults the device to the input, such as `torch.ones_like`. In general, use `torch.ones_like` and `torch.zeros_like` instead of `torch.ones` or `torch.zeros` where possible.

<details>

<summary>
<font size="3" color="green">
<b>Optional hint for <code><font size="4">get_noise</font></code></b>
</font>
</summary>

1. 
You will probably find [torch.randn](https://pytorch.org/docs/master/generated/torch.randn.html) useful here.
</details>


```python
# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_noise
def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    # NOTE: To use this on GPU with device='cuda', make sure to pass the device 
    # argument to the function you use to generate the noise.
    #### START CODE HERE ####
    return torch.randn(n_samples, z_dim, device=device)
    #### END CODE HERE ####
```


```python
# Verify the noise vector function
def test_get_noise(n_samples, z_dim, device='cpu'):
    noise = get_noise(n_samples, z_dim, device)
    
    # Make sure a normal distribution was used
    assert tuple(noise.shape) == (n_samples, z_dim)
    assert torch.abs(noise.std() - torch.tensor(1.0)) < 0.01
    assert str(noise.device).startswith(device)

test_get_noise(1000, 100, 'cpu')
if torch.cuda.is_available():
    test_get_noise(1000, 32, 'cuda')
print("Success!")
```

    Success!


## Discriminator
The second component that you need to construct is the discriminator. As with the generator component, you will start by creating a function that builds a neural network block for the discriminator.

*Note: You use leaky ReLUs to prevent the "dying ReLU" problem, which refers to the phenomenon where the parameters stop changing due to consistently negative values passed to a ReLU, which result in a zero gradient. You will learn more about this in the following lectures!* 


REctified Linear Unit (ReLU) |  Leaky ReLU
:-------------------------:|:-------------------------:
![](relu-graph.png)  |  ![](lrelu-graph.png)






```python
# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_discriminator_block
def get_discriminator_block(input_dim, output_dim):
    '''
    Discriminator Block
    Function for returning a neural network of the discriminator given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a discriminator neural network layer, with a linear transformation 
          followed by an nn.LeakyReLU activation with negative slope of 0.2 
          (https://pytorch.org/docs/master/generated/torch.nn.LeakyReLU.html)
    '''
    return nn.Sequential(
        #### START CODE HERE ####
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2, inplace=True)
        #### END CODE HERE ####
    )
```


```python
# Verify the discriminator block function
def test_disc_block(in_features, out_features, num_test=10000):
    block = get_discriminator_block(in_features, out_features)

    # Check there are two parts
    assert len(block) == 2
    test_input = torch.randn(num_test, in_features)
    test_output = block(test_input)

    # Check that the shape is right
    assert tuple(test_output.shape) == (num_test, out_features)
    
    # Check that the LeakyReLU slope is about 0.2
    assert -test_output.min() / test_output.max() > 0.1
    assert -test_output.min() / test_output.max() < 0.3
    assert test_output.std() > 0.3
    assert test_output.std() < 0.5

test_disc_block(25, 12)
test_disc_block(15, 28)
print("Success!")
```

    Success!


Now you can use these blocks to make a discriminator! The discriminator class holds 2 values:

*   The image dimension
*   The hidden dimension

The discriminator will build a neural network with 4 layers. It will start with the image tensor and transform it until it returns a single number (1-dimension tensor) output. This output classifies whether an image is fake or real. Note that you do not need a sigmoid after the output layer since it is included in the loss function. Finally, to use your discrimator's neural network you are given a forward pass function that takes in an image tensor to be classified.



```python
# UNQ_C5 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: Discriminator
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
            (MNIST images are 28x28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            # Hint: You want to transform the final output into a single value,
            #       so add one more linear map.
            #### START CODE HERE ####
            nn.Linear(hidden_dim, 1)
            
            #### END CODE HERE ####
        )

    def forward(self, image):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        '''
        return self.disc(image)
    
    # Needed for grading
    def get_disc(self):
        '''
        Returns:
            the sequential model
        '''
        return self.disc
```


```python
# Verify the discriminator class
def test_discriminator(z_dim, hidden_dim, num_test=100):
    
    disc = Discriminator(z_dim, hidden_dim).get_disc()

    # Check there are three parts
    assert len(disc) == 4

    # Check the linear layer is correct
    test_input = torch.randn(num_test, z_dim)
    test_output = disc(test_input)
    assert tuple(test_output.shape) == (num_test, 1)
    
    # Don't use a block
    assert not isinstance(disc[-1], nn.Sequential)

test_discriminator(5, 10)
test_discriminator(20, 8)
print("Success!")
```

    Success!


## Training
Now you can put it all together!
First, you will set your parameters:
  *   criterion: the loss function
  *   n_epochs: the number of times you iterate through the entire dataset when training
  *   z_dim: the dimension of the noise vector
  *   display_step: how often to display/visualize the images
  *   batch_size: the number of images per forward/backward pass
  *   lr: the learning rate
  *   device: the device type, here using a GPU (which runs CUDA), not CPU

Next, you will load the MNIST dataset as tensors using a dataloader.




```python
# Set your parameters
criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.00001

# Load MNIST dataset as tensors
dataloader = DataLoader(
    MNIST('.', download=False, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True)

### DO NOT EDIT ###
device = 'cuda'
```

Now, you can initialize your generator, discriminator, and optimizers. Note that each optimizer only takes the parameters of one particular model, since we want each optimizer to optimize only one of the models.


```python
gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
```

Before you train your GAN, you will need to create functions to calculate the discriminator's loss and the generator's loss. This is how the discriminator and generator will know how they are doing and improve themselves. Since the generator is needed when calculating the discriminator's loss, you will need to call .detach() on the generator result to ensure that only the discriminator is updated!

Remember that you have already defined a loss function earlier (`criterion`) and you are encouraged to use `torch.ones_like` and `torch.zeros_like` instead of `torch.ones` or `torch.zeros`. If you use `torch.ones` or `torch.zeros`, you'll need to pass `device=device` to them.


```python
# UNQ_C6 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_disc_loss
def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        real: a batch of real images
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    '''
    #     These are the steps you will need to complete:
    #       1) Create noise vectors and generate a batch (num_images) of fake images. 
    #            Make sure to pass the device argument to the noise.
    #       2) Get the discriminator's prediction of the fake image 
    #            and calculate the loss. Don't forget to detach the generator!
    #            (Remember the loss function you set earlier -- criterion. You need a 
    #            'ground truth' tensor in order to calculate the loss. 
    #            For example, a ground truth tensor for a fake image is all zeros.)
    #       3) Get the discriminator's prediction of the real image and calculate the loss.
    #       4) Calculate the discriminator's loss by averaging the real and fake loss
    #            and set it to disc_loss.
    #     Note: Please do not use concatenation in your solution. The tests are being updated to 
    #           support this, but for now, average the two losses as described in step (4).
    #     *Important*: You should NOT write your own loss function here - use criterion(pred, true)!
    #### START CODE HERE ####
    fake_noise=get_noise(num_images, z_dim, device=device)
    fake=gen(fake_noise)
    disc_fake_pred = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_real_pred = disc(real.detach())
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    disc_loss = (disc_fake_loss + disc_real_loss)/2
    #### END CODE HERE ####
    return disc_loss
```


```python
def test_disc_reasonable(num_images=10):
    # Don't use explicit casts to cuda - use the device argument
    import inspect, re
    lines = inspect.getsource(get_disc_loss)
    assert (re.search(r"to\(.cuda.\)", lines)) is None
    assert (re.search(r"\.cuda\(\)", lines)) is None
    
    z_dim = 64
    gen = torch.zeros_like
    disc = lambda x: x.mean(1)[:, None]
    criterion = torch.mul # Multiply
    real = torch.ones(num_images, z_dim)
    disc_loss = get_disc_loss(gen, disc, criterion, real, num_images, z_dim, 'cpu')
    assert torch.all(torch.abs(disc_loss.mean() - 0.5) < 1e-5)
    
    gen = torch.ones_like
    criterion = torch.mul # Multiply
    real = torch.zeros(num_images, z_dim)
    assert torch.all(torch.abs(get_disc_loss(gen, disc, criterion, real, num_images, z_dim, 'cpu')) < 1e-5)
    
    gen = lambda x: torch.ones(num_images, 10)
    disc = lambda x: x.mean(1)[:, None] + 10
    criterion = torch.mul # Multiply
    real = torch.zeros(num_images, 10)
    assert torch.all(torch.abs(get_disc_loss(gen, disc, criterion, real, num_images, z_dim, 'cpu').mean() - 5) < 1e-5)

    gen = torch.ones_like
    disc = nn.Linear(64, 1, bias=False)
    real = torch.ones(num_images, 64) * 0.5
    disc.weight.data = torch.ones_like(disc.weight.data) * 0.5
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    criterion = lambda x, y: torch.sum(x) + torch.sum(y)
    disc_loss = get_disc_loss(gen, disc, criterion, real, num_images, z_dim, 'cpu').mean()
    disc_loss.backward()
    assert torch.isclose(torch.abs(disc.weight.grad.mean() - 11.25), torch.tensor(3.75))
    
def test_disc_loss(max_tests = 10):
    z_dim = 64
    gen = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = Discriminator().to(device) 
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    num_steps = 0
    for real, _ in dataloader:
        cur_batch_size = len(real)
        real = real.view(cur_batch_size, -1).to(device)

        ### Update discriminator ###
        # Zero out the gradient before backpropagation
        disc_opt.zero_grad()

        # Calculate discriminator loss
        disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)
        assert (disc_loss - 0.68).abs() < 0.05

        # Update gradients
        disc_loss.backward(retain_graph=True)

        # Check that they detached correctly
        assert gen.gen[0][0].weight.grad is None

        # Update optimizer
        old_weight = disc.disc[0][0].weight.data.clone()
        disc_opt.step()
        new_weight = disc.disc[0][0].weight.data
        
        # Check that some discriminator weights changed
        assert not torch.all(torch.eq(old_weight, new_weight))
        num_steps += 1
        if num_steps >= max_tests:
            break

test_disc_reasonable()
test_disc_loss()
print("Success!")
```

    Success!



```python
# UNQ_C7 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_gen_loss
def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        gen_loss: a torch scalar loss value for the current batch
    '''
    #     These are the steps you will need to complete:
    #       1) Create noise vectors and generate a batch of fake images. 
    #           Remember to pass the device argument to the get_noise function.
    #       2) Get the discriminator's prediction of the fake image.
    #       3) Calculate the generator's loss. Remember the generator wants
    #          the discriminator to think that its fake images are real
    #     *Important*: You should NOT write your own loss function here - use criterion(pred, true)!

    #### START CODE HERE ####
    fake_noise = get_noise(num_images, z_dim, device=device)
    fake = gen(fake_noise)
    disc_fake_pred = disc(fake)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) #to have positive insight. (1-loss)
     
    #### END CODE HERE ####
    return gen_loss
```


```python
def test_gen_reasonable(num_images=10):
    # Don't use explicit casts to cuda - use the device argument
    import inspect, re
    lines = inspect.getsource(get_gen_loss)
    assert (re.search(r"to\(.cuda.\)", lines)) is None
    assert (re.search(r"\.cuda\(\)", lines)) is None
    
    z_dim = 64
    gen = torch.zeros_like
    disc = nn.Identity()
    criterion = torch.mul # Multiply
    gen_loss_tensor = get_gen_loss(gen, disc, criterion, num_images, z_dim, 'cpu')
    assert torch.all(torch.abs(gen_loss_tensor) < 1e-5)
    #Verify shape. Related to gen_noise parametrization
    assert tuple(gen_loss_tensor.shape) == (num_images, z_dim)

    gen = torch.ones_like
    disc = nn.Identity()
    criterion = torch.mul # Multiply
    real = torch.zeros(num_images, 1)
    gen_loss_tensor = get_gen_loss(gen, disc, criterion, num_images, z_dim, 'cpu')
    assert torch.all(torch.abs(gen_loss_tensor - 1) < 1e-5)
    #Verify shape. Related to gen_noise parametrization
    assert tuple(gen_loss_tensor.shape) == (num_images, z_dim)
    

def test_gen_loss(num_images):
    z_dim = 64
    gen = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = Discriminator().to(device) 
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    
    gen_loss = get_gen_loss(gen, disc, criterion, num_images, z_dim, device)
    
    # Check that the loss is reasonable
    assert (gen_loss - 0.7).abs() < 0.1
    gen_loss.backward()
    old_weight = gen.gen[0][0].weight.clone()
    gen_opt.step()
    new_weight = gen.gen[0][0].weight
    assert not torch.all(torch.eq(old_weight, new_weight))


test_gen_reasonable(10)
test_gen_loss(18)
print("Success!")
```

    Success!


Finally, you can put everything together! For each epoch, you will process the entire dataset in batches. For every batch, you will need to update the discriminator and generator using their loss. Batches are sets of images that will be predicted on before the loss functions are calculated (instead of calculating the loss function after each image). Note that you may see a loss to be greater than 1, this is okay since binary cross entropy loss can be any positive number for a sufficiently confident wrong guess. 

It’s also often the case that the discriminator will outperform the generator, especially at the start, because its job is easier. It's important that neither one gets too good (that is, near-perfect accuracy), which would cause the entire model to stop learning. Balancing the two models is actually remarkably hard to do in a standard GAN and something you will see more of in later lectures and assignments.

After you've submitted a working version with the original architecture, feel free to play around with the architecture if you want to see how different architectural choices can lead to better or worse GANs. For example, consider changing the size of the hidden dimension, or making the networks shallower or deeper by changing the number of layers.

<!-- In addition, be warned that this runs very slowly on a CPU. One way to run this more quickly is to use Google Colab: 

1.   Download the .ipynb
2.   Upload it to Google Drive and open it with Google Colab
3.   Make the runtime type GPU (under “Runtime” -> “Change runtime type” -> Select “GPU” from the dropdown)
4.   Replace `device = "cpu"` with `device = "cuda"`
5.   Make sure your `get_noise` function uses the right device -->

But remember, don’t expect anything spectacular: this is only the first lesson. The results will get better with later lessons as you learn methods to help keep your generator and discriminator at similar levels.

You should roughly expect to see this progression. On a GPU, this should take about 15 seconds per 500 steps, on average, while on CPU it will take roughly 1.5 minutes:
![MNIST Digits](MNIST_Progression.png)


```python
# UNQ_C8 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: 

cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
test_generator = True # Whether the generator should be tested
gen_loss = False
error = False
for epoch in range(n_epochs):
  
    # Dataloader returns the batches
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)

        # Flatten the batch of real images from the dataset
        real = real.view(cur_batch_size, -1).to(device)

        ### Update discriminator ###
        # Zero out the gradients before backpropagation
        disc_opt.zero_grad()

        # Calculate discriminator loss
        disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)

        # Update gradients
        disc_loss.backward(retain_graph=True)

        # Update optimizer
        disc_opt.step()

        # For testing purposes, to keep track of the generator weights
        if test_generator:
            old_generator_weights = gen.gen[0][0].weight.detach().clone()

        ### Update generator ###
        #     Hint: This code will look a lot like the discriminator updates!
        #     These are the steps you will need to complete:
        #       1) Zero out the gradients.
        #       2) Calculate the generator loss, assigning it to gen_loss.
        #       3) Backprop through the generator: update the gradients and optimizer.
        #### START CODE HERE ####
        gen_opt.zero_grad()
        gen_loss=get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
        gen_loss.backward()
        gen_opt.step()
        
        #### END CODE HERE ####

        # For testing purposes, to check that your code changes the generator weights
        if test_generator:
            try:
                assert lr > 0.0000002 or (gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
            except:
                error = True
                print("Runtime tests have failed")

        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / display_step

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / display_step

        ### Visualization code ###
        if cur_step % display_step == 0 and cur_step > 0:
            print(f"Epoch {epoch}, step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
            show_tensor_images(fake)
            show_tensor_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1

```


    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 1, step 500: Generator loss: 1.4475373233556743, discriminator loss: 0.4217771633863452



![png](output_31_4.png)



![png](output_31_5.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 2, step 1000: Generator loss: 1.7768991763591775, discriminator loss: 0.27962395521998384



![png](output_31_9.png)



![png](output_31_10.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 3, step 1500: Generator loss: 2.041974313735963, discriminator loss: 0.16707849788665774



![png](output_31_14.png)



![png](output_31_15.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 4, step 2000: Generator loss: 1.6807748553752888, discriminator loss: 0.22415010270476346



![png](output_31_19.png)



![png](output_31_20.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 5, step 2500: Generator loss: 1.5928061339855182, discriminator loss: 0.22971207100152966



![png](output_31_24.png)



![png](output_31_25.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 6, step 3000: Generator loss: 1.8635420479774478, discriminator loss: 0.1791343220472334



![png](output_31_29.png)



![png](output_31_30.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 7, step 3500: Generator loss: 2.2855837121009825, discriminator loss: 0.13966318468749525



![png](output_31_34.png)



![png](output_31_35.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 8, step 4000: Generator loss: 2.650408926010131, discriminator loss: 0.12534953506290908



![png](output_31_39.png)



![png](output_31_40.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 9, step 4500: Generator loss: 2.8774369215965283, discriminator loss: 0.10935433132946488



![png](output_31_44.png)



![png](output_31_45.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 10, step 5000: Generator loss: 3.296898881912234, discriminator loss: 0.08607413390278824



![png](output_31_49.png)



![png](output_31_50.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 11, step 5500: Generator loss: 3.55705529308319, discriminator loss: 0.07226575408875939



![png](output_31_54.png)



![png](output_31_55.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 12, step 6000: Generator loss: 3.584073222637179, discriminator loss: 0.07293257014453416



![png](output_31_59.png)



![png](output_31_60.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 13, step 6500: Generator loss: 3.8455388202667207, discriminator loss: 0.061823606818914356



![png](output_31_64.png)



![png](output_31_65.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 14, step 7000: Generator loss: 3.9122808384895302, discriminator loss: 0.06403436386585237



![png](output_31_69.png)



![png](output_31_70.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 15, step 7500: Generator loss: 3.967157605171206, discriminator loss: 0.0638870343416929



![png](output_31_74.png)



![png](output_31_75.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 17, step 8000: Generator loss: 4.154760721683502, discriminator loss: 0.05570941548794506



![png](output_31_81.png)



![png](output_31_82.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 18, step 8500: Generator loss: 4.07525518131256, discriminator loss: 0.0650831388533115



![png](output_31_86.png)



![png](output_31_87.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 19, step 9000: Generator loss: 4.189180427551269, discriminator loss: 0.07117475309967997



![png](output_31_91.png)



![png](output_31_92.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 20, step 9500: Generator loss: 4.2794408173561145, discriminator loss: 0.05903383233398199



![png](output_31_96.png)



![png](output_31_97.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 21, step 10000: Generator loss: 4.35004794692993, discriminator loss: 0.06263346967101095



![png](output_31_101.png)



![png](output_31_102.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 22, step 10500: Generator loss: 4.236064156055449, discriminator loss: 0.06294950219243776



![png](output_31_106.png)



![png](output_31_107.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 23, step 11000: Generator loss: 4.048557625293732, discriminator loss: 0.07272683238238094



![png](output_31_111.png)



![png](output_31_112.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 24, step 11500: Generator loss: 3.884071440219877, discriminator loss: 0.07794716517627237



![png](output_31_116.png)



![png](output_31_117.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 25, step 12000: Generator loss: 3.9920513806343076, discriminator loss: 0.08854050962626919



![png](output_31_121.png)



![png](output_31_122.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 26, step 12500: Generator loss: 3.9203925256729106, discriminator loss: 0.08769178218394522



![png](output_31_126.png)



![png](output_31_127.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 27, step 13000: Generator loss: 3.96509117221832, discriminator loss: 0.09618550802022219



![png](output_31_131.png)



![png](output_31_132.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 28, step 13500: Generator loss: 3.8501110410690327, discriminator loss: 0.10143955145031225



![png](output_31_136.png)



![png](output_31_137.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 29, step 14000: Generator loss: 3.8444834909439103, discriminator loss: 0.09440734849125143



![png](output_31_141.png)



![png](output_31_142.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 30, step 14500: Generator loss: 3.7114847617149347, discriminator loss: 0.10970555312931539



![png](output_31_146.png)



![png](output_31_147.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 31, step 15000: Generator loss: 3.4725400056839013, discriminator loss: 0.12133162436634297



![png](output_31_151.png)



![png](output_31_152.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 33, step 15500: Generator loss: 3.7922165760993933, discriminator loss: 0.10551291213184598



![png](output_31_158.png)



![png](output_31_159.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 34, step 16000: Generator loss: 3.598603702068328, discriminator loss: 0.1227019505277276



![png](output_31_163.png)



![png](output_31_164.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 35, step 16500: Generator loss: 3.6380697827339206, discriminator loss: 0.12945154887437818



![png](output_31_168.png)



![png](output_31_169.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 36, step 17000: Generator loss: 3.537538875579836, discriminator loss: 0.13474458311498164



![png](output_31_173.png)



![png](output_31_174.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 37, step 17500: Generator loss: 3.476137829303737, discriminator loss: 0.14868745326995852



![png](output_31_178.png)



![png](output_31_179.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 38, step 18000: Generator loss: 3.3991261878013623, discriminator loss: 0.14605226826667797



![png](output_31_183.png)



![png](output_31_184.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 39, step 18500: Generator loss: 3.3172059221267673, discriminator loss: 0.15349054150283328



![png](output_31_188.png)



![png](output_31_189.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 40, step 19000: Generator loss: 3.2617309961318988, discriminator loss: 0.14843417447805407



![png](output_31_193.png)



![png](output_31_194.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 41, step 19500: Generator loss: 3.3134564075470005, discriminator loss: 0.15021607211232188



![png](output_31_198.png)



![png](output_31_199.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 42, step 20000: Generator loss: 3.2072959723472585, discriminator loss: 0.17040632909536343



![png](output_31_203.png)



![png](output_31_204.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 43, step 20500: Generator loss: 3.228657251358032, discriminator loss: 0.17316192586719986



![png](output_31_208.png)



![png](output_31_209.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 44, step 21000: Generator loss: 3.2332784137725823, discriminator loss: 0.18531820809841146



![png](output_31_213.png)



![png](output_31_214.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 45, step 21500: Generator loss: 3.1130473518371606, discriminator loss: 0.19156752455234535



![png](output_31_218.png)



![png](output_31_219.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 46, step 22000: Generator loss: 3.0513907842636097, discriminator loss: 0.19792709010839477



![png](output_31_223.png)



![png](output_31_224.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 47, step 22500: Generator loss: 3.151818241596223, discriminator loss: 0.16750545686483367



![png](output_31_228.png)



![png](output_31_229.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 49, step 23000: Generator loss: 3.0933192858696033, discriminator loss: 0.18385891990363587



![png](output_31_235.png)



![png](output_31_236.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 50, step 23500: Generator loss: 3.0746033163070665, discriminator loss: 0.17869531346857565



![png](output_31_240.png)



![png](output_31_241.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 51, step 24000: Generator loss: 3.000071912765503, discriminator loss: 0.19365139329433426



![png](output_31_245.png)



![png](output_31_246.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 52, step 24500: Generator loss: 3.056108485698697, discriminator loss: 0.18230112397670747



![png](output_31_250.png)



![png](output_31_251.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 53, step 25000: Generator loss: 2.9800130395889286, discriminator loss: 0.18243409375846406



![png](output_31_255.png)



![png](output_31_256.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 54, step 25500: Generator loss: 2.9780291280746445, discriminator loss: 0.17903573825955396



![png](output_31_260.png)



![png](output_31_261.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 55, step 26000: Generator loss: 2.8625023918151857, discriminator loss: 0.20438724489510063



![png](output_31_265.png)



![png](output_31_266.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 56, step 26500: Generator loss: 2.8358225784301765, discriminator loss: 0.2250171471834182



![png](output_31_270.png)



![png](output_31_271.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 57, step 27000: Generator loss: 2.7773908863067627, discriminator loss: 0.22030922698974623



![png](output_31_275.png)



![png](output_31_276.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 58, step 27500: Generator loss: 2.7729872465133667, discriminator loss: 0.21703329862654208



![png](output_31_280.png)



![png](output_31_281.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 59, step 28000: Generator loss: 2.873808602809905, discriminator loss: 0.19622995278239252



![png](output_31_285.png)



![png](output_31_286.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 60, step 28500: Generator loss: 2.8725935864448546, discriminator loss: 0.2120400696694848



![png](output_31_290.png)



![png](output_31_291.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 61, step 29000: Generator loss: 2.7604892930984493, discriminator loss: 0.23162816497683533



![png](output_31_295.png)



![png](output_31_296.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 62, step 29500: Generator loss: 2.779860451698302, discriminator loss: 0.22522012221813204



![png](output_31_300.png)



![png](output_31_301.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 63, step 30000: Generator loss: 2.5900454854965216, discriminator loss: 0.24237585672736173



![png](output_31_305.png)



![png](output_31_306.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 65, step 30500: Generator loss: 2.806504800319671, discriminator loss: 0.20356976585090167



![png](output_31_312.png)



![png](output_31_313.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 66, step 31000: Generator loss: 2.6975659265518175, discriminator loss: 0.22758181118965154



![png](output_31_317.png)



![png](output_31_318.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 67, step 31500: Generator loss: 2.6922203993797273, discriminator loss: 0.21992728421092028



![png](output_31_322.png)



![png](output_31_323.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 68, step 32000: Generator loss: 2.6854614348411547, discriminator loss: 0.21902306875586494



![png](output_31_327.png)



![png](output_31_328.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 69, step 32500: Generator loss: 2.537592000484469, discriminator loss: 0.25051711723208414



![png](output_31_332.png)



![png](output_31_333.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 70, step 33000: Generator loss: 2.5714261388778703, discriminator loss: 0.24838071811199186



![png](output_31_337.png)



![png](output_31_338.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 71, step 33500: Generator loss: 2.6506621410846702, discriminator loss: 0.24311522337794333



![png](output_31_342.png)



![png](output_31_343.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 72, step 34000: Generator loss: 2.541562378406523, discriminator loss: 0.24104207718372375



![png](output_31_347.png)



![png](output_31_348.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 73, step 34500: Generator loss: 2.5554183859825153, discriminator loss: 0.23722499781847023



![png](output_31_352.png)



![png](output_31_353.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 74, step 35000: Generator loss: 2.5854522085189804, discriminator loss: 0.24214156618714341



![png](output_31_357.png)



![png](output_31_358.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 75, step 35500: Generator loss: 2.490996934413908, discriminator loss: 0.2701240391731262



![png](output_31_362.png)



![png](output_31_363.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 76, step 36000: Generator loss: 2.471104089975356, discriminator loss: 0.2529927763640878



![png](output_31_367.png)



![png](output_31_368.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 77, step 36500: Generator loss: 2.4353368291854895, discriminator loss: 0.2730822113752363



![png](output_31_372.png)



![png](output_31_373.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 78, step 37000: Generator loss: 2.5442081406116497, discriminator loss: 0.24994614267349224



![png](output_31_377.png)



![png](output_31_378.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 79, step 37500: Generator loss: 2.460033162593839, discriminator loss: 0.24985656222701094



![png](output_31_382.png)



![png](output_31_383.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 81, step 38000: Generator loss: 2.446002781391144, discriminator loss: 0.24751602178812013



![png](output_31_389.png)



![png](output_31_390.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 82, step 38500: Generator loss: 2.5053034021854415, discriminator loss: 0.24871441692113855



![png](output_31_394.png)



![png](output_31_395.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 83, step 39000: Generator loss: 2.4482913148403167, discriminator loss: 0.25872713235020617



![png](output_31_399.png)



![png](output_31_400.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 84, step 39500: Generator loss: 2.3583177044391648, discriminator loss: 0.2749155844748021



![png](output_31_404.png)



![png](output_31_405.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 85, step 40000: Generator loss: 2.217077934265137, discriminator loss: 0.30152967557311033



![png](output_31_409.png)



![png](output_31_410.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 86, step 40500: Generator loss: 2.3079483809471113, discriminator loss: 0.27238610056042667



![png](output_31_414.png)



![png](output_31_415.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 87, step 41000: Generator loss: 2.4132610013484945, discriminator loss: 0.2679653983712195



![png](output_31_419.png)



![png](output_31_420.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 88, step 41500: Generator loss: 2.245090328693392, discriminator loss: 0.30126254302263256



![png](output_31_424.png)



![png](output_31_425.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 89, step 42000: Generator loss: 2.223274507761001, discriminator loss: 0.2969118188023568



![png](output_31_429.png)



![png](output_31_430.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 90, step 42500: Generator loss: 2.188658065319061, discriminator loss: 0.3106399511694908



![png](output_31_434.png)



![png](output_31_435.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 91, step 43000: Generator loss: 2.0491640889644636, discriminator loss: 0.3256315568089487



![png](output_31_439.png)



![png](output_31_440.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 92, step 43500: Generator loss: 2.147357345342636, discriminator loss: 0.30220237240195275



![png](output_31_444.png)



![png](output_31_445.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 93, step 44000: Generator loss: 2.2167523138523095, discriminator loss: 0.30497491645812963



![png](output_31_449.png)



![png](output_31_450.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 94, step 44500: Generator loss: 2.1497612688541414, discriminator loss: 0.3126592393219471



![png](output_31_454.png)



![png](output_31_455.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 95, step 45000: Generator loss: 2.196577823638914, discriminator loss: 0.2954574288725852



![png](output_31_459.png)



![png](output_31_460.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 97, step 45500: Generator loss: 2.209590916395186, discriminator loss: 0.3027860298752786



![png](output_31_466.png)



![png](output_31_467.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 98, step 46000: Generator loss: 2.115759415864945, discriminator loss: 0.31039357450604393



![png](output_31_471.png)



![png](output_31_472.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 99, step 46500: Generator loss: 2.0553502373695385, discriminator loss: 0.32129350909590737



![png](output_31_476.png)



![png](output_31_477.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 100, step 47000: Generator loss: 1.9907424826622002, discriminator loss: 0.3332687194645405



![png](output_31_481.png)



![png](output_31_482.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 101, step 47500: Generator loss: 2.192462791204453, discriminator loss: 0.29901550200581556



![png](output_31_486.png)



![png](output_31_487.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 102, step 48000: Generator loss: 2.0942940981388105, discriminator loss: 0.32409744980931277



![png](output_31_491.png)



![png](output_31_492.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 103, step 48500: Generator loss: 1.9125031144618978, discriminator loss: 0.36058401694893816



![png](output_31_496.png)



![png](output_31_497.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 104, step 49000: Generator loss: 2.0938665401935586, discriminator loss: 0.29732174566388114



![png](output_31_501.png)



![png](output_31_502.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 105, step 49500: Generator loss: 2.1513770575523394, discriminator loss: 0.30092661485075956



![png](output_31_506.png)



![png](output_31_507.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 106, step 50000: Generator loss: 2.0938416426181803, discriminator loss: 0.32132503622770364



![png](output_31_511.png)



![png](output_31_512.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 107, step 50500: Generator loss: 2.124116159915923, discriminator loss: 0.3131028214991093



![png](output_31_516.png)



![png](output_31_517.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 108, step 51000: Generator loss: 2.1086565654277822, discriminator loss: 0.30684145766496657



![png](output_31_521.png)



![png](output_31_522.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 109, step 51500: Generator loss: 2.0717173461914085, discriminator loss: 0.31269744142889994



![png](output_31_526.png)



![png](output_31_527.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 110, step 52000: Generator loss: 2.2690765070915235, discriminator loss: 0.26432638868689556



![png](output_31_531.png)



![png](output_31_532.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 111, step 52500: Generator loss: 2.100777529478072, discriminator loss: 0.3116443395614626



![png](output_31_536.png)



![png](output_31_537.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 113, step 53000: Generator loss: 2.0140002353191364, discriminator loss: 0.317390662997961



![png](output_31_543.png)



![png](output_31_544.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 114, step 53500: Generator loss: 1.9767336485385887, discriminator loss: 0.3301564186513423



![png](output_31_548.png)



![png](output_31_549.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 115, step 54000: Generator loss: 1.9220612208843209, discriminator loss: 0.3307022140622138



![png](output_31_553.png)



![png](output_31_554.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 116, step 54500: Generator loss: 1.9731115002632118, discriminator loss: 0.32911668062210103



![png](output_31_558.png)



![png](output_31_559.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 117, step 55000: Generator loss: 2.038378506660461, discriminator loss: 0.31534316632151604



![png](output_31_563.png)



![png](output_31_564.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 118, step 55500: Generator loss: 1.9790320153236391, discriminator loss: 0.33493396085500754



![png](output_31_568.png)



![png](output_31_569.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 119, step 56000: Generator loss: 2.030488199472427, discriminator loss: 0.3178176791369915



![png](output_31_573.png)



![png](output_31_574.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 120, step 56500: Generator loss: 1.9925491731166833, discriminator loss: 0.3253525467514994



![png](output_31_578.png)



![png](output_31_579.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 121, step 57000: Generator loss: 1.9835014984607697, discriminator loss: 0.3340300580263139



![png](output_31_583.png)



![png](output_31_584.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 122, step 57500: Generator loss: 1.989871557712554, discriminator loss: 0.31753232631087286



![png](output_31_588.png)



![png](output_31_589.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 123, step 58000: Generator loss: 1.9082379064559947, discriminator loss: 0.34731969439983373



![png](output_31_593.png)



![png](output_31_594.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 124, step 58500: Generator loss: 1.889137478828428, discriminator loss: 0.3492783522009852



![png](output_31_598.png)



![png](output_31_599.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 125, step 59000: Generator loss: 1.9564334213733676, discriminator loss: 0.3286019286513329



![png](output_31_603.png)



![png](output_31_604.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 126, step 59500: Generator loss: 1.751823681116104, discriminator loss: 0.3854568971991533



![png](output_31_608.png)



![png](output_31_609.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



```python

```
