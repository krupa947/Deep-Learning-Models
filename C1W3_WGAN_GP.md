# Wasserstein GAN with Gradient Penalty (WGAN-GP)

### Goals
In this notebook, you're going to build a Wasserstein GAN with Gradient Penalty (WGAN-GP) that solves some of the stability issues with the GANs that you have been using up until this point. Specifically, you'll use a special kind of loss function known as the W-loss, where W stands for Wasserstein, and gradient penalties to prevent mode collapse.

*Fun Fact: Wasserstein is named after a mathematician at Penn State, Leonid Vaseršteĭn. You'll see it abbreviated to W (e.g. WGAN, W-loss, W-distance).*

### Learning Objectives
1.   Get hands-on experience building a more stable GAN: Wasserstein GAN with Gradient Penalty (WGAN-GP).
2.   Train the more advanced WGAN-GP model.



## Generator and Critic

You will begin by importing some useful packages, defining visualization functions, building the generator, and building the critic. Since the changes for WGAN-GP are done to the loss function during training, you can simply reuse your previous GAN code for the generator and critic class. Remember that in WGAN-GP, you no longer use a discriminator that classifies fake and real as 0 and 1 but rather a critic that scores images with real numbers.

#### Packages and Visualizations


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

def make_grad_hook():
    '''
    Function to keep track of gradients for visualization purposes, 
    which fills the grads list when using model.apply(grad_hook).
    '''
    grads = []
    def grad_hook(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            grads.append(m.weight.grad)
    return grads, grad_hook
```

#### Generator and Noise


```python
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
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        x = noise.view(len(noise), self.z_dim, 1, 1)
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

#### Critic


```python
class Critic(nn.Module):
    '''
    Critic Class
    Values:
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            self.make_crit_block(im_chan, hidden_dim),
            self.make_crit_block(hidden_dim, hidden_dim * 2),
            self.make_crit_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_crit_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a critic block of DCGAN;
        a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the critic: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        '''
        crit_pred = self.crit(image)
        return crit_pred.view(len(crit_pred), -1)
```

## Training Initializations
Now you can start putting it all together.
As usual, you will start by setting the parameters:
  *   n_epochs: the number of times you iterate through the entire dataset when training
  *   z_dim: the dimension of the noise vector
  *   display_step: how often to display/visualize the images
  *   batch_size: the number of images per forward/backward pass
  *   lr: the learning rate
  *   beta_1, beta_2: the momentum terms
  *   c_lambda: weight of the gradient penalty
  *   crit_repeats: number of times to update the critic per generator update - there are more details about this in the *Putting It All Together* section
  *   device: the device type

You will also load and transform the MNIST dataset to tensors.





```python
n_epochs = 100
z_dim = 64
display_step = 50
batch_size = 128
lr = 0.0002
beta_1 = 0.5
beta_2 = 0.999
c_lambda = 10
crit_repeats = 5
device = 'cuda'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataloader = DataLoader(
    MNIST('.', download=False, transform=transform),
    batch_size=batch_size,
    shuffle=True)
```

Then, you can initialize your generator, critic, and optimizers.


```python
gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
crit = Critic().to(device) 
crit_opt = torch.optim.Adam(crit.parameters(), lr=lr, betas=(beta_1, beta_2))

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
gen = gen.apply(weights_init)
crit = crit.apply(weights_init)

```

## Gradient Penalty
Calculating the gradient penalty can be broken into two functions: (1) compute the gradient with respect to the images and (2) compute the gradient penalty given the gradient.

You can start by getting the gradient. The gradient is computed by first creating a mixed image. This is done by weighing the fake and real image using epsilon and then adding them together. Once you have the intermediate image, you can get the critic's output on the image. Finally, you compute the gradient of the critic score's on the mixed images (output) with respect to the pixels of the mixed images (input). You will need to fill in the code to get the gradient wherever you see *None*. There is a test function in the next block for you to test your solution.


```python
# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_gradient
def get_gradient(crit, real, fake, epsilon):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = crit(mixed_images)
    
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        # Note: You need to take the gradient of outputs with respect to inputs.
        # This documentation may be useful, but it should not be necessary:
        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
        #### START CODE HERE ####
        inputs=mixed_images,
        outputs=mixed_scores,
        #### END CODE HERE ####
        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient

```


```python
# UNIT TEST
# DO NOT MODIFY THIS
def test_get_gradient(image_shape):
    real = torch.randn(*image_shape, device=device) + 1
    fake = torch.randn(*image_shape, device=device) - 1
    epsilon_shape = [1 for _ in image_shape]
    epsilon_shape[0] = image_shape[0]
    epsilon = torch.rand(epsilon_shape, device=device).requires_grad_()
    gradient = get_gradient(crit, real, fake, epsilon)
    assert tuple(gradient.shape) == image_shape
    assert gradient.max() > 0
    assert gradient.min() < 0
    return gradient

gradient = test_get_gradient((256, 1, 28, 28))
print("Success!")
```

    Success!


The second function you need to complete is to compute the gradient penalty given the gradient. First, you calculate the magnitude of each image's gradient. The magnitude of a gradient is also called the norm. Then, you calculate the penalty by squaring the distance between each magnitude and the ideal norm of 1 and taking the mean of all the squared distances.

Again, you will need to fill in the code wherever you see *None*. There are hints below that you can view if you need help and there is a test function in the next block for you to test your solution.

<details>

<summary>
<font size="3" color="green">
<b>Optional hints for <code><font size="4">gradient_penalty</font></code></b>
</font>
</summary>


1.   Make sure you take the mean at the end.
2.   Note that the magnitude of each gradient has already been calculated for you.

</details>



```python
# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: gradient_penalty
def gradient_penalty(gradient):
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    '''
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)
    
    # Penalize the mean squared distance of the gradient norms from 1
    #### START CODE HERE ####
    penalty = torch.mean((gradient_norm-1)**2)
    #### END CODE HERE ####
    return penalty
```


```python
# UNIT TEST
def test_gradient_penalty(image_shape):
    bad_gradient = torch.zeros(*image_shape)
    bad_gradient_penalty = gradient_penalty(bad_gradient)
    assert torch.isclose(bad_gradient_penalty, torch.tensor(1.))

    image_size = torch.prod(torch.Tensor(image_shape[1:]))
    good_gradient = torch.ones(*image_shape) / torch.sqrt(image_size)
    good_gradient_penalty = gradient_penalty(good_gradient)
    assert torch.isclose(good_gradient_penalty, torch.tensor(0.))

    random_gradient = test_get_gradient(image_shape)
    random_gradient_penalty = gradient_penalty(random_gradient)
    assert torch.abs(random_gradient_penalty - 1) < 0.1

test_gradient_penalty((256, 1, 28, 28))
print("Success!")
```

    Success!


## Losses
Next, you need to calculate the loss for the generator and the critic.

For the generator, the loss is calculated by maximizing the critic's prediction on the generator's fake images. The argument has the scores for all fake images in the batch, but you will use the mean of them.

There are optional hints below and a test function in the next block for you to test your solution.

<details><summary><font size="3" color="green"><b>Optional hints for <code><font size="4">get_gen_loss</font></code></b></font></summary>

1. This can be written in one line.
2. This is the negative of the mean of the critic's scores.

</details>


```python
# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_gen_loss
def get_gen_loss(crit_fake_pred):
    '''
    Return the loss of a generator given the critic's scores of the generator's fake images.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    '''
    #### START CODE HERE ####
    gen_loss = -1 * torch.mean(crit_fake_pred)
    #### END CODE HERE ####
    return gen_loss
```


```python
# UNIT TEST
assert torch.isclose(
    get_gen_loss(torch.tensor(1.)), torch.tensor(-1.0)
)

assert torch.isclose(
    get_gen_loss(torch.rand(10000)), torch.tensor(-0.5), 0.05
)

print("Success!")
```

    Success!


For the critic, the loss is calculated by maximizing the distance between the critic's predictions on the real images and the predictions on the fake images while also adding a gradient penalty. The gradient penalty is weighed according to lambda. The arguments are the scores for all the images in the batch, and you will use the mean of them.

There are hints below if you get stuck and a test function in the next block for you to test your solution.

<details><summary><font size="3" color="green"><b>Optional hints for <code><font size="4">get_crit_loss</font></code></b></font></summary>

1. The higher the mean fake score, the higher the critic's loss is.
2. What does this suggest about the mean real score?
3. The higher the gradient penalty, the higher the critic's loss is, proportional to lambda.


</details>



```python
# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_crit_loss
def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    '''
    Return the loss of a critic given the critic's scores for fake and real images,
    the gradient penalty, and gradient penalty weight.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
        crit_real_pred: the critic's scores of the real images
        gp: the unweighted gradient penalty
        c_lambda: the current weight of the gradient penalty 
    Returns:
        crit_loss: a scalar for the critic's loss, accounting for the relevant factors
    '''
    #### START CODE HERE ####
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp
    #### END CODE HERE ####
    return crit_loss
```


```python
# UNIT TEST
assert torch.isclose(
    get_crit_loss(torch.tensor(1.), torch.tensor(2.), torch.tensor(3.), 0.1),
    torch.tensor(-0.7)
)
assert torch.isclose(
    get_crit_loss(torch.tensor(20.), torch.tensor(-20.), torch.tensor(2.), 10),
    torch.tensor(60.)
)

print("Success!")
```

    Success!


## Putting It All Together
Before you put everything together, there are a few things to note.
1.   Even on GPU, the **training will run more slowly** than previous labs because the gradient penalty requires you to compute the gradient of a gradient -- this means potentially a few minutes per epoch! For best results, run this for as long as you can while on GPU.
2.   One important difference from earlier versions is that you will **update the critic multiple times** every time you update the generator This helps prevent the generator from overpowering the critic. Sometimes, you might see the reverse, with the generator updated more times than the critic. This depends on architectural (e.g. the depth and width of the network) and algorithmic choices (e.g. which loss you're using). 
3.   WGAN-GP isn't necessarily meant to improve overall performance of a GAN, but just **increases stability** and avoids mode collapse. In general, a WGAN will be able to train in a much more stable way than the vanilla DCGAN from last assignment, though it will generally run a bit slower. You should also be able to train your model for more epochs without it collapsing.


<!-- Once again, be warned that this runs very slowly on a CPU. One way to run this more quickly is to download the .ipynb and upload it to Google Drive, then open it with Google Colab and make the runtime type GPU and replace
`device = "cpu"`
with
`device = "cuda"`
and make sure that your `get_noise` function uses the right device.  -->

Here is a snapshot of what your WGAN-GP outputs should resemble:
![MNIST Digits Progression](MNIST_WGAN_Progression.png)


```python
import matplotlib.pyplot as plt

cur_step = 0
generator_losses = []
critic_losses = []
for epoch in range(n_epochs):
    # Dataloader returns the batches
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.to(device)

        mean_iteration_critic_loss = 0
        for _ in range(crit_repeats):
            ### Update critic ###
            crit_opt.zero_grad()
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
            crit_fake_pred = crit(fake.detach())
            crit_real_pred = crit(real)

            epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
            gradient = get_gradient(crit, real, fake.detach(), epsilon)
            gp = gradient_penalty(gradient)
            crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)

            # Keep track of the average critic loss in this batch
            mean_iteration_critic_loss += crit_loss.item() / crit_repeats
            # Update gradients
            crit_loss.backward(retain_graph=True)
            # Update optimizer
            crit_opt.step()
        critic_losses += [mean_iteration_critic_loss]

        ### Update generator ###
        gen_opt.zero_grad()
        fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
        fake_2 = gen(fake_noise_2)
        crit_fake_pred = crit(fake_2)
        
        gen_loss = get_gen_loss(crit_fake_pred)
        gen_loss.backward()

        # Update the weights
        gen_opt.step()

        # Keep track of the average generator loss
        generator_losses += [gen_loss.item()]

        ### Visualization code ###
        if cur_step % display_step == 0 and cur_step > 0:
            gen_mean = sum(generator_losses[-display_step:]) / display_step
            crit_mean = sum(critic_losses[-display_step:]) / display_step
            print(f"Step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")
            show_tensor_images(fake)
            show_tensor_images(real)
            step_bins = 20
            num_examples = (len(generator_losses) // step_bins) * step_bins
            plt.plot(
                range(num_examples // step_bins), 
                torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Generator Loss"
            )
            plt.plot(
                range(num_examples // step_bins), 
                torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Critic Loss"
            )
            plt.legend()
            plt.show()

        cur_step += 1

```


    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 50: Generator loss: -0.130625586041715, critic loss: 2.0482199134826655



![png](output_26_2.png)



![png](output_26_3.png)



![png](output_26_4.png)


    Step 100: Generator loss: 1.5752828937768937, critic loss: -3.0023532475233083



![png](output_26_6.png)



![png](output_26_7.png)



![png](output_26_8.png)


    Step 150: Generator loss: 1.3426154951006175, critic loss: -12.510224712371826



![png](output_26_10.png)



![png](output_26_11.png)



![png](output_26_12.png)


    Step 200: Generator loss: -0.17739903200417756, critic loss: -31.71788439178466



![png](output_26_14.png)



![png](output_26_15.png)



![png](output_26_16.png)


    Step 250: Generator loss: -0.2625879709050059, critic loss: -69.76223231506347



![png](output_26_18.png)



![png](output_26_19.png)



![png](output_26_20.png)


    Step 300: Generator loss: 2.5600806605815887, critic loss: -122.41021563720705



![png](output_26_22.png)



![png](output_26_23.png)



![png](output_26_24.png)


    Step 350: Generator loss: 5.512920761108399, critic loss: -184.45037261962895



![png](output_26_26.png)



![png](output_26_27.png)



![png](output_26_28.png)


    Step 400: Generator loss: 7.258047590255737, critic loss: -251.1092144165039



![png](output_26_30.png)



![png](output_26_31.png)



![png](output_26_32.png)


    Step 450: Generator loss: 6.03194568157196, critic loss: -304.80175726318356



![png](output_26_34.png)



![png](output_26_35.png)



![png](output_26_36.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 500: Generator loss: 3.6002049080282448, critic loss: -352.9464402465821



![png](output_26_40.png)



![png](output_26_41.png)



![png](output_26_42.png)


    Step 550: Generator loss: -3.286598073244095, critic loss: -394.18788391113276



![png](output_26_44.png)



![png](output_26_45.png)



![png](output_26_46.png)


    Step 600: Generator loss: -9.892032408118247, critic loss: -408.9308211669924



![png](output_26_48.png)



![png](output_26_49.png)



![png](output_26_50.png)


    Step 650: Generator loss: 2.234330582618713, critic loss: -449.5412332763673



![png](output_26_52.png)



![png](output_26_53.png)



![png](output_26_54.png)


    Step 700: Generator loss: 4.286422241926193, critic loss: -538.523931274414



![png](output_26_56.png)



![png](output_26_57.png)



![png](output_26_58.png)


    Step 750: Generator loss: 2.36698536247015, critic loss: -582.8566899414063



![png](output_26_60.png)



![png](output_26_61.png)



![png](output_26_62.png)


    Step 800: Generator loss: 5.028563871085644, critic loss: -600.5233717041015



![png](output_26_64.png)



![png](output_26_65.png)



![png](output_26_66.png)


    Step 850: Generator loss: -28.197967977523803, critic loss: -503.33571166992203



![png](output_26_68.png)



![png](output_26_69.png)



![png](output_26_70.png)


    Step 900: Generator loss: 30.619763507843018, critic loss: -662.9892854003905



![png](output_26_72.png)



![png](output_26_73.png)



![png](output_26_74.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 950: Generator loss: -14.387984209060669, critic loss: -600.8140206298829



![png](output_26_78.png)



![png](output_26_79.png)



![png](output_26_80.png)


    Step 1000: Generator loss: -14.37071210861206, critic loss: -565.9937134399413



![png](output_26_82.png)



![png](output_26_83.png)



![png](output_26_84.png)


    Step 1050: Generator loss: -56.04518758535385, critic loss: -443.1736217651367



![png](output_26_86.png)



![png](output_26_87.png)



![png](output_26_88.png)


    Step 1100: Generator loss: -98.75891690731049, critic loss: -430.84557601928714



![png](output_26_90.png)



![png](output_26_91.png)



![png](output_26_92.png)


    Step 1150: Generator loss: -68.12173253536224, critic loss: -281.10739573669434



![png](output_26_94.png)



![png](output_26_95.png)



![png](output_26_96.png)


    Step 1200: Generator loss: -52.88708537578583, critic loss: -395.157559829712



![png](output_26_98.png)



![png](output_26_99.png)



![png](output_26_100.png)


    Step 1250: Generator loss: -91.56160719037057, critic loss: -376.53576583862304



![png](output_26_102.png)



![png](output_26_103.png)



![png](output_26_104.png)


    Step 1300: Generator loss: -34.34722891330719, critic loss: -422.9345966796874



![png](output_26_106.png)



![png](output_26_107.png)



![png](output_26_108.png)


    Step 1350: Generator loss: -49.59554964184761, critic loss: -254.28388545227054



![png](output_26_110.png)



![png](output_26_111.png)



![png](output_26_112.png)


    Step 1400: Generator loss: -92.74991739034652, critic loss: -322.69940316772465



![png](output_26_114.png)



![png](output_26_115.png)



![png](output_26_116.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 1450: Generator loss: -76.68635872840882, critic loss: -284.67132684326174



![png](output_26_120.png)



![png](output_26_121.png)



![png](output_26_122.png)


    Step 1500: Generator loss: -25.582783927321433, critic loss: 71.09894213104248



![png](output_26_124.png)



![png](output_26_125.png)



![png](output_26_126.png)


    Step 1550: Generator loss: -19.180611877441407, critic loss: 89.26951387023927



![png](output_26_128.png)



![png](output_26_129.png)



![png](output_26_130.png)


    Step 1600: Generator loss: -21.41633312225342, critic loss: 58.318354133605965



![png](output_26_132.png)



![png](output_26_133.png)



![png](output_26_134.png)


    Step 1650: Generator loss: -17.984939575195312, critic loss: 38.66341965484619



![png](output_26_136.png)



![png](output_26_137.png)



![png](output_26_138.png)


    Step 1700: Generator loss: -15.772660121917724, critic loss: 24.178126739501945



![png](output_26_140.png)



![png](output_26_141.png)



![png](output_26_142.png)


    Step 1750: Generator loss: -13.84817283630371, critic loss: 10.237353134155274



![png](output_26_144.png)



![png](output_26_145.png)



![png](output_26_146.png)


    Step 1800: Generator loss: -15.630541343688964, critic loss: 2.731426147460938



![png](output_26_148.png)



![png](output_26_149.png)



![png](output_26_150.png)


    Step 1850: Generator loss: -15.788389959335326, critic loss: -9.738187232971189



![png](output_26_152.png)



![png](output_26_153.png)



![png](output_26_154.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 1900: Generator loss: -17.802727583646774, critic loss: -30.560255622863775



![png](output_26_158.png)



![png](output_26_159.png)



![png](output_26_160.png)


    Step 1950: Generator loss: -36.76341435313225, critic loss: -46.592250617980966



![png](output_26_162.png)



![png](output_26_163.png)



![png](output_26_164.png)


    Step 2000: Generator loss: 1.378508256971836, critic loss: 7.328655189514165



![png](output_26_166.png)



![png](output_26_167.png)



![png](output_26_168.png)


    Step 2050: Generator loss: -35.65433660149574, critic loss: -75.91738320159912



![png](output_26_170.png)



![png](output_26_171.png)



![png](output_26_172.png)


    Step 2100: Generator loss: -45.38937075138092, critic loss: -114.94485208129882



![png](output_26_174.png)



![png](output_26_175.png)



![png](output_26_176.png)


    Step 2150: Generator loss: -37.20313910841942, critic loss: -84.63100650024415



![png](output_26_178.png)



![png](output_26_179.png)



![png](output_26_180.png)


    Step 2200: Generator loss: -33.497978281974795, critic loss: -57.66513304138182



![png](output_26_182.png)



![png](output_26_183.png)



![png](output_26_184.png)


    Step 2250: Generator loss: -40.590220906734466, critic loss: -57.7543399963379



![png](output_26_186.png)



![png](output_26_187.png)



![png](output_26_188.png)


    Step 2300: Generator loss: -45.57214757442475, critic loss: -26.71737350463867



![png](output_26_190.png)



![png](output_26_191.png)



![png](output_26_192.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 2350: Generator loss: -49.56800471931696, critic loss: 62.63643272399902



![png](output_26_196.png)



![png](output_26_197.png)



![png](output_26_198.png)


    Step 2400: Generator loss: -40.142805969119074, critic loss: 134.09671130371092



![png](output_26_200.png)



![png](output_26_201.png)



![png](output_26_202.png)


    Step 2450: Generator loss: -37.12225740432739, critic loss: 50.17535716247559



![png](output_26_204.png)



![png](output_26_205.png)



![png](output_26_206.png)


    Step 2500: Generator loss: -20.0742778301239, critic loss: 29.375967590332028



![png](output_26_208.png)



![png](output_26_209.png)



![png](output_26_210.png)


    Step 2550: Generator loss: -23.771729710102083, critic loss: 61.34303070068359



![png](output_26_212.png)



![png](output_26_213.png)



![png](output_26_214.png)


    Step 2600: Generator loss: -29.474248645305632, critic loss: 65.76818814086916



![png](output_26_216.png)



![png](output_26_217.png)



![png](output_26_218.png)


    Step 2650: Generator loss: -29.469099984169006, critic loss: 53.77151232910157



![png](output_26_220.png)



![png](output_26_221.png)



![png](output_26_222.png)


    Step 2700: Generator loss: -30.539001564383508, critic loss: 41.819471229553216



![png](output_26_224.png)



![png](output_26_225.png)



![png](output_26_226.png)


    Step 2750: Generator loss: -12.60761087179184, critic loss: 65.21129431152342



![png](output_26_228.png)



![png](output_26_229.png)



![png](output_26_230.png)


    Step 2800: Generator loss: -19.30113144516945, critic loss: 53.79302299499513



![png](output_26_232.png)



![png](output_26_233.png)



![png](output_26_234.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 2850: Generator loss: -3.567881937623024, critic loss: 75.367941116333



![png](output_26_238.png)



![png](output_26_239.png)



![png](output_26_240.png)


    Step 2900: Generator loss: -2.514114396572113, critic loss: 49.798620986938474



![png](output_26_242.png)



![png](output_26_243.png)



![png](output_26_244.png)


    Step 2950: Generator loss: -3.8210985231399537, critic loss: 39.486391662597654



![png](output_26_246.png)



![png](output_26_247.png)



![png](output_26_248.png)


    Step 3000: Generator loss: -4.860496149063111, critic loss: 31.566427490234375



![png](output_26_250.png)



![png](output_26_251.png)



![png](output_26_252.png)


    Step 3050: Generator loss: -5.623742504119873, critic loss: 24.89963571929931



![png](output_26_254.png)



![png](output_26_255.png)



![png](output_26_256.png)


    Step 3100: Generator loss: -7.47586256980896, critic loss: 19.533668891906743



![png](output_26_258.png)



![png](output_26_259.png)



![png](output_26_260.png)


    Step 3150: Generator loss: -10.197880229949952, critic loss: 15.450349647521973



![png](output_26_262.png)



![png](output_26_263.png)



![png](output_26_264.png)


    Step 3200: Generator loss: -13.271646118164062, critic loss: 12.91025173950195



![png](output_26_266.png)



![png](output_26_267.png)



![png](output_26_268.png)


    Step 3250: Generator loss: -18.06815034866333, critic loss: 10.590590869903563



![png](output_26_270.png)



![png](output_26_271.png)



![png](output_26_272.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 3300: Generator loss: -22.49696334838867, critic loss: 8.506864353179932



![png](output_26_276.png)



![png](output_26_277.png)



![png](output_26_278.png)


    Step 3350: Generator loss: -24.29716339111328, critic loss: 7.339937934875487



![png](output_26_280.png)



![png](output_26_281.png)



![png](output_26_282.png)


    Step 3400: Generator loss: -22.458544616699218, critic loss: 7.298649055480958



![png](output_26_284.png)



![png](output_26_285.png)



![png](output_26_286.png)


    Step 3450: Generator loss: -22.97047954559326, critic loss: 5.332734235763549



![png](output_26_288.png)



![png](output_26_289.png)



![png](output_26_290.png)


    Step 3500: Generator loss: -24.226596336364747, critic loss: 2.239778391838074



![png](output_26_292.png)



![png](output_26_293.png)



![png](output_26_294.png)


    Step 3550: Generator loss: -23.401591606140137, critic loss: 0.48267895889282236



![png](output_26_296.png)



![png](output_26_297.png)



![png](output_26_298.png)


    Step 3600: Generator loss: -22.37114387512207, critic loss: -1.0119440689086914



![png](output_26_300.png)



![png](output_26_301.png)



![png](output_26_302.png)


    Step 3650: Generator loss: -20.749863777160645, critic loss: -2.051532866477966



![png](output_26_304.png)



![png](output_26_305.png)



![png](output_26_306.png)


    Step 3700: Generator loss: -19.580641708374024, critic loss: -2.5415955867767335



![png](output_26_308.png)



![png](output_26_309.png)



![png](output_26_310.png)


    Step 3750: Generator loss: -19.14418151855469, critic loss: -3.2073547258377073



![png](output_26_312.png)



![png](output_26_313.png)



![png](output_26_314.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 3800: Generator loss: -19.47313846588135, critic loss: -3.9072689170837407



![png](output_26_318.png)



![png](output_26_319.png)



![png](output_26_320.png)


    Step 3850: Generator loss: -18.16820295333862, critic loss: -4.354966764450072



![png](output_26_322.png)



![png](output_26_323.png)



![png](output_26_324.png)


    Step 3900: Generator loss: -14.58892889022827, critic loss: -4.483721103668213



![png](output_26_326.png)



![png](output_26_327.png)



![png](output_26_328.png)


    Step 3950: Generator loss: -12.37450351715088, critic loss: -4.985765390396118



![png](output_26_330.png)



![png](output_26_331.png)



![png](output_26_332.png)


    Step 4000: Generator loss: -8.806865434646607, critic loss: -5.262891201019288



![png](output_26_334.png)



![png](output_26_335.png)



![png](output_26_336.png)


    Step 4050: Generator loss: -4.1554736778140065, critic loss: -6.80510946273804



![png](output_26_338.png)



![png](output_26_339.png)



![png](output_26_340.png)


    Step 4100: Generator loss: 0.4794450068473816, critic loss: 0.10657841777801508



![png](output_26_342.png)



![png](output_26_343.png)



![png](output_26_344.png)


    Step 4150: Generator loss: -0.8636934538185597, critic loss: -7.4803892812728865



![png](output_26_346.png)



![png](output_26_347.png)



![png](output_26_348.png)


    Step 4200: Generator loss: 1.5017413572967053, critic loss: -12.641745088577268



![png](output_26_350.png)



![png](output_26_351.png)



![png](output_26_352.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 4250: Generator loss: 4.5390826761722565, critic loss: -14.246752826690674



![png](output_26_356.png)



![png](output_26_357.png)



![png](output_26_358.png)


    Step 4300: Generator loss: 8.234810752272606, critic loss: -15.560261985778805



![png](output_26_360.png)



![png](output_26_361.png)



![png](output_26_362.png)


    Step 4350: Generator loss: 9.623318338394165, critic loss: -14.481831627845763



![png](output_26_364.png)



![png](output_26_365.png)



![png](output_26_366.png)


    Step 4400: Generator loss: 12.433157618045806, critic loss: -14.747471801757811



![png](output_26_368.png)



![png](output_26_369.png)



![png](output_26_370.png)


    Step 4450: Generator loss: 11.712206230163574, critic loss: -19.748798114776616



![png](output_26_372.png)



![png](output_26_373.png)



![png](output_26_374.png)


    Step 4500: Generator loss: 13.647397565841676, critic loss: -20.224746063232427



![png](output_26_376.png)



![png](output_26_377.png)



![png](output_26_378.png)


    Step 4550: Generator loss: 15.264295654296875, critic loss: -19.279802787780763



![png](output_26_380.png)



![png](output_26_381.png)



![png](output_26_382.png)


    Step 4600: Generator loss: 17.946443481445314, critic loss: -21.854390754699708



![png](output_26_384.png)



![png](output_26_385.png)



![png](output_26_386.png)


    Step 4650: Generator loss: 20.012934427261353, critic loss: -19.10390859985352



![png](output_26_388.png)



![png](output_26_389.png)



![png](output_26_390.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 4700: Generator loss: 22.646349277496338, critic loss: -23.769650470733644



![png](output_26_394.png)



![png](output_26_395.png)



![png](output_26_396.png)


    Step 4750: Generator loss: 25.913204936981202, critic loss: -17.469927989006045



![png](output_26_398.png)



![png](output_26_399.png)



![png](output_26_400.png)


    Step 4800: Generator loss: 26.62237548828125, critic loss: -22.334841453552237



![png](output_26_402.png)



![png](output_26_403.png)



![png](output_26_404.png)


    Step 4850: Generator loss: 29.517843589782714, critic loss: -18.25146270465851



![png](output_26_406.png)



![png](output_26_407.png)



![png](output_26_408.png)


    Step 4900: Generator loss: 29.367914772033693, critic loss: -21.81997471809387



![png](output_26_410.png)



![png](output_26_411.png)



![png](output_26_412.png)


    Step 4950: Generator loss: 30.22893627166748, critic loss: -24.126487655639654



![png](output_26_414.png)



![png](output_26_415.png)



![png](output_26_416.png)


    Step 5000: Generator loss: 33.48539505004883, critic loss: -21.47382785987854



![png](output_26_418.png)



![png](output_26_419.png)



![png](output_26_420.png)


    Step 5050: Generator loss: 33.50176837921143, critic loss: -21.960488512039184



![png](output_26_422.png)



![png](output_26_423.png)



![png](output_26_424.png)


    Step 5100: Generator loss: 33.3051212310791, critic loss: -23.076463254928584



![png](output_26_426.png)



![png](output_26_427.png)



![png](output_26_428.png)


    Step 5150: Generator loss: 34.96341821670532, critic loss: -14.060136177062986



![png](output_26_430.png)



![png](output_26_431.png)



![png](output_26_432.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 5200: Generator loss: 35.1782275390625, critic loss: -23.323828798294077



![png](output_26_436.png)



![png](output_26_437.png)



![png](output_26_438.png)


    Step 5250: Generator loss: 35.48184192657471, critic loss: -23.389818849563593



![png](output_26_440.png)



![png](output_26_441.png)



![png](output_26_442.png)


    Step 5300: Generator loss: 36.089873695373534, critic loss: -23.729705904006963



![png](output_26_444.png)



![png](output_26_445.png)



![png](output_26_446.png)


    Step 5350: Generator loss: 40.088353385925295, critic loss: -17.152549465179444



![png](output_26_448.png)



![png](output_26_449.png)



![png](output_26_450.png)


    Step 5400: Generator loss: 37.73068225860596, critic loss: -24.30978824424744



![png](output_26_452.png)



![png](output_26_453.png)



![png](output_26_454.png)


    Step 5450: Generator loss: 40.197304801940916, critic loss: -20.49822038173675



![png](output_26_456.png)



![png](output_26_457.png)



![png](output_26_458.png)


    Step 5500: Generator loss: 39.044875869750975, critic loss: -24.812960309982294



![png](output_26_460.png)



![png](output_26_461.png)



![png](output_26_462.png)


    Step 5550: Generator loss: 38.94231884002686, critic loss: -22.015111637115478



![png](output_26_464.png)



![png](output_26_465.png)



![png](output_26_466.png)


    Step 5600: Generator loss: 44.46220054626465, critic loss: -21.016517791748043



![png](output_26_468.png)



![png](output_26_469.png)



![png](output_26_470.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 5650: Generator loss: 43.453633861541746, critic loss: -5.379834325790407



![png](output_26_474.png)



![png](output_26_475.png)



![png](output_26_476.png)


    Step 5700: Generator loss: 41.61182655334473, critic loss: -19.252753385543826



![png](output_26_478.png)



![png](output_26_479.png)



![png](output_26_480.png)


    Step 5750: Generator loss: 39.00077779769897, critic loss: -18.975539939880367



![png](output_26_482.png)



![png](output_26_483.png)



![png](output_26_484.png)


    Step 5800: Generator loss: 40.19796318054199, critic loss: -21.084398067474368



![png](output_26_486.png)



![png](output_26_487.png)



![png](output_26_488.png)


    Step 5850: Generator loss: 39.48652278900146, critic loss: -21.816018103599546



![png](output_26_490.png)



![png](output_26_491.png)



![png](output_26_492.png)


    Step 5900: Generator loss: 43.040359840393066, critic loss: -4.251242648601536



![png](output_26_494.png)



![png](output_26_495.png)



![png](output_26_496.png)


    Step 5950: Generator loss: 39.37861522674561, critic loss: -8.06414608860016



![png](output_26_498.png)



![png](output_26_499.png)



![png](output_26_500.png)


    Step 6000: Generator loss: 40.33249862670898, critic loss: -19.000110496520996



![png](output_26_502.png)



![png](output_26_503.png)



![png](output_26_504.png)


    Step 6050: Generator loss: 39.02000717163086, critic loss: -23.316631217956537



![png](output_26_506.png)



![png](output_26_507.png)



![png](output_26_508.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 6100: Generator loss: 41.63564712524414, critic loss: -19.211717683792116



![png](output_26_512.png)



![png](output_26_513.png)



![png](output_26_514.png)


    Step 6150: Generator loss: 37.57478622436523, critic loss: -22.515522434234622



![png](output_26_516.png)



![png](output_26_517.png)



![png](output_26_518.png)


    Step 6200: Generator loss: 38.973279304504395, critic loss: -20.36654423236847



![png](output_26_520.png)



![png](output_26_521.png)



![png](output_26_522.png)


    Step 6250: Generator loss: 42.5930033493042, critic loss: 0.774031168460846



![png](output_26_524.png)



![png](output_26_525.png)



![png](output_26_526.png)


    Step 6300: Generator loss: 40.116243019104004, critic loss: -3.9722281308174137



![png](output_26_528.png)



![png](output_26_529.png)



![png](output_26_530.png)


    Step 6350: Generator loss: 31.973823890686035, critic loss: -22.856053272247312



![png](output_26_532.png)



![png](output_26_533.png)



![png](output_26_534.png)


    Step 6400: Generator loss: 39.27843086242676, critic loss: -14.907237329483031



![png](output_26_536.png)



![png](output_26_537.png)



![png](output_26_538.png)


    Step 6450: Generator loss: 35.6552038192749, critic loss: -21.077704190254202



![png](output_26_540.png)



![png](output_26_541.png)



![png](output_26_542.png)


    Step 6500: Generator loss: 34.32813472747803, critic loss: -22.576587754249577



![png](output_26_544.png)



![png](output_26_545.png)



![png](output_26_546.png)


    Step 6550: Generator loss: 34.12170143127442, critic loss: -19.604217237472533



![png](output_26_548.png)



![png](output_26_549.png)



![png](output_26_550.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 6600: Generator loss: 30.23227699279785, critic loss: -22.351317975044253



![png](output_26_554.png)



![png](output_26_555.png)



![png](output_26_556.png)


    Step 6650: Generator loss: 30.88545967102051, critic loss: -21.57576067066193



![png](output_26_558.png)



![png](output_26_559.png)



![png](output_26_560.png)


    Step 6700: Generator loss: 33.113734016418455, critic loss: -6.984330187320714



![png](output_26_562.png)



![png](output_26_563.png)



![png](output_26_564.png)


    Step 6750: Generator loss: 32.86422634124756, critic loss: -2.2865829181671136



![png](output_26_566.png)



![png](output_26_567.png)



![png](output_26_568.png)


    Step 6800: Generator loss: 23.385224514007568, critic loss: -11.43333789920807



![png](output_26_570.png)



![png](output_26_571.png)



![png](output_26_572.png)


    Step 6850: Generator loss: 27.539349632263182, critic loss: -20.597385007858275



![png](output_26_574.png)



![png](output_26_575.png)



![png](output_26_576.png)


    Step 6900: Generator loss: 29.424481658935548, critic loss: -21.671156100273123



![png](output_26_578.png)



![png](output_26_579.png)



![png](output_26_580.png)


    Step 6950: Generator loss: 27.778595561981202, critic loss: -17.006155375480652



![png](output_26_582.png)



![png](output_26_583.png)



![png](output_26_584.png)


    Step 7000: Generator loss: 23.619869956970216, critic loss: -23.360183818817134



![png](output_26_586.png)



![png](output_26_587.png)



![png](output_26_588.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 7050: Generator loss: 25.772095584869383, critic loss: -18.141080478668215



![png](output_26_592.png)



![png](output_26_593.png)



![png](output_26_594.png)


    Step 7100: Generator loss: 22.349760551452636, critic loss: -25.013060353279105



![png](output_26_596.png)



![png](output_26_597.png)



![png](output_26_598.png)


    Step 7150: Generator loss: 26.92783639907837, critic loss: -21.484167627334596



![png](output_26_600.png)



![png](output_26_601.png)



![png](output_26_602.png)


    Step 7200: Generator loss: 20.354009685516356, critic loss: -24.110806318283075



![png](output_26_604.png)



![png](output_26_605.png)



![png](output_26_606.png)


    Step 7250: Generator loss: 32.80087413787842, critic loss: -3.961628347396849



![png](output_26_608.png)



![png](output_26_609.png)



![png](output_26_610.png)


    Step 7300: Generator loss: 26.83991563796997, critic loss: -9.583222163200377



![png](output_26_612.png)



![png](output_26_613.png)



![png](output_26_614.png)


    Step 7350: Generator loss: 28.21845782279968, critic loss: -16.144121856689456



![png](output_26_616.png)



![png](output_26_617.png)



![png](output_26_618.png)


    Step 7400: Generator loss: 27.775209617614745, critic loss: -15.779850683212278



![png](output_26_620.png)



![png](output_26_621.png)



![png](output_26_622.png)


    Step 7450: Generator loss: 25.176293182373048, critic loss: -12.408704935073851



![png](output_26_624.png)



![png](output_26_625.png)



![png](output_26_626.png)


    Step 7500: Generator loss: 23.57172414779663, critic loss: -15.641278692245482



![png](output_26_628.png)



![png](output_26_629.png)



![png](output_26_630.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 7550: Generator loss: 17.599635200500487, critic loss: -21.763309240341187



![png](output_26_634.png)



![png](output_26_635.png)



![png](output_26_636.png)


    Step 7600: Generator loss: 24.786430950164796, critic loss: -14.373321559906008



![png](output_26_638.png)



![png](output_26_639.png)



![png](output_26_640.png)


    Step 7650: Generator loss: 23.14211502075195, critic loss: -14.855736728668214



![png](output_26_642.png)



![png](output_26_643.png)



![png](output_26_644.png)


    Step 7700: Generator loss: 22.389617977142333, critic loss: -12.068287328720094



![png](output_26_646.png)



![png](output_26_647.png)



![png](output_26_648.png)


    Step 7750: Generator loss: 20.77182255268097, critic loss: -18.345362041473383



![png](output_26_650.png)



![png](output_26_651.png)



![png](output_26_652.png)


    Step 7800: Generator loss: 24.508546743392944, critic loss: -14.03289521789551



![png](output_26_654.png)



![png](output_26_655.png)



![png](output_26_656.png)


    Step 7850: Generator loss: 20.34327045440674, critic loss: -14.223609367370605



![png](output_26_658.png)



![png](output_26_659.png)



![png](output_26_660.png)


    Step 7900: Generator loss: 20.172256059646607, critic loss: -16.261908998489375



![png](output_26_662.png)



![png](output_26_663.png)



![png](output_26_664.png)


    Step 7950: Generator loss: 19.5455418920517, critic loss: -17.902805373191836



![png](output_26_666.png)



![png](output_26_667.png)



![png](output_26_668.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 8000: Generator loss: 24.52613558769226, critic loss: -16.643973132133482



![png](output_26_672.png)



![png](output_26_673.png)



![png](output_26_674.png)


    Step 8050: Generator loss: 22.989151821136474, critic loss: -14.596836254119872



![png](output_26_676.png)



![png](output_26_677.png)



![png](output_26_678.png)


    Step 8100: Generator loss: 19.419717893600463, critic loss: -6.377537971496583



![png](output_26_680.png)



![png](output_26_681.png)



![png](output_26_682.png)


    Step 8150: Generator loss: 21.13484073340893, critic loss: -23.874659317016594



![png](output_26_684.png)



![png](output_26_685.png)



![png](output_26_686.png)


    Step 8200: Generator loss: 13.845207160711288, critic loss: -16.027204204082494



![png](output_26_688.png)



![png](output_26_689.png)



![png](output_26_690.png)


    Step 8250: Generator loss: 28.48545162200928, critic loss: -10.08053355407715



![png](output_26_692.png)



![png](output_26_693.png)



![png](output_26_694.png)


    Step 8300: Generator loss: 17.40008955001831, critic loss: -17.666810785293574



![png](output_26_696.png)



![png](output_26_697.png)



![png](output_26_698.png)


    Step 8350: Generator loss: 23.694710960388182, critic loss: -14.392058992385865



![png](output_26_700.png)



![png](output_26_701.png)



![png](output_26_702.png)


    Step 8400: Generator loss: 24.85252607345581, critic loss: -14.096645816802981



![png](output_26_704.png)



![png](output_26_705.png)



![png](output_26_706.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 8450: Generator loss: 22.220264134407042, critic loss: -15.613581662654878



![png](output_26_710.png)



![png](output_26_711.png)



![png](output_26_712.png)


    Step 8500: Generator loss: 24.358921003341674, critic loss: -16.196997099876402



![png](output_26_714.png)



![png](output_26_715.png)



![png](output_26_716.png)


    Step 8550: Generator loss: 18.810594147443773, critic loss: -16.751674543380737



![png](output_26_718.png)



![png](output_26_719.png)



![png](output_26_720.png)


    Step 8600: Generator loss: 20.486098294258117, critic loss: -13.767835325717927



![png](output_26_722.png)



![png](output_26_723.png)



![png](output_26_724.png)


    Step 8650: Generator loss: 18.43568805217743, critic loss: -17.23280039596557



![png](output_26_726.png)



![png](output_26_727.png)



![png](output_26_728.png)


    Step 8700: Generator loss: 23.48956495285034, critic loss: -15.100981256484989



![png](output_26_730.png)



![png](output_26_731.png)



![png](output_26_732.png)


    Step 8750: Generator loss: 18.808236775398253, critic loss: -18.33817656612397



![png](output_26_734.png)



![png](output_26_735.png)



![png](output_26_736.png)


    Step 8800: Generator loss: 22.7076859664917, critic loss: -14.516003434181219



![png](output_26_738.png)



![png](output_26_739.png)



![png](output_26_740.png)


    Step 8850: Generator loss: 19.985014972686766, critic loss: -12.705390045166014



![png](output_26_742.png)



![png](output_26_743.png)



![png](output_26_744.png)


    Step 8900: Generator loss: 23.677678928375244, critic loss: -1.8468135862350474



![png](output_26_746.png)



![png](output_26_747.png)



![png](output_26_748.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 8950: Generator loss: 22.315099449157714, critic loss: -13.75334200668335



![png](output_26_752.png)



![png](output_26_753.png)



![png](output_26_754.png)


    Step 9000: Generator loss: 15.440033779144287, critic loss: -15.37245401477814



![png](output_26_756.png)



![png](output_26_757.png)



![png](output_26_758.png)


    Step 9050: Generator loss: 17.52918846130371, critic loss: -16.988115967750552



![png](output_26_760.png)



![png](output_26_761.png)



![png](output_26_762.png)


    Step 9100: Generator loss: 24.029694938659667, critic loss: -14.035324464797975



![png](output_26_764.png)



![png](output_26_765.png)



![png](output_26_766.png)


    Step 9150: Generator loss: 19.20696735858917, critic loss: -17.501260425567626



![png](output_26_768.png)



![png](output_26_769.png)



![png](output_26_770.png)


    Step 9200: Generator loss: 24.684222450256346, critic loss: -11.758951999187468



![png](output_26_772.png)



![png](output_26_773.png)



![png](output_26_774.png)


    Step 9250: Generator loss: 20.849301586151125, critic loss: -16.71927216768264



![png](output_26_776.png)



![png](output_26_777.png)



![png](output_26_778.png)


    Step 9300: Generator loss: 25.649389533996583, critic loss: 8.44355576276779



![png](output_26_780.png)



![png](output_26_781.png)



![png](output_26_782.png)


    Step 9350: Generator loss: 23.175159873962404, critic loss: -2.4225587985515595



![png](output_26_784.png)



![png](output_26_785.png)



![png](output_26_786.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 9400: Generator loss: 27.627716369628907, critic loss: -3.3068381681442265



![png](output_26_790.png)



![png](output_26_791.png)



![png](output_26_792.png)


    Step 9450: Generator loss: 20.008814878463745, critic loss: -7.4836717720031745



![png](output_26_794.png)



![png](output_26_795.png)



![png](output_26_796.png)


    Step 9500: Generator loss: 23.42931803703308, critic loss: -12.898664532661433



![png](output_26_798.png)



![png](output_26_799.png)



![png](output_26_800.png)


    Step 9550: Generator loss: 24.84683762550354, critic loss: -11.401532751083373



![png](output_26_802.png)



![png](output_26_803.png)



![png](output_26_804.png)


    Step 9600: Generator loss: 23.12930522918701, critic loss: -14.097914521217346



![png](output_26_806.png)



![png](output_26_807.png)



![png](output_26_808.png)


    Step 9650: Generator loss: 13.2649853849411, critic loss: -17.492697520256044



![png](output_26_810.png)



![png](output_26_811.png)



![png](output_26_812.png)


    Step 9700: Generator loss: 26.271662311553953, critic loss: -1.7209728460311897



![png](output_26_814.png)



![png](output_26_815.png)



![png](output_26_816.png)



```python

```
