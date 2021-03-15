# Spectrally Normalized Generative Adversarial Networks (SN-GAN)

*Please note that this is an optional notebook, meant to introduce more advanced concepts if you're up for a challenge, so don't worry if you don't completely follow!*

**Goals**

In this notebook, you'll learn about and implement **spectral normalization**, a weight normalization technique to stabilize the training of the discriminator, as proposed in [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957) (Miyato et al. 2018).

**Background**

As its name suggests, SN-GAN normalizes the weight matrices in the discriminator by their corresponding [spectral norm](https://calculus.subwiki.org/wiki/Spectral_norm#:~:text=The%20spectral%20norm%20of%20a,where%20denotes%20the%20Euclidean%20norm.), which helps control the Lipschitz constant of the discriminator. As you have learned with WGAN, [Lipschitz continuity](https://en.wikipedia.org/wiki/Lipschitz_continuity) is important in ensuring the boundedness of the optimal discriminator. In the WGAN case, this makes it so that the underlying W-loss function for the discriminator (or more precisely, the critic) is valid.

As a result, spectral normalization helps improve stability and avoid vanishing gradient problems, such as mode collapse.

## Spectral Norm

Notationally, the spectral norm of a matrix $W$ is typically represented as $\sigma(W)$. For neural network purposes, this $W$ matrix represents a weight matrix in one of the network's layers. The spectral norm of a matrix is the matrix's largest singular value, which can be obtained via singular value decomposition (SVD).

**A Quick Refresher on SVD**

SVD is a generalization of [eigendecomposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix) and is used to factorize a matrix as $W = U\Sigma V^\top$, where $U, V$ are orthogonal matrices and $\Sigma$ is a matrix of singular values on its diagonal. Note that $\Sigma$ doesn't have to be square.

\begin{align*}
    \Sigma = \begin{bmatrix}\sigma_1 & & \\ & \sigma_2 \\ & & \ddots \\ & & & \sigma_n\end{bmatrix}
\end{align*}

where $\sigma_1$ and $\sigma_n$ are the largest and smallest singular values, respectively. Intuitively, larger values correspond to larger amounts of stretching a matrix can apply to another vector. Following this notation, $\sigma(W) = \sigma_1$.

**Applying SVD to Spectral Normalization**

To spectrally normalize the weight matrix, you divide every value in the matrix by its spectral norm. As a result, a spectrally normalized matrix $\overline{W}_{SN}$ can be expressed as

\begin{align*}
  \overline{W}_{SN} = \dfrac{W}{\sigma(W)},
\end{align*}

In practice, computing the SVD of $W$ is expensive, so the authors of the SN-GAN paper do something very neat. They instead approximate the left and right singular vectors, $\tilde{u}$ and $\tilde{v}$ respectively, through power iteration such that $\sigma(W) \approx \tilde{u}^\top W\tilde{v}$.

Starting from randomly initialization, $\tilde{u}$ and $\tilde{v}$ are updated according to

\begin{align*}
  \tilde{u} &:= \dfrac{W^\top\tilde{u}}{||W^\top\tilde{u}||_2} \\
  \tilde{v} &:= \dfrac{W\tilde{v}}{||W\tilde{v}||_2}
\end{align*}

In practice, one round of iteration is sufficient to "achieve satisfactory performance" as per the authors.

Don't worry if you don't completely follow this! The algorithm is conveniently implemented as `torch.nn.utils.spectral_norm` in PyTorch, so as long as you get the general gist of how it might be useful and when to use it, then you're all set.


## A Bit of History on Spectral Normalization

This isn't the first time that spectral norm has been proposed in the context of deep learning models. There's a paper called [Spectral Norm Regularization for Improving the Generalizability of Deep Learning](https://arxiv.org/abs/1705.10941) (Yoshida et al. 2017) that proposes **spectral norm regularization**, which they showed to improve the generalizability of models by adding extra loss terms onto the loss function (just as L2 regularization and gradient penalty do!). These extra loss terms specifically penalize the spectral norm of the weights. You can think of this as *data-independent* regularization because the gradient with respect to $W$ isn't a function of the minibatch.

**Spectral normalization**, on the other hand, sets the spectral norm of the weight matrices to 1 -- it's a much harder constraint than adding a loss term, which is a form of "soft" regularization. As the authors show in the paper, you can think of spectral normalization as *data-dependent* regularization, since the gradient with respect to $W$ is dependent on the mini-batch statistics (shown in Section 2.1 of the [main paper](https://arxiv.org/pdf/1802.05957.pdf)). Spectral normalization essentially prevents the transformation of each layer
from becoming to sensitive in one direction and mitigates exploding gradients.

## DCGAN with Spectral Normalization

In rest of this notebook, you will walk through how to apply spectral normalization to DCGAN as an example, using your earlier DCGAN implementation. You can always add spectral normalization to your other models too.

Here, you start with the same setup and helper function, as you've seen before. 


```python
# Some setup
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0) # Set for our testing purposes, please do not change!

'''
Function for visualizing images: Given a tensor of images, number of images, and
size per image, plots and prints the images in an uniform grid.
'''
def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
```

### DCGAN Generator

Since spectral normalization is only applied to the matrices in the discriminator, the generator implementation is the same as the original.


```python
class Generator(nn.Module):
    '''
    Generator Class
    Values:
    z_dim: the dimension of the noise vector, a scalar
    im_chan: the number of channels of the output image, a scalar
            MNIST is black-and-white, so that's our default
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
        Function to return a sequence of operations corresponding to a generator block of the DCGAN, 
        corresponding to a transposed convolution, a batchnorm (except for in the last layer), and an activation
        Parameters:
        input_channels: how many channels the input feature representation has
        output_channels: how many channels the output feature representation should have
        kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
        stride: the stride of the convolution
        final_layer: whether we're on the final layer (affects activation and batchnorm)
        '''
        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else: # Final Layer
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def unsqueeze_noise(self, noise):
        '''
        Function for completing a forward pass of the Generator: Given a noise vector, 
        returns a copy of that noise with width and height = 1 and channels = z_dim.
        Parameters:
        noise: a noise tensor with dimensions (batch_size, z_dim)
        '''
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        '''
        Function for completing a forward pass of the Generator: Given a noise vector, 
        returns a generated image.
        Parameters:
        noise: a noise tensor with dimensions (batch_size, z_dim)
        '''
        x = self.unsqueeze_noise(noise)
        return self.gen(x)

def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating a noise vector: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
    n_samples: the number of samples in the batch, a scalar
    z_dim: the dimension of the noise vector, a scalar
    device: the device type
    '''
    return torch.randn(n_samples, z_dim, device=device)
```

### DCGAN Discriminator

For the discriminator, you can wrap each `nn.Conv2d` with `nn.utils.spectral_norm`. In the backend, this introduces parameters for $\tilde{u}$ and $\tilde{v}$ in addition to $W$ so that the $W_{SN}$ can be computed as $\tilde{u}^\top W\tilde{v}$ in runtime.

Pytorch also provides a `nn.utils.remove_spectral_norm` function, which collapses the 3 separate parameters into a single explicit $\overline{W}_{SN} := \tilde{u}^\top W\tilde{v}$. You should only apply this to your convolutional layers during inference to improve runtime speed.

It is important note that spectral norm does not eliminate the need for batch norm. Spectral norm affects the weights of each layer, while batch norm affects the activations of each layer. You can see both in a discriminator architecture, but you can also see just one of them. Hope this is something you have fun experimenting with!


```python
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
    im_chan: the number of channels of the output image, a scalar
            MNIST is black-and-white (1 channel), so that's our default.
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
        Function to return a sequence of operations corresponding to a discriminator block of the DCGAN, 
        corresponding to a convolution, a batchnorm (except for in the last layer), and an activation
        Parameters:
        input_channels: how many channels the input feature representation has
        output_channels: how many channels the output feature representation should have
        kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
        stride: the stride of the convolution
        final_layer: whether we're on the final layer (affects activation and batchnorm)
        '''
        
        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(input_channels, output_channels, kernel_size, stride)),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else: # Final Layer
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(input_channels, output_channels, kernel_size, stride)),
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the Discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
        image: a flattened image tensor with dimension (im_dim)
        '''
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)
```

### Training SN-DCGAN

You can now put everything together and train a spectrally normalized DCGAN! Here are all your parameters for initialization and optimization. 


```python
criterion = nn.BCEWithLogitsLoss()
n_epochs = 50
z_dim = 64
display_step = 500
batch_size = 128
# A learning rate of 0.0002 works well on DCGAN
lr = 0.0002

# These parameters control the optimizer's momentum, which you can read more about here:
# https://distill.pub/2017/momentum/ but you don’t need to worry about it for this course
beta_1 = 0.5 
beta_2 = 0.999
device = 'cuda'

# We tranform our image values to be between -1 and 1 (the range of the tanh activation)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataloader = DataLoader(
    MNIST(".", download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True)
```

Now, initialize the generator, the discriminator, and the optimizers.


```python
gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
disc = Discriminator().to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))

# We initialize the weights to the normal distribution
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

Finally, train the whole thing! And babysit those outputs :)


```python
cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
for epoch in range(n_epochs):
    # Dataloader returns the batches
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.to(device)

        ## Update Discriminator ##
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

        ## Update Generator ##
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


    Step 500: Generator loss: 0.6945227464437488, discriminator loss: 0.6962386629581455



![png](output_14_4.png)



![png](output_14_5.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 1000: Generator loss: 0.6933016135692599, discriminator loss: 0.6932045490741728



![png](output_14_9.png)



![png](output_14_10.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 1500: Generator loss: 0.6934151297807691, discriminator loss: 0.6932072701454163



![png](output_14_14.png)



![png](output_14_15.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 2000: Generator loss: 0.6932168151140211, discriminator loss: 0.6931754803657539



![png](output_14_19.png)



![png](output_14_20.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 2500: Generator loss: 0.6933222810029981, discriminator loss: 0.6931877311468129



![png](output_14_24.png)



![png](output_14_25.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 3000: Generator loss: 0.6932423771619797, discriminator loss: 0.6931750576496132



![png](output_14_29.png)



![png](output_14_30.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 3500: Generator loss: 0.6932030458450312, discriminator loss: 0.693170372128486



![png](output_14_34.png)



![png](output_14_35.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 4000: Generator loss: 0.6934343438148497, discriminator loss: 0.6931683244705202



![png](output_14_39.png)



![png](output_14_40.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 4500: Generator loss: 0.6933999778032303, discriminator loss: 0.6932130590677259



![png](output_14_44.png)



![png](output_14_45.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 5000: Generator loss: 0.6933219933509822, discriminator loss: 0.6931870919466017



![png](output_14_49.png)



![png](output_14_50.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 5500: Generator loss: 0.6931735270023349, discriminator loss: 0.6931608335971836



![png](output_14_54.png)



![png](output_14_55.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 6000: Generator loss: 0.6931522936820981, discriminator loss: 0.6931524305343625



![png](output_14_59.png)



![png](output_14_60.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 6500: Generator loss: 0.6931612219810482, discriminator loss: 0.6931549491882325



![png](output_14_64.png)



![png](output_14_65.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 7000: Generator loss: 0.6931605077981944, discriminator loss: 0.6931525099277497



![png](output_14_69.png)



![png](output_14_70.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 7500: Generator loss: 0.6931624424457549, discriminator loss: 0.693157375097274



![png](output_14_74.png)



![png](output_14_75.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 8000: Generator loss: 0.6931558561325077, discriminator loss: 0.6931499680280693



![png](output_14_81.png)



![png](output_14_82.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 8500: Generator loss: 0.6931617960929867, discriminator loss: 0.6931504572629932



![png](output_14_86.png)



![png](output_14_87.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 9000: Generator loss: 0.6937785160541537, discriminator loss: 0.693411365866661



![png](output_14_91.png)



![png](output_14_92.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 9500: Generator loss: 0.6931566863059999, discriminator loss: 0.6931490652561193



![png](output_14_96.png)



![png](output_14_97.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 10000: Generator loss: 0.6931687335968028, discriminator loss: 0.6931518800258633



![png](output_14_101.png)



![png](output_14_102.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 10500: Generator loss: 0.6931729444265372, discriminator loss: 0.6931533823013303



![png](output_14_106.png)



![png](output_14_107.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 11000: Generator loss: 0.6932281781435014, discriminator loss: 0.6931799182891851



![png](output_14_111.png)



![png](output_14_112.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 11500: Generator loss: 0.6931518677473073, discriminator loss: 0.6931494034528728



![png](output_14_116.png)



![png](output_14_117.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 12000: Generator loss: 0.6931561827659601, discriminator loss: 0.6931490161418912



![png](output_14_121.png)



![png](output_14_122.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 12500: Generator loss: 0.6932575666904447, discriminator loss: 0.6931624687910076



![png](output_14_126.png)



![png](output_14_127.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 13000: Generator loss: 0.6933974095582955, discriminator loss: 0.6932005815505983



![png](output_14_131.png)



![png](output_14_132.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 13500: Generator loss: 0.6931839333772658, discriminator loss: 0.693158835172653



![png](output_14_136.png)



![png](output_14_137.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 14000: Generator loss: 0.6935374225378037, discriminator loss: 0.6931180803775789



![png](output_14_141.png)



![png](output_14_142.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 14500: Generator loss: 0.73205586194992, discriminator loss: 0.6631317365169522



![png](output_14_146.png)



![png](output_14_147.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 15000: Generator loss: 0.7311158010959626, discriminator loss: 0.6740955169200892



![png](output_14_151.png)



![png](output_14_152.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 15500: Generator loss: 0.713291535735131, discriminator loss: 0.6825448076725009



![png](output_14_158.png)



![png](output_14_159.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 16000: Generator loss: 0.7121077469587327, discriminator loss: 0.683784301161766



![png](output_14_163.png)



![png](output_14_164.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 16500: Generator loss: 0.7064441860914231, discriminator loss: 0.6834688985347751



![png](output_14_168.png)



![png](output_14_169.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 17000: Generator loss: 0.6983533978462224, discriminator loss: 0.6896136378049857



![png](output_14_173.png)



![png](output_14_174.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 17500: Generator loss: 0.7003933570384979, discriminator loss: 0.6905953921079632



![png](output_14_178.png)



![png](output_14_179.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 18000: Generator loss: 0.6978763546943665, discriminator loss: 0.6917775844335559



![png](output_14_183.png)



![png](output_14_184.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 18500: Generator loss: 0.6968508794307716, discriminator loss: 0.6926840102672578



![png](output_14_188.png)



![png](output_14_189.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 19000: Generator loss: 0.6979714887142181, discriminator loss: 0.6927323584556574



![png](output_14_193.png)



![png](output_14_194.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 19500: Generator loss: 0.6979179352521897, discriminator loss: 0.6924955518245691



![png](output_14_198.png)



![png](output_14_199.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 20000: Generator loss: 0.6962578185796736, discriminator loss: 0.6931823527812958



![png](output_14_203.png)



![png](output_14_204.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 20500: Generator loss: 0.6958250544071197, discriminator loss: 0.6933818224668497



![png](output_14_208.png)



![png](output_14_209.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 21000: Generator loss: 0.6963146553039551, discriminator loss: 0.6934867432117457



![png](output_14_213.png)



![png](output_14_214.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 21500: Generator loss: 0.6960208708047859, discriminator loss: 0.6936893427371981



![png](output_14_218.png)



![png](output_14_219.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 22000: Generator loss: 0.6948469418287273, discriminator loss: 0.693603474259377



![png](output_14_223.png)



![png](output_14_224.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 22500: Generator loss: 0.6947548015117638, discriminator loss: 0.6938420623540875



![png](output_14_228.png)



![png](output_14_229.png)


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Step 23000: Generator loss: 0.6947393279075621, discriminator loss: 0.6941232293844223



![png](output_14_235.png)



![png](output_14_236.png)


    

