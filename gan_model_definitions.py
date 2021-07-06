import torch
import torch.nn as nn

# Size of feature maps in generator
ngf = 64
# Size of z latent vector (i.e. size of generator input)
nz = 200
# Size of inner layer for image type classification
ntc = 50

class Generator(nn.Module):
    def __init__(self, image_content_types):
        super(Generator, self).__init__()
        self.n_img_cont_types = len(image_content_types)
        self.image_conv = nn.Sequential(
            # input is Z, going into a convolution
            nn.utils.parametrizations.spectral_norm(
                nn.ConvTranspose2d(nz, ngf * 32, 4, 1, 0, bias = False)
            ),
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True),
            nn.utils.parametrizations.spectral_norm(
                nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias = False)
            ),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            nn.utils.parametrizations.spectral_norm(
                nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias = False)
            ),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.utils.parametrizations.spectral_norm(
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias = False)
            ),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.utils.parametrizations.spectral_norm(
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias = False)
            ),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.utils.parametrizations.spectral_norm(
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias = False)
            ),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.utils.parametrizations.spectral_norm(
                nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias = False)
            ),
            nn.Tanh()
            # state size (nc) x 256 x 256
        )
        
    def forward(self, input):
        generated_image = self.image_conv(input)
        return generated_image, input.view((-1, nz))[:,-self.n_img_cont_types:]

# Size of feature maps in discriminator
ndf = 64
# Size of final linear layer to take image class into account
ncl = 50

class Discriminator(nn.Module):
    def __init__(self, image_content_types):
        super(Discriminator, self).__init__()
        self.image_conv = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(3, ndf, 4, 2, 1, bias = False)
            ),
            nn.LeakyReLU(0.2, inplace = True),
            nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias = False)
            ),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace = True),
            nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias = False)
            ),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace = True),
            nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias = False)
            ),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace = True),
            nn.utils.parametrizations.spectral_norm(            
                nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias = False)
            ),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace = True),
            nn.utils.parametrizations.spectral_norm(            
                nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias = False)
            ),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace = True),
            nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(ndf * 32, ncl, 4, 1, 0, bias = False)
            ),
            nn.Sigmoid()
        )
        self.image_classify = nn.Sequential(
            nn.Linear(ncl + len(image_content_types), 1),
            nn.Sigmoid(),
        )

    def forward(self, image_input, class_input):
        conv = self.image_conv(image_input)
        return self.image_classify(torch.cat([conv.view((-1, ncl)), class_input], axis = 1))