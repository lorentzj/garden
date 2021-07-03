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
        self.image_conv = nn.Sequential(
            # input is Z, going into a convolution
            nn.utils.parametrizations.spectral_norm(
                nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias = False)
            ),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.utils.parametrizations.spectral_norm(
                nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias = False)
            ),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.utils.parametrizations.spectral_norm(
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias = False)
            ),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.utils.parametrizations.spectral_norm(
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias = False)
            ),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.utils.parametrizations.spectral_norm(
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias = False)
            ),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.utils.parametrizations.spectral_norm(
                nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias = False)
            ),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )
        
        self.image_classify = nn.Sequential(
            nn.Linear(nz, ntc),
            nn.ReLU(True),
            nn.Linear(ntc, len(image_content_types)),
            nn.Sigmoid()
        )

    def forward(self, input):
        generated_image = self.image_conv(input)
        image_class = self.image_classify(input.view((-1, nz)))
        return generated_image, image_class

# Size of feature maps in discriminator
ndf = 64
# Size of final linear layer to take image class into account
ncl = 50

class Discriminator(nn.Module):
    def __init__(self, image_content_types):
        super(Discriminator, self).__init__()
        self.image_conv = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(3, ndf, 4, 2, 1, bias = False)
            ),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf) x 64 x 64
            nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias = False)
            ),
            nn.BatchNorm2d(ndf * 2),
            # state size. (ndf) x 32 x 32
            nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias = False)
            ),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf*2) x 16 x 16
            nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias = False)
            ),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf*4) x 8 x 8
            nn.utils.parametrizations.spectral_norm(            
                nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias = False)
            ),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf*8) x 4 x 4
            nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(ndf * 16, ncl, 4, 1, 0, bias = False)
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