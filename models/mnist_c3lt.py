import torch
import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder
from models.classifier import Classifier
from torchvision import transforms
from models.discriminator import Discriminator
from torch.optim import Adam
import models.global_config as g_conf
from models.helper_layers import ReshapeLayer
from models.base_model_c3lt import c3lt
from models.encmodifier import Encmodifier
import settings.environment as env
from data.utils_MNIST import createDataLoader
import os

class MnistEncoder(Encoder):

    def __init__(self,
                 image_dim: int = 28,
                 image_channels: int = 3,
                 feature_dim: int = 32,
                 channel_multiplier: int = 32):
        """
        Parameters
        ----------
        feature_dim: Dimension of the latent representation
        channel_multiplier: Number of channels in successive convolution layers will be multiples of this number
        """
        super(MnistEncoder, self).__init__(image_dim=image_dim, image_channels=image_channels, feature_dim=feature_dim)
        self._model = nn.Sequential()
        cm = channel_multiplier
        self._model.add_module('EncCon1', nn.Conv2d(self.image_channels, cm, kernel_size=5, stride=2, padding=2,
                                                    bias=False))
        self._model.add_module('EncAct1', nn.LeakyReLU(0.2))

        self._model.add_module('EncCon2', nn.Conv2d(cm, cm * 2, kernel_size=5, stride=2, padding=2, bias=False))
        self._model.add_module('EncAct2', nn.LeakyReLU(0.2))

        self._model.add_module('EncCon3', nn.Conv2d(2 * cm, 4 * cm, kernel_size=5, stride=2, padding=2, bias=False))
        self._model.add_module('EncAct3', nn.LeakyReLU(0.2))

        self._model.add_module('EncFla1', nn.Flatten())
        self._model.add_module('EncLin1', nn.Linear(in_features= 4 * 4 * cm * 4, out_features=self.feature_dim,
                                                    bias=False))
        self._model.add_module('EncAct4', nn.LeakyReLU(0.2))
        # self._model.add_module('EncAct3', nn.Tanh())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._model(inputs)

# CAUTION: here the "classifier" also includes an encoder block, so it's a "full" classifier
class MnistClassifier(Classifier):

    def __init__(self,
                 image_dim: int = 28,
                 image_channels: int = 3,
                 feature_dim: int = 32,
                 channel_multiplier: int = 32,
                num_classes: int = 10):
        """
        Parameters
        ----------
        feature_dim: Dimension of the latent representation
        """
        super(MnistClassifier, self).__init__(image_dim=image_dim, 
                                            image_channels=image_channels,
                                            feature_dim=feature_dim,
                                             num_classes=num_classes)

        # Build encoder
        self._encoder = nn.Sequential()
        cm = channel_multiplier
        self._encoder.add_module('EncCon1', nn.Conv2d(self.image_channels, cm, kernel_size=5, stride=2, padding=2,
                                                    bias=False))
        self._encoder.add_module('EncAct1', nn.LeakyReLU(0.2))

        self._encoder.add_module('EncCon2', nn.Conv2d(cm, cm * 2, kernel_size=5, stride=2, padding=2, bias=False))
        self._encoder.add_module('EncAct2', nn.LeakyReLU(0.2))

        self._encoder.add_module('EncCon3', nn.Conv2d(cm * 2, cm * 4, kernel_size=5, stride=2, padding=2, bias=False))
        self._encoder.add_module('EncAct3', nn.LeakyReLU(0.2))

        self._encoder.add_module('EncFla1', nn.Flatten())
        self._encoder.add_module('EncLin1', nn.Linear(in_features= 4 * 4 * cm * 4, out_features=self.feature_dim,
                                                    bias=False))
        self._encoder.add_module('EncAct4', nn.LeakyReLU(0.2))
        # self._encoder.add_module('EncAct3', nn.Tanh())

        # Build classifier
        self._classifier = nn.Sequential()
        self._classifier.add_module('ClaLin1', nn.Linear(self.feature_dim, self.num_classes))

    def forward(self, 
                inputs: torch.Tensor,
                return_feat: bool=False
                ) -> torch.Tensor:

        if not return_feat:      
            return self._classifier(self._encoder(inputs))
        else:
            feats = []
            s = inputs
            for layer in self._encoder:
                s = layer(s)
                feats += [s.clone()]
            for layer in self._classifier:
                s = layer(s)
                feats += [s.clone()]

            return feats

class MnistDecoder(Decoder):

    def __init__(self,
                 image_dim: int = 28,
                 image_channels: int = 3,
                 feature_dim: int = 32,
                 channel_multiplier: int = 32):
        """
        Parameters
        ----------
        feature_dim: Dimension of the latent representation
        channel_multiplier: Number of channels in transpose convolution layers will be multiples of this number
        """
        super(MnistDecoder, self).__init__(image_dim=image_dim, image_channels=image_channels, feature_dim=feature_dim)
        cm = channel_multiplier
        self._model = nn.Sequential()

        self._model.add_module('DecLin1', nn.Linear(self.feature_dim, 4 * 4 * cm * 4, bias=False))
        self._model.add_module('DecAct1', nn.LeakyReLU(0.2))
        self._model.add_module('DecRes1', ReshapeLayer(out_shape=(cm * 4, 4, 4)))

        self._model.add_module('DecTco1', nn.ConvTranspose2d(cm * 4, cm * 2, 5, stride=2, padding=2, bias=False))
        self._model.add_module('DecAct2', nn.LeakyReLU(0.2))

        self._model.add_module('DecTco', nn.ConvTranspose2d(cm * 2, cm, 5, stride=2, padding=2, 
                                                            output_padding=1,
                                                             bias=False))
        self._model.add_module('DecAct3', nn.LeakyReLU(0.2))

        self._model.add_module('DecTco3', nn.ConvTranspose2d(cm, self.image_channels, 5, stride=2, padding=2,
                                                             output_padding=1, 
                                                             bias=False))
        self._model.add_module('DecAct4', nn.Tanh())

    def forward(self,
                encodings: torch.Tensor) -> torch.Tensor:
        return self._model(encodings)


class MnistDiscriminator(Discriminator):

    def __init__(self,
                image_dim: int=28,
                image_channels: int=1,
                 channel_multiplier: int = 32):
        super(MnistDiscriminator, self).__init__(image_dim=image_dim, image_channels=image_channels)

        cm = channel_multiplier
        self._model = nn.Sequential()
        self._model.add_module('DisCon1', nn.Conv2d(self.image_channels, cm, kernel_size=5, stride=2, padding=2,
                                                    bias=False))
        self._model.add_module('DisAct1', nn.LeakyReLU(0.2))

        self._model.add_module('DisCon2', nn.Conv2d(cm, cm * 2, kernel_size=5, stride=2, padding=2, bias=False))
        self._model.add_module('DisAct2', nn.LeakyReLU(0.2))

        self._model.add_module('DisCon3', nn.Conv2d(cm * 2, cm * 4, kernel_size=5, stride=2, padding=2, bias=False))
        self._model.add_module('DisAct3', nn.LeakyReLU(0.2))

        self._model.add_module('DisFla1', nn.Flatten())
        self._model.add_module('DisLin1', nn.Linear(in_features=4 * 4 * cm * 4, out_features=cm, bias=False))
        self._model.add_module('EncAct4', nn.LeakyReLU(0.2))

        self._model.add_module('DisLin2', nn.Linear(cm, 1))

    def forward(self,
                images: torch.Tensor) -> torch.Tensor:
        return self._model(images)

class MnistEncmodifier(Encmodifier):
    def __init__(self, input_dim: int = 32, n_steps: int = 1):
        super(MnistEncmodifier, self).__init__(input_dim, n_steps)
        self._model = nn.Sequential()
        self._model.add_module('Lin1', nn.Linear(input_dim, input_dim))
        self._model.add_module('Act1', nn.ReLU())
        self._model.add_module('Lin2', nn.Linear(input_dim, input_dim))
        self._model.add_module('Act2', nn.ReLU())

    def forward(self, z):
        z_shape = z.shape
        z = z.view(z.shape[0], -1)
        z_prev = z
        z_norm = torch.norm(z, dim=1).view(-1, 1)

        for _ in range(1, self.n_steps + 1):
            z_step = z_prev + self._model(z_prev.view(z_shape)).view(z_shape[0], -1)
            z_step_norm = torch.norm(z_step, dim=1).view(-1, 1)
            z_step = z_step * z_norm / z_step_norm
            z_prev = z_step

        return z_step.view(z_shape)

class MnistBase(c3lt):
    pass


if __name__ == '__main__':

    """
    TODAY:
    - Tune classifier and GAN training on CCC, save the models.
    - Tune c3lt. 
    """
    
    # Controls
    lr_class = 3e-4

    # ---------- #
    lr_dec = 3e-4
    lr_dis = 1e-4
    # ---------- #

    batch_size = 256
    n_epochs = 100
    feat_dim = 512
    channel_mul = 64
    image_dim = 28
    image_channels = 3
    beta0 = 0.5
    verbose = True
    num_classes = 1

    # --------------------- #
    class_coef = 1.0
    prox_coef = 1.0
    cycle_coef = 1.0
    adversarial_coef = 1.0
    smoothness_coef = 10.0
    entropy_coef = 1.0
    n_steps = 1
    rewrite_classifier = False
    rewrite_gan = False
    # --------------------- #

    # Prepare the model
    model = MnistBase(
                MnistEncoder(image_dim, image_channels, feat_dim, channel_mul),
                MnistDecoder(image_dim, image_channels, feat_dim, channel_mul),
                MnistClassifier(image_dim, image_channels, feat_dim, channel_mul, num_classes),
                MnistDiscriminator(image_dim, image_channels, channel_mul),
                MnistEncmodifier(feat_dim, n_steps),
                MnistEncmodifier(feat_dim, n_steps),
                class_coef,
                prox_coef,
                cycle_coef,
                adversarial_coef,
                smoothness_coef,
                entropy_coef
    )

    model.to(g_conf.DEVICE)

    # Generate train and val dataloaders to train classifier and GAN (decoder + discriminator)
    tfm = transforms.Normalize(mean=(0.5,), std=(0.5,))
    train_loader = createDataLoader(env.DATASETS_FOLDER, tfm, batch_size, is_train=True, resize=11,
                                    keep_channel_dim=True,
                                    rewrite=False
                                    )

    test_loader = createDataLoader(env.DATASETS_FOLDER, tfm, batch_size, is_train=False, resize=11,
                                    keep_channel_dim=True,
                                    rewrite=False
                                    )

    if not os.path.isfile(os.path.join(env.MODELS_FOLDER, "classifier.pt")) or rewrite_classifier:
        # Train classifier
        optimizer_class = Adam(list(model.classifier.parameters()),
                            lr=lr_class)

        model.fit_classifier(train_loader, test_loader, optimizer_class, verbose=True)
        model.save_classifier(os.path.join(env.MODELS_FOLDER, "classifier.pt"))
    else:
        model.load_classifier(os.path.join(env.MODELS_FOLDER, "classifier.pt"))
        test_acc = model.compute_class_accuracy(test_loader)
        print("Test accuracy of the (pre-trained) classifier: {:.2f} %".format(test_acc))

    # Copy the weights of the classifier encoder onto the encoder used for c3lt
    model.copyClasstoEnc()
    # model.classifier._encoder == model.encoder._model

    if not os.path.isfile(os.path.join(env.MODELS_FOLDER, "classifier.pt")) or rewrite_gan:
        # Train GAN
        optimizer_G = Adam(list(model.decoder.parameters()), lr=lr_dec, betas=(beta0, 0.999))
        optimizer_D = Adam(list(model.discriminator.parameters()), lr=lr_dis, betas=(beta0, 0.999))
        model.fit_gan(train_loader, optimizer_D, optimizer_G, n_epochs, env.IMAGES_FOLDER, verbose=True)
        model.save_gan(os.path.join(env.MODELS_FOLDER, "gan.pt"))
    else:
        model.load_gan(os.path.join(env.MODELS_FOLDER, "gan.pt"))
        _, (images, _) = next(enumerate(train_loader))
        model._debug_gan(images, env.IMAGES_FOLDER)
    
    # Generate train and val dataloaders to train the enc_modifier and inv_enc_modifier (c3lt)
    tfm = transforms.Normalize(mean=(0.5,), std=(0.5,))
    train_loader_pos = createDataLoader(env.DATASETS_FOLDER, tfm, batch_size, is_train=True, resize=11,
                                        keep_channel_dim=True, size_dset=60000,
                                        rewrite=False,
                                        label=1
                                        )

    train_loader_neg = createDataLoader(env.DATASETS_FOLDER, tfm, batch_size, is_train=True, resize=11,
                                        keep_channel_dim=True, size_dset=60000,
                                        rewrite=False, 
                                        label=0
                                        )

    test_loader_pos = createDataLoader(env.DATASETS_FOLDER, tfm, batch_size, is_train=False, resize=11,
                                        keep_channel_dim=True, size_dset=10000,
                                        rewrite=False,
                                        label=1
                                        )

    test_loader_neg = createDataLoader(env.DATASETS_FOLDER, tfm, batch_size, is_train=False, resize=11,
                                        keep_channel_dim=True, size_dset=10000,
                                        rewrite=False,
                                        label=0
                                        )

    # Train the enc_modifier and inv_enc_modifier (c3lt)
    optimizer_enc_modifiers = Adam(list(model.enc_modifier.parameters()) + 
                                    list(model.inv_enc_modifier.parameters()), lr=lr_dec, betas=(beta0, 0.999))

    model.fit_c3lt((train_loader_pos, train_loader_neg), 
                    optimizer_enc_modifiers, n_epochs, 
                    env.IMAGES_FOLDER, verbose=True
    )

    print("Done")
