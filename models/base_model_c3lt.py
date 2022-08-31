import torch
import os.path
import torch.nn as nn
from tqdm import tqdm
from shutil import rmtree
from models.encoder import Encoder
from models.decoder import Decoder
import models.global_config as g_conf
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from models.classifier import Classifier
from typing import Callable
from torch.utils.data import DataLoader
from models.discriminator import Discriminator
import torch.nn.functional as functional
from sklearn.metrics import accuracy_score
from typing import Tuple, List, Any, Union
from models.encmodifier import Encmodifier
from pathlib import Path
import torch.nn.functional as F



class c3lt(nn.Module):
    """
    Implements the base model which combines encoder, decoder, classifier, and discriminator to learn disentangled
    features.
    """

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 classifier: Classifier,
                 discriminator: Discriminator,
                 enc_modifier: Encmodifier,
                 inv_enc_modifier: Encmodifier,
                 class_coef: float = 1.0,
                 prox_coef: float = 1.0,
                 cycle_coef: float = 1.0,
                 adversarial_coef: float = 1.0,
                 smoothness_coef: float = 1.0,
                 entropy_coef: float = 1.0,
                 ) -> None:

        """
        Parameters
        ----------
        encoder: An instance of the Encoder network
        decoder: An instance of the Decoder network
        classifier: An instance of the classifier network
        discriminator: An instance of the discriminator network
        enc_modifier: A function that takes (?, encoder.feature_dim) and (?, 1) tensors as input and produces a
                      (?, encoder.feature_dim) output
        classification_coef: Coefficient for the classification loss term in the total loss
        enc_recon_coef: Coefficient used for the encoding reconstruction term in the total loss
        decoder_coef: Coefficient for the decoder loss term in the total loss
        """

        super(c3lt, self).__init__()

        self._config = dict()
        self._config['encoder'] = encoder.config
        self._config['decoder'] = decoder.config

        # ------------------------------------------------ #
        """
        Again: now classifier = encoder + former classifier
        """
        self._config['classifier'] = classifier.config
        # ----------------------------------------------- #


        self._config['discriminator'] = discriminator.config

        # --------------------------------------------------- #
        self._config['enc_modifier'] = enc_modifier.config
        self._config['inv_enc_modifier'] = inv_enc_modifier.config
        self._config['prox_coef'] = prox_coef
        self._config['cycle_coef'] = cycle_coef
        self._config['adversarial_coef'] = adversarial_coef
        self._config['smoothness_coef'] = smoothness_coef
        self._config['entropy_coef'] = entropy_coef
        # --------------------------------------------------- #


        self._encoder = encoder
        self._decoder = decoder
        self._classifier = classifier
        self._discriminator = discriminator
        self._enc_modifier = enc_modifier

        # -------------------------------------- #
        self._class_coef = class_coef
        self._prox_coef = prox_coef
        self._cycle_coef = cycle_coef
        self._adversarial_coef = adversarial_coef
        self._smoothness_coef = smoothness_coef
        self._entropy_coef = entropy_coef
        # -------------------------------------- #
        
        # -------------------------------------------------- #
        """
        correspond to the g and h functions of the c3lt paper
        """
        self._enc_modifier = enc_modifier
        self._inv_enc_modifier = inv_enc_modifier
        # -------------------------------------------------- #

    def _fit_c3lt_batch(self,
                optimizer: Optimizer,
                images_pos: torch.Tensor,
                images_neg: torch.Tensor) -> Any:
        """
        Forward function to be used to apply the c3lt procedure to train g and h

        Parameters
        ----------
        optimizer: A Optimizer object which optimizes the parameters of self._enc_modifier and
                    self._inv_enc_modifier (EVERYTHING ELSE BEING FROZEN!)

        images_pos: A (?, encoder.image_channels, encoder.image_dim, encoder.image_dim) torch tensor
                    corresponding to an image of the positive class

        images_neg: A (?, encoder.image_channels, encoder.image_dim, encoder.image_dim) torch tensor
                    corresponding to an image of the negative class

        Returns
        -------
   
        """
        
        optimizer.zero_grad()

        z_pos = self._encoder(images_pos)
        z_pos_modif = self._enc_modifier(z_pos)
        images_pos_cf = self._decoder(z_pos_modif)

        z_neg = self._encoder(images_neg)
        z_neg_modif = self._inv_enc_modifier(z_neg)
        images_neg_cf = self._decoder(z_neg_modif)

        # Classifier loss
        logits_pos_cf = self._classifier(images_pos_cf)
        labels_pos_cf = torch.zeros(logits_pos_cf.size(0), 1).to(g_conf.DEVICE)

        logits_neg_cf = self._classifier(images_neg_cf)
        labels_neg_cf = torch.ones(logits_neg_cf.size(0), 1).to(g_conf.DEVICE)

        loss_class = 0.5 * ( self._loss_classifier(labels_pos_cf, logits_pos_cf) +  \
                    self._loss_classifier(labels_neg_cf, logits_neg_cf) )

        # Proximity loss
        prox_loss = 0.5 * (self._proximity_loss(images_pos, images_pos_cf) +  \
                            self._proximity_loss(images_neg, images_neg_cf) )

        # Cycle-consistency loss
        z_pos_cycle = self._inv_enc_modifier(z_pos_modif)
        z_neg_cycle = self._enc_modifier(z_neg_modif)
        images_pos_cycle = self._decoder(z_pos_cycle)
        images_neg_cycle = self._decoder(z_neg_cycle)

        cycle_loss = self._cycle_loss(z_pos, z_pos_cycle,
                                        z_neg, z_neg_cycle,
                                        images_pos, images_pos_cycle,
                                        images_neg, images_neg_cycle)
        
        # Adversarial loss
        is_fake = torch.cat([self._discriminator.forward(images_pos_cf),
                            self._discriminator.forward(images_pos_cycle),
                            self._discriminator.forward(images_neg_cf),
                            self._discriminator.forward(images_neg_cycle),                           
                            ], dim=0)

        labels_G = torch.ones(is_fake.size(0), 1).to(g_conf.DEVICE)
        adversarial_loss = self._loss_discriminator(is_fake, labels_G)

        # Compute total loss
        loss =  self._class_coef * loss_class + \
                self._prox_coef * prox_loss + \
                self._cycle_coef * cycle_loss + \
                self._adversarial_coef * adversarial_loss

        loss.backward()
        optimizer.step()

        return loss.item(), (loss_class.item(), prox_loss.item(), cycle_loss.item(), adversarial_loss.item())

    def fit_c3lt(self,
            data_loader: DataLoader,
            optimizer: Optimizer,
            num_epochs: int,
            path: Union[str, Path],
            verbose: bool = False) -> Any:
        """
        Implements the whole c3lt training procedure to learn the enc_modifier and inv_enc_modifier

        Parameters
        ----------
        data_loader: An instance of DataLoader class that provides training batches of pos and neg
        optimizer: Optimizer object to optimize the parameters of self._enc_modifier and self._inv_enc_modifier
        num_epochs: Number of training epochs
        verbose: Whether to print the progress

        Returns
        -------
        TBC
        """

        c3lt_losses, losses_class, prox_losses, cycle_losses, adversarial_losses = [], [], [], [], []

        for epoch in range(num_epochs):
            loop = tqdm(zip(*data_loader)) if verbose else zip(*data_loader)

            avg_c3lt_loss = 0.0
            avg_loss_class = 0.0
            avg_prox_loss = 0.0
            avg_cycle_loss = 0.0
            avg_adversarial_loss = 0.0

            for idx, ((images_pos, _), (images_neg, _)) in enumerate(loop):

                # Train on the batch
                images_pos = images_pos.to(g_conf.DEVICE)
                images_neg = images_neg.to(g_conf.DEVICE)

                # labels = labels.to(g_conf.DEVICE)
                c3lt_loss, (loss_class, prox_loss, cycle_loss, adversarial_loss) = self._fit_c3lt_batch(optimizer, images_pos, images_neg)

                avg_c3lt_loss = (avg_c3lt_loss * idx + c3lt_loss) / (idx + 1)
                avg_loss_class = (avg_loss_class * idx + loss_class) / (idx + 1)
                avg_prox_loss = (avg_prox_loss * idx + prox_loss) / (idx + 1)
                avg_cycle_loss = (avg_cycle_loss * idx + cycle_loss) / (idx + 1)
                avg_adversarial_loss = (avg_adversarial_loss * idx + adversarial_loss) / (idx + 1)

                # Update the information on the screen
                if verbose:
                    loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
                    loop.set_postfix(c3lt='{:.2f}'.format(avg_c3lt_loss),
                                     loss_class='{:.2f}'.format(avg_loss_class), 
                                     loss_prox='{:.2f}'.format(avg_prox_loss), 
                                    loss_cycle='{:.2f}'.format(avg_cycle_loss),
                                    loss_adv='{:.2f}'.format(avg_adversarial_loss),
                                    )
                # if idx == 0: break

            # Record average losses
            c3lt_losses.append(avg_c3lt_loss)
            losses_class.append(avg_loss_class)
            prox_losses.append(avg_prox_loss)
            cycle_losses.append(avg_cycle_loss)
            adversarial_losses.append(avg_adversarial_loss)

            # Debug c3lt by visualizing counterfactuals
            self._debug_c3lt(images_pos, images_neg, path)

            # if epoch == 0: break

        return c3lt_losses, losses_class, prox_losses, cycle_losses, adversarial_losses
    
    def _perceptual_loss(self, 
                        images: torch.Tensor, 
                        images_cycle: torch.Tensor, 
                        reduction: str = "mean") -> torch.Tensor:

        L1 = torch.nn.L1Loss(reduction=reduction)
        feats_pos = self._classifier(images, return_feat=True)
        feats_pos_cycle = self._classifier(images_cycle, return_feat=True)
        out = 0

        for f, g in zip(feats_pos, feats_pos_cycle):
            out += L1(f, g)

        return out / (1 + len(feats_pos))

    def _cycle_loss(self, 
                    z_pos: torch.Tensor, z_pos_cycle: torch.Tensor,
                    z_neg: torch.Tensor, z_neg_cycle: torch.Tensor,
                    images_pos: torch.Tensor, images_pos_cycle: torch.Tensor,
                    images_neg: torch.Tensor, images_neg_cycle: torch.Tensor,
                    reduction: str="mean"
                    ) -> torch.Tensor:
        
        L1 = torch.nn.L1Loss(reduction=reduction)
        return  self._perceptual_loss(images_pos, images_pos_cycle, reduction=reduction) + \
                self._perceptual_loss(images_neg, images_neg_cycle, reduction=reduction) + \
                L1(z_pos, z_pos_cycle) + L1(z_neg, z_neg_cycle)

    @staticmethod
    def _smoothness_loss(masks, beta=2, reduction="mean"):
        """
            smoothness loss that encourages smooth masks.
        :param masks:
        :param beta:
        :param reduction:
        :return:
        """
        # TODO RGB images
        masks = masks[:, 0, :, :]
        a = torch.mean(torch.abs((masks[:, :-1, :] - masks[:, 1:, :]).view(masks.shape[0], -1)).pow(beta), dim=1)
        b = torch.mean(torch.abs((masks[:, :, :-1] - masks[:, :, 1:]).view(masks.shape[0], -1)).pow(beta), dim=1)
        if reduction == "mean":
            return (a + b).mean() / 2
        else:
            return (a + b).sum() / 2

    @staticmethod
    def _entropy_loss(masks, reduction="mean"):
        """
            entropy loss that encourages binary masks.
        :param masks:
        :param reduction:
        :return:
        """
        # TODO RGB images
        masks = masks[:, 0, :, :]
        b, h, w = masks.shape
        if reduction == "mean":
            return torch.minimum(masks.view(b, -1), 1.0 - masks.view(b, -1)).mean()
        else:
            return torch.minimum(masks.view(b, -1), 1.0 - masks.view(b, -1)).sum()

    # @staticmethod
    def _proximity_loss(self,
                        images: torch.Tensor,
                        images_cf: torch.Tensor,
                        reduction: str="mean") -> torch.Tensor:
        """
        Computes the loss function for the proximity condition between
        input images and generated counterfactuals

        Parameters
        ----------
        images (torch.Tensor): input image
        images_cf (torch.Tensor): generated counterfactual 

        Returns
        -------
        prox_loss: Negative log likelihood loss        
        
        """
        L1 = nn.L1Loss(reduction=reduction)

        masks = self._gen_masks(images, images_cf, mode='mse')
        smooth = self._smoothness_loss(masks, reduction=reduction)
        entropy = self._entropy_loss(masks, reduction=reduction)

        return  (L1(images, images_cf) + self._smoothness_coef * smooth + self._entropy_coef * entropy ) / ( 1 + self._smoothness_coef + self._entropy_coef)

    @staticmethod
    def _gen_masks(inputs, targets, mode='abs'):
        """
        generates a difference masks give two images (inputs and targets).
        :param inputs:
        :param targets:
        :param mode:
        :return:
        """
        # TODO RGB images
        masks = targets - inputs
        masks = masks.view(inputs.size(0), -1)

        if mode == 'abs':
            masks = masks.abs()
            # normalize 0 to 1
            masks -= masks.min(1, keepdim=True)[0]
            masks /= masks.max(1, keepdim=True)[0]

        elif mode == "mse":
            masks = masks ** 2
            masks -= masks.min(1, keepdim=True)[0]
            masks /= masks.max(1, keepdim=True)[0]

        elif mode == 'normal':
            # normalize -1 to 1
            min_m = masks.min(1, keepdim=True)[0]
            max_m = masks.max(1, keepdim=True)[0]
            masks = 2 * (masks - min_m) / (max_m - min_m) - 1

        else:
            raise ValueError("mode value is not valid!")

        return masks.view(inputs.shape)



    @staticmethod
    def _loss_classifier(y_true: torch.Tensor,
                         y_pred: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss function for the classification problem.

        Parameters
        ----------
        y_true: A (?, 1) true label torch vector
        y_pred: Output of the classifier

        Returns
        -------
        classification_loss: Negative log likelihood loss
        """
        # return functional.nll_loss(y_pred, y_true)
        return functional.binary_cross_entropy_with_logits(y_pred, y_true.float())

    @staticmethod
    def _loss_discriminator(discriminator_output: torch.Tensor,
                            ground_truth: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss of the discriminator

        Parameters
        ----------
        discriminator_output: Output of the discriminator.
        ground_truth: Target labels to use  (1 for real, 0 for fake)

        Returns
        -------
        discriminator_loss: Binary cross entropy for the discriminator
        """
        return functional.binary_cross_entropy_with_logits(discriminator_output, ground_truth)

    @staticmethod
    def _enc_recon_loss(enc1: torch.Tensor,
                        enc2: torch.Tensor) -> torch.Tensor:
        """
        Computes the encoding reconstruction loss

        Parameters
        ----------
        enc1: First set of encodings
        enc2: Second set of encodings

        Returns
        -------
        enc_recon_loss: A metric of difference between the two sets of encodings
        """
        return functional.mse_loss(enc1, enc2)

    def _fit_gan_batch(self,
                images: torch.Tensor,
                optimizer_D: Optimizer,
                optimizer_G: Optimizer) -> float:
        """
        Trains the discriminator and the generator of a GAN

        Parameters
        ----------
        images: A (?, discriminator.image_channels, discriminator.image_dim, discriminator.image_dim) tensor
        optimizer: Optimizer to use for this update

        Returns
        -------
        loss: Loss incurred by the discriminator
        """
        
        self.train()
        feature_dim = self._config['encoder']['feature_dim']
        batch_size = images.size(0)

        # Train discriminator
        optimizer_D.zero_grad()
        is_real = self._discriminator.forward(images)

        z = torch.randn(batch_size, feature_dim).to(g_conf.DEVICE)
        fake_images = self._decoder(z.detach())
        is_fake = self._discriminator.forward(fake_images)
        
        out = torch.cat([is_real, is_fake], dim=0)
        labels_D = torch.cat([torch.ones(images.size(0), 1),
                                  torch.zeros(images.size(0), 1)], dim=0).to(g_conf.DEVICE)
        D_loss = self._loss_discriminator(out, labels_D)
        D_loss.backward()
        optimizer_D.step()

        # Train generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, feature_dim).to(g_conf.DEVICE)
        fake_images = self._decoder(z)
        is_fake = self._discriminator.forward(fake_images)
        labels_G = torch.ones(images.size(0), 1).to(g_conf.DEVICE)
        G_loss = self._loss_discriminator(is_fake, labels_G)
        G_loss.backward()
        optimizer_G.step()

        return D_loss.item(), G_loss.item()

    def _fit_classifier_batch(self,
                     images: torch.Tensor,
                     labels: torch.Tensor,
                     optimizer: Optimizer) -> Tuple[float, Tuple[float, float]]:
        """
        Trains the encoder on a single batch

        Parameters
        ----------
        images: A (?, encoder.image_channels, encoder.image_dim, encoder.image_dim) tensor
        labels: A (?, 1) label values tensor
        optimizer: Optimizer to use for this update

        Returns
        -------
        classifier_loss: cross entropy loss
        """

        optimizer.zero_grad()
        self.train()
        prediction = self._classifier.forward(images)
        classification_loss = self._loss_classifier(labels, prediction)
        classification_loss.backward()
        optimizer.step()
        """
        for name, param in self._classifier.named_parameters():
            print(name, param.mean())
        """
        return classification_loss.item()

    def fit_classifier(self,
            train_dataloader: DataLoader,
            test_dataloader: DataLoader,
            optimizer_classifier: Optimizer,
            num_epochs: int = 10,
            verbose: bool = False) -> Any:
        """
        Train the encoder using the provided data loader.

        Parameters
        ----------
        data_loader: An instance of DataLoader class that provides training batches
        optimizer_enc: An instance of Optimizer class for encoder and classifier
        num_epochs: Number of training epochs
        verbose: Whether to print the progress

        Returns
        -------
        TBC
        """

        class_losses = []
        for epoch in range(num_epochs):
            loop = tqdm(train_dataloader) if verbose else train_dataloader

            avg_class_loss = 0.0
            for idx, (images, labels) in enumerate(loop):

                # Train on the batch
                images = images.to(g_conf.DEVICE)
                labels = labels.to(g_conf.DEVICE)
                class_loss = self._fit_classifier_batch(images, labels, optimizer_classifier)

                avg_class_loss = (avg_class_loss * idx + class_loss) / (idx + 1)

                # Update the information on the screen
                if verbose:
                    loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
                    loop.set_postfix(Enc='{:.2f}'.format(avg_class_loss))

                # if idx == 0: break

            # Record average losses
            test_acc = self.compute_class_accuracy(test_dataloader)
            print("Test accuracy: {:.2f} %".format(test_acc))
            class_losses.append(avg_class_loss)

            # if epoch == 0: break

        return class_losses

    def save_classifier(self, path: Union[str, Path]):
        torch.save(self._classifier.state_dict(), path)

    def load_classifier(self, path: Union[str, Path]):
        self._classifier.load_state_dict(torch.load(path,map_location=g_conf.DEVICE))

    def save_gan(self, path: Union[str, Path]):
        torch.save({"discriminator": self._discriminator.state_dict(),
                    "decoder": self._decoder.state_dict()},
                    path)

    def load_gan(self, path: Union[str, Path]):
        params = torch.load(path, map_location=g_conf.DEVICE)
        self._discriminator.load_state_dict(params["discriminator"])
        self._decoder.load_state_dict(params["decoder"])

    def copyClasstoEnc(self):
        self._encoder._model.load_state_dict(self._classifier._encoder.state_dict())

    def fit_gan(self,
            data_loader: DataLoader,
            optimizer_D: Optimizer,
            optimizer_G: Optimizer,
            num_epochs: int,
            path: Union[str, Path],
            verbose: bool = False) -> Any:
        """
        Train the GAN using the provided data loader.

        Parameters
        ----------
        data_loader: An instance of DataLoader class that provides training batches
        optimizer_enc: An instance of Optimizer class for encoder and classifier
        num_epochs: Number of training epochs
        verbose: Whether to print the progress

        Returns
        -------
        TBC
        """
        D_losses = []
        G_losses = []

        for epoch in range(num_epochs):
            loop = tqdm(data_loader) if verbose else data_loader

            avg_D_loss = 0.0
            avg_G_loss = 0.0
            for idx, (images, _) in enumerate(loop):

                # Train on the batch
                images = images.to(g_conf.DEVICE)
                D_loss, G_loss = self._fit_gan_batch(images, optimizer_D, optimizer_G)

                avg_D_loss = (avg_D_loss * idx + D_loss) / (idx + 1)
                avg_G_loss = (avg_G_loss * idx + G_loss) / (idx + 1)

                # Update the information on the screen
                if verbose:
                    loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
                    loop.set_postfix(Disc='{:.2f}'.format(avg_D_loss),
                                    Gen='{:.2f}'.format(avg_G_loss)
                                    )
                # if idx == 0: break
            # Record average losses
            D_losses.append(avg_D_loss)
            G_losses.append(avg_G_loss)

            # Debug GAN by visualizing batch_size generated images
            self._debug_gan(images, path)

            # if epoch == 0: break

        return D_losses, G_losses

    @torch.no_grad()
    def compute_class_accuracy(self,
              data_loader: DataLoader,
              metric: Callable = accuracy_score,
              verbose: bool = False):
        """
        Evaluates the model on the given data against the given metric

        Parameters
        ----------
        data_loader: An instance of the DataLoader class
        metric: A function that takes y_true and y_pred as input and produces the desired score
        verbose: Whether to display progress while evaluating

        Returns
        -------
        score: Average score obtained by the model
        """

        loop = tqdm(data_loader) if verbose else data_loader
        score = 0
        num_examples = 0
        for _, (images, labels) in enumerate(loop):
            # Compute the predictions
            self.eval()
            images = images.to(g_conf.DEVICE)
            labels = labels.to(g_conf.DEVICE)
            predictions = F.sigmoid(self._classifier.forward(images))

            # Compute the scores
            labels = labels.cpu().detach().numpy().reshape((-1, 1))
            # predictions = torch.argmax(predictions, dim=1).cpu().detach().numpy().reshape((-1, 1))
            predictions = torch.where(predictions >= 0.5, torch.ones_like(predictions), torch.zeros_like(predictions)).cpu().numpy()
            score += metric(labels, predictions) * labels.shape[0]
            num_examples += labels.shape[0]

        return 100 * score / num_examples

    @torch.no_grad()
    def _debug_gan(self,
                    images: torch.Tensor,
                    path: Union[str, Path],
                    mean: float = 0.5,
                    scale: float = 0.5) -> None:
        """
        Exports concatenated actual and reconstructed images

        Parameters
        ----------
        images: A (?, encoder.image_channels, encoder.image_dim, encoder.image_dim) torch tensor
        base_dir: Location of the base folder where images are to be stored
        sub_dir: Name of subdirectory where images will be stored. If None, the base directory is cleaned and images
                 are stored in the base directory
        mean: Mean to be added to the image before saving
        scale: Scale factor to be multiplied to the image before saving

        Returns
        -------
        None
        """
        # Set the output path and create appropriate directories
        if os.path.exists(path):
            rmtree(path)
        os.mkdir(path)

        # Get fake images
        self.eval()
        feature_dim = self._config['encoder']['feature_dim']
        batch_size = images.size(0)

        z = torch.randn(batch_size, feature_dim).to(g_conf.DEVICE)
        fake_images = self._decoder(z)

        fake_images = torch.permute(fake_images, (0, 2, 3, 1))
        fake_images = torch.clip(fake_images * scale + mean, 0.0, 1.0)
        fake_images = fake_images.squeeze().cpu().detach().numpy()

        # Save the images
        for idx in range(fake_images.shape[0]):
            plt.imsave(os.path.join(path, str(idx + 1) + '.png'), fake_images[idx])
    

    @torch.no_grad()
    def _debug_c3lt(self,
                    images_pos: torch.Tensor,
                    images_neg: torch.Tensor,
                    path: Union[str, Path],
                    mean: float = 0.5,
                    scale: float = 0.5) -> None:
        """
        Exports concatenated actual and counterfactuals

        Parameters
        ----------
        images: A (?, encoder.image_channels, encoder.image_dim, encoder.image_dim) torch tensor
        base_dir: Location of the base folder where images are to be stored
        sub_dir: Name of subdirectory where images will be stored. If None, the base directory is cleaned and images
                 are stored in the base directory
        mean: Mean to be added to the image before saving
        scale: Scale factor to be multiplied to the image before saving

        Returns
        -------
        None
        """
        # Set the output path and create appropriate directories
        if os.path.exists(path):
            rmtree(path)
        os.mkdir(path)

        # Get fake images
        self.eval()

        z_pos = self._encoder(images_pos)
        z_pos_modif = self._enc_modifier(z_pos)
        images_pos_cf = self._decoder(z_pos_modif)

        z_neg = self._encoder(images_neg)
        z_neg_modif = self._inv_enc_modifier(z_neg)
        images_neg_cf = self._decoder(z_neg_modif)

        merged = torch.cat([images_pos, images_pos_cf, images_neg, images_neg_cf], dim=3)
        merged = torch.permute(merged, (0, 2, 3, 1))
        merged = torch.clip(merged * scale + mean, 0.0, 1.0)
        merged = merged.squeeze().cpu().detach().numpy()

        # Save the images
        for idx in range(merged.shape[0]):
            plt.imsave(os.path.join(path, str(idx + 1) + '.png'), merged[idx])
    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    @property
    def classifier(self):
        return self._classifier

    @property
    def discriminator(self):
        return self._discriminator

    @property
    def enc_modifier(self):
        return self._enc_modifier

    @property
    def inv_enc_modifier(self):
        return self._inv_enc_modifier

    @property
    def config(self):
        return self._config
