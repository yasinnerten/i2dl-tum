import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

class Encoder(nn.Module):

    def __init__(self, hparams, input_size=28 * 28, latent_dim=20):
        super().__init__()

        # set hyperparams
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.hparams = hparams
        self.encoder = None

        ########################################################################
        # TODO: Initialize your encoder!                                       #
        #                                                                      #
        # Possible layers: nn.Linear(), nn.BatchNorm1d(), nn.ReLU(),           #
        # nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU().                             #
        # Look online for the APIs.                                            #
        #                                                                      #
        # Hint 1:                                                              #
        # Wrap them up in nn.Sequential().                                     #
        # Example: nn.Sequential(nn.Linear(10, 20), nn.ReLU())                 #
        #                                                                      #
        # Hint 2:                                                              #
        # The latent_dim should be the output size of your encoder.            #
        # We will have a closer look at this parameter later in the exercise.  #
        ########################################################################

        # Enhanced encoder architecture with BatchNorm and Dropout for better performance
        n_hidden1 = self.hparams.get('n_hidden_encoder', 256)
        n_hidden2 = self.hparams.get('n_hidden_encoder2', 128)
        n_hidden3 = self.hparams.get('n_hidden_encoder3', 64)
        dropout_rate = self.hparams.get('dropout_rate', 0.3)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, n_hidden1),
            nn.BatchNorm1d(n_hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(n_hidden1, n_hidden2),
            nn.BatchNorm1d(n_hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(n_hidden2, n_hidden3),
            nn.BatchNorm1d(n_hidden3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(n_hidden3, latent_dim)
        )

        # Store latent_dim for use in decoder and classifier
        self.latent_dim = latent_dim

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # feed x into encoder!
        return self.encoder(x)

class Decoder(nn.Module):

    def __init__(self, hparams, latent_dim=20, output_size=28 * 28):
        super().__init__()

        # set hyperparams
        self.hparams = hparams
        self.decoder = None

        ########################################################################
        # TODO: Initialize your decoder!                                       #
        ########################################################################

        # Enhanced decoder architecture: mirror the encoder with BatchNorm and Dropout
        n_hidden3 = self.hparams.get('n_hidden_encoder3', 64)
        n_hidden2 = self.hparams.get('n_hidden_encoder2', 128)
        n_hidden1 = self.hparams.get('n_hidden_encoder', 256)
        dropout_rate = self.hparams.get('dropout_rate', 0.3)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, n_hidden3),
            nn.BatchNorm1d(n_hidden3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(n_hidden3, n_hidden2),
            nn.BatchNorm1d(n_hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(n_hidden2, n_hidden1),
            nn.BatchNorm1d(n_hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(n_hidden1, output_size),
            nn.Sigmoid()  # Sigmoid to constrain output to [0,1] range for image pixels
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # feed x into decoder!
        return self.decoder(x)


class Autoencoder(nn.Module):

    def __init__(self, hparams, encoder, decoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        # Define models
        self.encoder = encoder
        self.decoder = decoder
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.set_optimizer()

    def forward(self, x):
        reconstruction = None
        ########################################################################
        # TODO: Feed the input image to your encoder to generate the latent    #
        #  vector. Then decode the latent vector and get your reconstruction   #
        #  of the input.                                                       #
        ########################################################################

        # Encode input to latent space
        latent = self.encoder(x)
        # Decode latent representation back to input space
        reconstruction = self.decoder(latent)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return reconstruction

    def set_optimizer(self):

        self.optimizer = None
        ########################################################################
        # TODO: Define your optimizer.                                         #
        ########################################################################

        # Enhanced optimizer with weight decay for regularization
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.hparams.get('learning_rate', 1e-3),
            weight_decay=self.hparams.get('weight_decay', 1e-5),
            betas=(0.9, 0.999)
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def training_step(self, batch, loss_func):
        """
        This function is called for every batch of data during training.
        It should return the loss for the batch.
        """
        loss = None
        ########################################################################
        # TODO:                                                                #
        # Complete the training step, similarly to the way it is shown in      #
        # train_classifier() in the notebook, following the deep learning      #
        # pipeline.                                                            #
        #                                                                      #
        # Hint 1:                                                              #
        # Don't forget to reset the gradients before each training step!       #
        #                                                                      #
        # Hint 2:                                                              #
        # Don't forget to set the model to training mode before training!      #
        #                                                                      #
        # Hint 3:                                                              #
        # Don't forget to reshape the input, so it fits fully connected layers.#
        #                                                                      #
        # Hint 4:                                                              #
        # Don't forget to move the data to the correct device!                 #
        ########################################################################

        # Set model to training mode
        self.train()

        # Reset gradients
        self.optimizer.zero_grad()

        # Get images from batch (autoencoder doesn't use labels)
        images = batch
        images = images.to(self.device)

        # Flatten images for fully connected layers
        images = images.view(images.shape[0], -1)

        # Forward pass
        reconstruction = self.forward(images)

        # Compute loss (MSE between input and reconstruction)
        loss = loss_func(reconstruction, images)

        # Backward pass
        loss.backward()

        # Update parameters
        self.optimizer.step()

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return loss

    def validation_step(self, batch, loss_func):
        """
        This function is called for every batch of data during validation.
        It should return the loss for the batch.
        """
        loss = None
        ########################################################################
        # TODO:                                                                #
        # Complete the validation step, similraly to the way it is shown in    #
        # train_classifier() in the notebook.                                  #
        #                                                                      #
        # Hint 1:                                                              #
        # Here we don't supply as many tips. Make sure you follow the pipeline #
        # from the notebook.                                                   #
        ########################################################################

        # Set model to evaluation mode (no gradients computed)
        self.eval()

        # Get images from batch
        images = batch
        images = images.to(self.device)

        # Flatten images for fully connected layers
        images = images.view(images.shape[0], -1)

        # Forward pass (no gradients)
        with torch.no_grad():
            reconstruction = self.forward(images)
            # Compute loss (MSE between input and reconstruction)
            loss = loss_func(reconstruction, images)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return loss

    def getReconstructions(self, loader=None):

        assert loader is not None, "Please provide a dataloader for reconstruction"
        self.eval()
        self = self.to(self.device)

        reconstructions = []

        for batch in loader:
            X = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            reconstruction = self.forward(flattened_X)
            reconstructions.append(
                reconstruction.view(-1, 28, 28).cpu().detach().numpy())

        return np.concatenate(reconstructions, axis=0)


class Classifier(nn.Module):

    def __init__(self, hparams, encoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        self.encoder = encoder
        self.model = nn.Identity()
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        
        ########################################################################
        # TODO:                                                                #
        # Given an Encoder, finalize your classifier, by adding a classifier   #
        # block of fully connected layers.                                     #
        ########################################################################

        # Enhanced classifier with deeper architecture and regularization
        n_hidden1 = self.hparams.get('n_hidden_classifier', 256)
        n_hidden2 = self.hparams.get('n_hidden_classifier2', 128)
        n_hidden3 = self.hparams.get('n_hidden_classifier3', 64)
        dropout_rate = self.hparams.get('classifier_dropout', 0.35)
        
        self.model = nn.Sequential(
            nn.Linear(self.encoder.latent_dim, n_hidden1),
            nn.BatchNorm1d(n_hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(n_hidden1, n_hidden2),
            nn.BatchNorm1d(n_hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(n_hidden2, n_hidden3),
            nn.BatchNorm1d(n_hidden3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(n_hidden3, 10)  # 10 classes for MNIST digits
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        self.set_optimizer()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.model(x)
        return x

    def set_optimizer(self):

        self.optimizer = None
        ########################################################################
        # TODO: Implement your optimizer. Send it to the classifier parameters #
        # and the relevant learning rate (from self.hparams)                   #
        ########################################################################

        # Enhanced optimizer with different learning rates for encoder and classifier
        # This allows fine-tuning the pretrained encoder more carefully
        encoder_lr = self.hparams.get('encoder_lr', self.hparams.get('learning_rate', 1e-3))
        classifier_lr = self.hparams.get('learning_rate', 1e-3)
        
        self.optimizer = torch.optim.Adam(
            [
                {'params': self.encoder.parameters(), 'lr': encoder_lr},
                {'params': self.model.parameters(), 'lr': classifier_lr}
            ],
            weight_decay=self.hparams.get('weight_decay', 1e-4),
            betas=(0.9, 0.999)
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def getAcc(self, loader=None, use_tta=False, n_tta=10):
        """
        Get accuracy with optional Test-Time Augmentation (TTA)
        
        Args:
            loader: DataLoader
            use_tta: Whether to use test-time augmentation
            n_tta: Number of augmentations to average over
        """
        assert loader is not None, "Please provide a dataloader for accuracy evaluation"

        self.eval()
        self = self.to(self.device)
            
        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            
            if use_tta:
                # Test-Time Augmentation: average predictions over multiple augmentations
                batch_scores = []
                for _ in range(n_tta):
                    X_aug = self._augment_for_tta(X)
                    flattened_X = X_aug.view(X_aug.shape[0], -1)
                    score = self.forward(flattened_X)
                    batch_scores.append(score.detach().cpu().numpy())
                # Average over augmentations
                score = np.mean(batch_scores, axis=0)
            else:
                flattened_X = X.view(X.shape[0], -1)
                score = self.forward(flattened_X)
                score = score.detach().cpu().numpy()
            
            scores.append(score)
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc
    
    def _augment_for_tta(self, X):
        """Apply random augmentation for test-time augmentation"""
        X_aug = X.clone()
        # Apply random augmentations similar to training
        for i in range(X_aug.shape[0]):
            img = X_aug[i]
            if img.dim() == 2:
                img = img.unsqueeze(0)
            
            # Random rotation
            if np.random.random() > 0.5:
                angle = np.random.uniform(-8, 8)
                img = transforms.functional.rotate(img, angle, interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
            
            # Random translation
            if np.random.random() > 0.5:
                translate = (np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5))
                img = transforms.functional.affine(img, angle=0, translate=translate, scale=1.0, shear=0, fill=0)
            
            # Random scaling
            if np.random.random() > 0.5:
                scale = np.random.uniform(0.96, 1.04)
                img = transforms.functional.affine(img, angle=0, translate=(0, 0), scale=scale, shear=0, fill=0)
            
            if img.shape[0] == 1:
                img = img.squeeze(0)
            
            X_aug[i] = img
        
        return X_aug

