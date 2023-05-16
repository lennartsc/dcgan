import numpy as np
from keras.layers import Input
from keras.models import Model
from utils import get_noise, rescale_pixel_values, load_data, compute_total_loss, write_logs_to_file


class Critic:
    """
    This class represents a Critic in the context of Generative Adversarial Networks (GANs).
    The Critic is trained to differentiate between real and generated (fake) images
    and is being used for the evaluation of a DCGAN generator.
    """
    def __init__(self, img_shape, layer_stack, initial_weights, loss_func, optimizer):
        """
        Initializes the Critic with the given parameters.

        Parameters:
        - img_shape: The shape of the input images.
        - layer_stack: A list of Keras layers defining the architecture of the Critic.
        - initial_weights: The initial weights for the Critic.
        - loss_func: The loss function used to train the Critic.
        - optimizer: The optimization algorithm used to update the Critic's weights.
        """
        self.img_shape = img_shape
        self.layer_stack = layer_stack
        self.initial_weights = initial_weights
        self.loss_func = loss_func
        self.optimizer = optimizer

        self.model = self._build_model()
        self.train_losses = {"epoch": [], "target loss": [], "fake loss": [], "total loss": []}

    def _build_model(self):
        """
        Constructs the Critic model using the given layer stack.

        This method creates an input tensor with the shape of the input image, and successively applies each layer
        in the stack to this input tensor. The output is a Model instance, which represents the Critic with
        the input image tensor as input and the final tensor as output.
        """
        unknown_image = Input(shape=self.img_shape)
        critic_output = unknown_image
        for layer in self.layer_stack:
            critic_output = layer(critic_output)
        critic = Model(inputs=unknown_image, outputs=critic_output).compile(loss=self.loss_func,
                                                                            optimizer=self.optimizer)
        return critic

    def reset(self):
        """
        Resets the Critic's weights to the initial weights and clears the training losses.
        """
        self.model.set_weights(self.initial_weights)
        self.train_losses = {"epoch": [], "target loss": [], "fake loss": [], "total loss": []}

    def train(self, epochs, batch_size, generator, noise_dist, noise_length, results_dir,
              data_dir, gan_epoch, loss_mean, rescale_imgs=True):
        """
        Trains the Critic model for a given number of epochs, using real images and fake images generated by the generator.

        The training is performed in batches, where each batch consists of real and fake images. The Critic model is
        trained to classify real images as 1 and fake images as 0. The loss for each epoch is calculated and stored.
        """

        for epoch in range(epochs):

            # Generate fake batch.
            noise = get_noise(dist=noise_dist, n_samples=batch_size, length=noise_length)
            fake_batch = generator.predict(noise)
            if rescale_imgs:
                fake_batch = rescale_pixel_values(fake_batch)

            # Load target batch from train data.
            train_batch = load_data(data_dir=data_dir, data_type="train", batch_size=batch_size, rescale_imgs=rescale_imgs)
            train_target_batch = train_batch["t1"]

            # Train critic on both batches.
            target_loss = self.model.train_on_batch(train_target_batch, np.ones((batch_size, 1)))
            fake_loss = self.model.train_on_batch(fake_batch, np.zeros((batch_size, 1)))

            # Log epoch training results.
            total_loss = compute_total_loss(
                loss_target=target_loss,
                loss_fake=fake_loss,
                mean=loss_mean)

            self.train_losses["epoch"].append(epoch)
            self.train_losses["target loss"].append(target_loss)
            self.train_losses["fake loss"].append(fake_loss)
            self.train_losses["total loss"].append(total_loss)

        # Write all epoch training results to file.
        write_logs_to_file(
            dict_losses=self.train_losses,
            res_dir=f"{results_dir}/logs/critic_training",
            filename=f"gan_epoch{gan_epoch}"
        )

    def evaluate(self, data_dir, batch_size, rescale_imgs, generator, noise_dist, noise_length):
        """
        Evaluates the Critic model's performance on validation data.

        The evaluation is performed in batches, where each batch consists of real and fake images. The Critic model is
        used to classify the images, and the loss is calculated for each batch.
        """

        # Compute loss given target validation images.
        val_data_target = load_data(
            data_dir=data_dir,
            data_type="validation",
            batch_size=batch_size,
            one_batch=False,
            rescale_imgs=rescale_imgs
        )["t1"]
        n_batches = int(val_data_target.shape[0] / batch_size)
        val_target_loss = 0
        for batch in range(n_batches):
            val_target_batch = val_data_target[batch * batch_size:batch * batch_size + batch_size, :, :, ]
            val_target_loss = val_target_loss + self.model.test_on_batch(
                val_target_batch,
                np.zeros((batch_size, 1))
            )
        val_target_loss = val_target_loss / n_batches

        # Compute loss given fake images.
        val_fake_loss = 0
        for batch in range(n_batches):
            noise = get_noise(dist=noise_dist, n_samples=batch_size, length=noise_length)
            fake_batch = generator.predict(noise)
            if rescale_imgs:
                fake_batch = rescale_pixel_values(fake_batch)
            val_fake_loss = val_fake_loss + self.model.test_on_batch(fake_batch, np.ones((batch_size, 1)))
        val_fake_loss = val_fake_loss / n_batches

        return val_target_loss, val_fake_loss