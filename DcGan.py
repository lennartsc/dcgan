from Critic import Critic
from keras.layers import Input
from keras.models import Model
from utils import (
    load_data,
    get_noise,
    rescale_pixel_values,
    compute_total_loss,
    write_logs_to_file,
)
import numpy as np
import datetime


class DcGan:
    """
    Class representing a Deep Convolutional Generative Adversarial Network (DCGAN).
    """

    def __init__(
        self,
        img_shape,
        init_weights,
        noise_dist,
        noise_length,
        rescale_imgs,
        results_dir,
        data_dir,
        disc_layer_stack,
        disc_loss_func,
        disc_optimizer,
        gen_layer_stack,
        gen_loss_func,
        gen_optimizer,
        critic_layer_stack,
        critic_loss_func,
        critic_optimizer,
    ):
        """
        Initialize the GAN with defined image shape, initial weights, noise distribution, noise length,
        decision about rescaling the images, results directory, data directory, and specifications for the discriminator,
        generator, and critic including their layers, loss functions, and optimizers.
        """

        self.img_shape = img_shape
        self.init_weights = init_weights
        self.noise_dist = noise_dist
        self.noise_length = noise_length
        self.rescale_imgs = rescale_imgs
        self.results_dir = results_dir
        self.data_dir = data_dir

        # Initialize discriminator.
        self.discriminator = self._build_discriminator(disc_layer_stack)
        self.discriminator.compile(loss=disc_loss_func, optimizer=disc_optimizer)

        # Initialize generator. Compile it as a combined model of generator and non-trainable discriminator.
        self.generator = self._build_generator(gen_layer_stack)
        z = Input(shape=(1, 1, self.noise_length))
        fake_img = self.generator(z)
        self.discriminator.trainable = False
        disc_output_fake = self.discriminator(fake_img)
        self.combined = Model(inputs=z, outputs=disc_output_fake)
        self.combined.compile(loss=gen_loss_func, optimizer=gen_optimizer)

        # Initialize critic.
        self.critic = Critic(
            img_shape=self.img_shape,
            layer_stack=critic_layer_stack,
            initial_weights=self.init_weights,
            loss_func=critic_loss_func,
            optimizer=critic_optimizer,
        )

        # Initialize dictionaries for evaluation.
        self.val_losses = {
            "epoch": [],
            "target loss": [],
            "fake loss": [],
            "total loss": [],
        }

    def _build_discriminator(self, disc_layer_stack):
        """
        Constructs the discriminator component of the GAN using a given layer stack.

        Parameters:
        - disc_layer_stack: A list of Keras layers defining the architecture of the discriminator.

        This method creates an input tensor with the shape of the input image, and successively applies each layer
        in the stack to this input tensor. The output is a Model instance, which represents the discriminator with
        the input image tensor as input and the final tensor as output.

        Returns:
        - A Keras Model instance representing the discriminator.
        """

        unknown_img = Input(shape=self.img_shape)
        disc_output = unknown_img
        for layer in disc_layer_stack:
            disc_output = layer(disc_output)
        return Model(inputs=unknown_img, outputs=disc_output)

    def _build_generator(self, gen_layer_stack):
        """
        Constructs the generator component of the GAN using a given layer stack.

        Parameters:
        - gen_layer_stack: A list of Keras layers defining the architecture of the generator.

        The method creates a noise input tensor and successively applies each layer in the stack to the noise tensor.
        The output is a Model instance, which represents the generator with the noise tensor as input and the final
        tensor as output.

        Returns:
        - A Keras Model instance representing the generator.
        """

        noise = Input(shape=(1, 1, self.noise_length))
        fake_img = noise
        for layer in gen_layer_stack:
            fake_img = layer(fake_img)
        return Model(inputs=noise, outputs=fake_img)

    def _evaluate_generator(
        self, gan_train_epoch, batch_size, critic_train_epochs, disc_critic_loss_mean
    ):
        """
        Evaluates the generator's performance. The evaluation is done by training the critic from scratch using
        the training data and then using the trained critic to evaluate the generator's performance against validation data.

        Parameters:
        - gan_train_epoch: The current training epoch of the GAN.
        - batch_size: The size of the batch for training the critic.
        - critic_train_epochs: The total number of training epochs for the critic.
        - disc_critic_loss_mean: Flag to determine whether to use a mean loss for the critic.

        The method resets the critic if it's not the first epoch, trains the critic using the generator's output,
        evaluates the generator's performance using the critic and validation data, and computes the total validation loss.
        The method also logs the validation losses for each type (target, fake, total) and returns the total validation loss.
        """

        # (Re-)Initialize critic.
        if gan_train_epoch != 0:
            self.critic.reset()

        # Train critic from scratch using training data.
        self.critic.train(
            epochs=critic_train_epochs,
            batch_size=batch_size,
            generator=self.generator,
            noise_dist=self.noise_dist,
            noise_length=self.noise_length,
            results_dir=self.results_dir,
            data_dir=self.data_dir,
            gan_epoch=gan_train_epoch,
            loss_mean=disc_critic_loss_mean,
            rescale_imgs=self.rescale_imgs,
        )

        # Evaluate fixed generator using trained critic and validation data.
        val_target_loss, val_fake_loss = self.critic.evaluate(
            data_dir=self.data_dir,
            batch_size=batch_size,
            rescale_imgs=self.rescale_imgs,
            generator=self.generator,
            noise_dist=self.noise_dist,
            noise_length=self.noise_length,
        )
        val_loss_total = compute_total_loss(
            loss_target=val_target_loss,
            loss_fake=val_fake_loss,
            mean=disc_critic_loss_mean,
        )
        self.val_losses["epoch"].append(gan_train_epoch)
        self.val_losses["target loss"].append(val_target_loss)
        self.val_losses["fake loss"].append(val_fake_loss)
        self.val_losses["total loss"].append(val_loss_total)

        return val_loss_total

    def train(
        self,
        n_epochs_gan,
        batch_size,
        evaluation_interval,
        n_epochs_critic,
        disc_critic_loss_mean,
        rotated_train,
    ):
        """
        Train the GAN model. This method orchestrates the training process of both the generator and discriminator
        components of the GAN. Training occurs for a specified number of epochs, with evaluations performed at
        specified intervals.

        Parameters:
        - n_epochs_gan: The total number of training epochs for the GAN.
        - batch_size: The size of the batch for training the GAN.
        - evaluation_interval: The interval (in number of epochs) at which the GAN's performance is evaluated.
        - n_epochs_critic: The total number of training epochs for the critic.
        - disc_critic_loss_mean: Flag to determine whether to use a mean loss for the critic.
        - rotated_train: Flag to determine whether to use rotated training data.

        The method trains the discriminator and generator in tandem. It periodically evaluates the generator's
        performance using the critic, and logs the validation losses to a file.
        """

        start_time = datetime.datetime.now()

        for epoch in range(n_epochs_gan + 1):
            if (epoch % evaluation_interval) == 0:
                # Evaluate current generator.
                val_loss_total = self._evaluate_generator(
                    gan_train_epoch=epoch,
                    batch_size=batch_size,
                    critic_train_epochs=n_epochs_critic,
                    disc_critic_loss_mean=disc_critic_loss_mean,
                )

                # Print evaluation summary.
                print(
                    "[End evaluation epoch: %d/%d] [Validation loss: %f] [Time: %s]"
                    % (
                        epoch,
                        n_epochs_gan,
                        val_loss_total,
                        datetime.datetime.now() - start_time,
                    )
                )

            # Train discriminator and generator.
            noise = get_noise(
                dist=self.noise_dist, n_samples=batch_size, length=self.noise_length
            )
            fake_batch = self.generator.predict(noise)
            if self.rescale_imgs:
                fake_batch = rescale_pixel_values(fake_batch)
            train_target_batch = load_data(
                data_dir=self.data_dir,
                data_type="train",
                batch_size=batch_size,
                one_batch=True,
                rescale_imgs=self.rescale_imgs,
                rotated_train=rotated_train,
            )["t1"]
            self.discriminator.train_on_batch(
                train_target_batch, np.ones((batch_size, 1))
            )
            self.discriminator.train_on_batch(fake_batch, np.zeros((batch_size, 1)))
            self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

        # Log validation losses.
        write_logs_to_file(
            dict_losses=self.val_losses,
            res_dir=f"{self.results_dir}/logs",
            filename="val_losses",
        )
