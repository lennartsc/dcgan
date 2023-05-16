import config
import os
from DcGan import DcGan


def train_dcgan():
    """
    Trains a Deep Convolutional Generative Adversarial Network (DCGAN).

    This function initializes a DcGan object and trains it according to the provided configuration settings.

    Parameters:
        None

    Returns:
        None
    """
    # Define GPU to run training on.
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU

    # Instantiate DCGAN.
    dcgan = DcGan(
        img_shape=config.IMG_SHAPE,
        init_weights=config.INIT_WEIGHTS,
        noise_dist=config.NOISE_DIST,
        noise_length=config.NOISE_LENGTH,
        rescale_imgs=config.RESCALE_PIXEL_VALUES,
        disc_layer_stack=config.DISC_LAYER_STACK,
        disc_loss_func=config.DISC_LOSS,
        disc_optimizer=config.DISC_OPTIMIZER,
        gen_layer_stack=config.GEN_LAYER_STACK,
        gen_loss_func=config.GEN_LOSS,
        gen_optimizer=config.GEN_OPTIMIZER,
        critic_layer_stack=config.CRITIC_LAYER_STACK,
        critic_loss_func=config.CRITIC_LOSS,
        critic_optimizer=config.CRITIC_OPTIMIZER,
        results_dir=config.RES_DIR,
        data_dir=config.DATA_DIR,
    )

    # Train DcGan.
    dcgan.train(
        n_epochs_gan=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        evaluation_interval=config.EVALUATION_INTERVAL,
        n_epochs_critic=config.CRITIC_TRAIN_EPOCHS,
        disc_critic_loss_mean=config.DISC_CRITIC_LOSS_MEAN,
        rotated_train=config.ROTATED_TRAIN
    )


if __name__ == "__main__":
    train_dcgan()
