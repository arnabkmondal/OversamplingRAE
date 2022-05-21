import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import os
import pickle
import argparse
import math
from sklearn.metrics import confusion_matrix
from pathlib import Path

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['mnist', 'fashion', 'svhn', 'cifar10', 'celeba'])
parser.add_argument('--z_dim', type=int, default=-1)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--save_interval', type=int, default=1000)
parser.add_argument('--plot_interval', type=int, default=1000)
parser.add_argument('--training_steps', type=int, default=100001)
parser.add_argument('--bn_axis', type=int, default=-1)
parser.add_argument('--ae_loss_fn', type=str, default='mse', choices=['mse', 'mae', 'bce'])
parser.add_argument('--gan_loss_fn', type=str, default='wgan',
                    choices=['wgan', 'dcgan'])
parser.add_argument('--gp', type=int, default=0, choices=[0, 1])
parser.add_argument('--gradient_penalty_weight', type=float, default=10)
parser.add_argument('--disc_training_ratio', type=int, default=5)

args = parser.parse_args()

if args.z_dim == -1:
    if args.dataset == 'mnist' or args.dataset == 'fashion':
        args.z_dim = 64
    elif args.dataset == 'svhn':
        args.z_dim = 128
    elif args.dataset == 'cifar10':
        args.z_dim = 256
    elif args.dataset == 'celeba':
        args.z_dim = 256
    else:
        raise Exception(f'Please Provide Latent Dimension of AE for {args.dataset} Dataset.')

trained_models_dir = f'./osa_Models_stable/{args.dataset}/zdim{args.z_dim}/'
training_data_dir = f'./osa_Samples_stable/{args.dataset}/zdim{args.z_dim}/'

os.makedirs(trained_models_dir, exist_ok=True)
os.makedirs(training_data_dir, exist_ok=True)

SEED = None
np.random.seed(SEED)
DPI = None

HEADER = '\033[95m'
OK_BLUE = '\033[94m'
OK_GREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
END_C = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

ds = args.dataset

data = np.load(f'../imb_data/{ds}.npz')
trainS, trainL, testS, testL = data['trainS'], data['trainL'], data['testS'], data['testL']

def build_encoder(h, w, c, nz):
    inp = tf.keras.Input(shape=(h, w, c), name="input-image")

    x = tf.keras.layers.Conv2D(
        filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', activation=None,  use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Conv2D(
        filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', activation=None,  use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(1024, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # z = tf.keras.layers.Dense(nz, activation=None, activity_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    z = tf.keras.layers.Dense(nz, activation=None)(x)

    return tf.keras.Model(inp, z, name="encoder")


def build_mixer(nz, n_class, n_component=2):
    inp1 = tf.keras.Input(shape=(nz,), name="latent1")
    inp2 = tf.keras.Input(shape=(nz,), name="latent2")
    noise = tf.keras.Input(shape=(nz,), name="noise")
    lbl = tf.keras.Input(shape=(n_class,), name='input-label')

    x = tf.keras.layers.Concatenate()([inp1, inp2, noise, lbl])
    # x = tf.keras.layers.Concatenate()([inp1, inp2, noise])

    x = tf.keras.layers.Dense(1024, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(1024, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(1024, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(1024, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    alpha = tf.keras.layers.Dense(n_component, activation='softmax')(x)

    return tf.keras.Model([inp1, inp2, noise, lbl], alpha, name='mixer')


def build_decoder(h, w, c, nz, n_class):
    inp = tf.keras.Input(shape=(nz,), name="latent")
    lbl = tf.keras.Input(shape=(n_class,), name='input-label')
    
    x = tf.keras.layers.Concatenate()([inp, lbl])
    x = tf.keras.layers.Dense(1024, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Dense(np.prod((h // 4, w // 4, 128)))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Reshape((h // 4, w // 4, 128))(x)

    x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                        activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                        activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters=c, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='sigmoid')(x)

    return tf.keras.Model([inp, lbl], x, name='decoder')


def build_discriminator(h, w, c, n_class):
    inp = tf.keras.Input(shape=(h, w, c), name='input-image')
    lbl = tf.keras.Input(shape=(n_class,), name='input-label')

    x = tf.keras.layers.Conv2D(
        filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', activation=None,  use_bias=False)(inp)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Conv2D(
        filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', activation=None,  use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Concatenate()([x, lbl])

    x = tf.keras.layers.Dense(1024, activation=None)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    logit = tf.keras.layers.Dense(1, activation=None)(x)

    return tf.keras.Model([inp, lbl], logit, name="discriminator")


def build_linear_probe(nz, num_classes):
    linear_probe_model = tf.keras.Sequential(name='linear-classifier-seq-model')
    linear_probe_model.add(tf.keras.layers.Dense(num_classes, input_shape=(nz,), activation=None))

    return linear_probe_model


def discriminator_loss(real_output, fake_output, gan_loss_fn='dcgan'):
    if gan_loss_fn == 'dcgan':
        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
        fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    elif gan_loss_fn == 'wgan':
        return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)


def generator_loss(fake_output, gan_loss_fn='dcgan'):
    if gan_loss_fn == 'dcgan':
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)
    elif gan_loss_fn == 'wgan':
        return -tf.reduce_mean(fake_output)


class TrainingDataGenerator:
    def __init__(self, x_train, y_train, n_class, side_length, channels):
        self.x_train = x_train / np.float32(255.0)
        self.y_train = y_train
        self.n_class = n_class
        self.side_length = side_length
        self.channels = channels
        self._create_pairs()

    def _create_pairs(self):
        self.img1 = []
        self.img2 = []
        self.label = []
        print(HEADER + 'Preparing Dataset' + END_C)
        for cl in tqdm(range(self.n_class)):
            class_label = cl
            filter = np.where(self.y_train == class_label)
            x_train, y_train = self.x_train[filter[0]], self.y_train[filter[0]]
            n_digits = x_train.shape[0]
            n_rnd_idx = 40
            image_indices = np.random.default_rng().choice(n_digits, size=n_rnd_idx, replace=n_digits < n_rnd_idx)
            self.img1.append(x_train[image_indices[:n_rnd_idx // 2], :, :, :])
            self.img2.append(x_train[image_indices[n_rnd_idx // 2:], :, :, :])
            self.label.append(y_train[image_indices[:n_rnd_idx // 2]].reshape([-1, ]))

        self.img1 = np.concatenate(self.img1, axis=0)
        self.img2 = np.concatenate(self.img2, axis=0)
        self.label = np.concatenate(self.label, axis=0).reshape([-1, ])

        return

    def get_batch(self, bs, cl=None):
        if cl is None:
            n_digits = self.x_train.shape[0]
            image_indices = np.random.default_rng().choice(n_digits, size=bs, replace=False)
            return self.x_train[image_indices, :, :, :], self.y_train[image_indices].reshape([-1, ])
        elif cl == -1:
            r_idx = np.random.default_rng().choice(len(self.img1), size=bs, replace=False)
            return self.img1[r_idx], self.img2[r_idx], self.label[r_idx]
        else:
            filter = np.where(self.y_train == cl)
            x_train, y_train = self.x_train[filter[0]], self.y_train[filter[0]]
            n_digits = x_train.shape[0]
            image_indices = np.random.default_rng().choice(n_digits, size=2 * bs, replace=n_digits < 2 * bs)
            return x_train[image_indices[:bs], :, :, :], x_train[image_indices[bs:], :, :, :], \
                   y_train[image_indices[:bs]].reshape([-1,])


def image_grid(images, sv_path, img_per_row):
    n_row = math.ceil(images.shape[0] // img_per_row)
    images = (images - np.min(images)) / (np.max(images) - np.min(images))
    ch = images.shape[3]
    images = np.squeeze(images * 255.0) if ch == 1 else images * 255.0
    img_grid = []
    for i in range(n_row):
        img_grid.append(np.concatenate(images[i * img_per_row:min(images.shape[0], (i + 1)*img_per_row)], axis=1))
    img_grid = np.concatenate(img_grid, axis=0)
    img_grid = Image.fromarray(np.uint8(img_grid), mode='L') if ch == 1 else Image.fromarray(np.uint8(img_grid), mode='RGB')
    img_grid.save(f"{sv_path}")
    return


def plot_graph(x, y, x_label, y_label, samples_dir, img_name):
    plt.close('all')
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(visible=True, which='both')
    plt.savefig(samples_dir + img_name, dpi=DPI)


def compute_acsa(enc_ema, lp_ema, x, y):

    z = enc_ema.predict(x / np.float32(255.0))
    y_clf_pred = lp_ema.predict(z)

    cm = confusion_matrix(y.flatten(), np.argmax(y_clf_pred, axis=1))
    class_wise_acc = cm.diagonal()/cm.sum(axis=1)

    return class_wise_acc.mean()


def rand_brightness(x):
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) - 0.5
    x = x + magnitude
    return x


def rand_saturation(x):
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) * 2
    x_mean = tf.reduce_mean(x, axis=3, keepdims=True)
    x = (x - x_mean) * magnitude + x_mean
    return x


def rand_contrast(x):
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) + 0.5
    x_mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
    x = (x - x_mean) * magnitude + x_mean
    return x


def rand_cutout(x, ratio=0.5):
    batch_size = tf.shape(x)[0]
    image_size = tf.shape(x)[1:3]
    cutout_size = tf.cast(tf.cast(image_size, tf.float32) * ratio + 0.5, tf.int32)
    offset_x = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=image_size[0] + (1 - cutout_size[0] % 2), dtype=tf.int32)
    offset_y = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=image_size[1] + (1 - cutout_size[1] % 2), dtype=tf.int32)
    grid_batch, grid_x, grid_y = tf.meshgrid(tf.range(batch_size, dtype=tf.int32), tf.range(cutout_size[0], dtype=tf.int32), tf.range(cutout_size[1], dtype=tf.int32), indexing='ij')
    cutout_grid = tf.stack([grid_batch, grid_x + offset_x - cutout_size[0] // 2, grid_y + offset_y - cutout_size[1] // 2], axis=-1)
    mask_shape = tf.stack([batch_size, image_size[0], image_size[1]])
    cutout_grid = tf.maximum(cutout_grid, 0)
    cutout_grid = tf.minimum(cutout_grid, tf.reshape(mask_shape - 1, [1, 1, 1, 3]))
    mask = tf.maximum(1 - tf.scatter_nd(cutout_grid, tf.ones([batch_size, cutout_size[0], cutout_size[1]], dtype=tf.float32), mask_shape), 0)
    x = x * tf.expand_dims(mask, axis=3)
    return x


# "hard sigmoid", useful for binary accuracy calculation from logits
def step(values):
    # negative values -> 0.0, positive values -> 1.0
    return 0.5 * (1.0 + tf.sign(values))


# augments images with a probability that is dynamically updated during training
class AdaptiveAugmenter(tf.keras.Model):
    def __init__(self, image_size, channels, max_translation=0.125, max_rotation=0.125, 
                 max_zoom=0.25, target_acc=0.85, integration_steps=1000):
        super().__init__()

        self.target_accuracy = target_acc
        self.integration_steps = integration_steps

        # stores the current probability of an image being augmented
        self.probability = tf.Variable(0.0)

        # the corresponding augmentation names from the paper are shown above each layer
        # the authors show (see figure 4), that the blitting and geometric augmentations
        # are the most helpful in the low-data regime
        self.augmenter = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(image_size, image_size, channels)),
                # blitting/x-flip:
                tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
                # blitting/integer translation:
                tf.keras.layers.experimental.preprocessing.RandomTranslation(
                    height_factor=max_translation,
                    width_factor=max_translation,
                    interpolation="nearest",
                ),
                # geometric/rotation:
                tf.keras.layers.experimental.preprocessing.RandomRotation(factor=max_rotation),
                # geometric/isotropic and anisotropic scaling:
                tf.keras.layers.experimental.preprocessing.RandomZoom(
                    height_factor=(-max_zoom, 0.0), width_factor=(-max_zoom, 0.0)
                ),
                # random brightness, saturation, contrast
                tf.keras.layers.Lambda(rand_brightness),
                tf.keras.layers.Lambda(rand_saturation),
                tf.keras.layers.Lambda(rand_contrast),

                tf.keras.layers.Lambda(rand_cutout)
            ],
            name="adaptive_augmenter",
        )

    def call(self, images, training):
        if training:
            augmented_images = self.augmenter(images, training)
            batch_size = images.shape[0]
            # during training either the original or the augmented images are selected
            # based on self.probability
            augmentation_values = tf.random.uniform(
                shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
            )
            augmentation_bools = tf.math.less(augmentation_values, self.probability)

            images = tf.where(augmentation_bools, augmented_images, images)
        return images

    def update(self, real_logits):
        current_accuracy = tf.reduce_mean(step(real_logits))

        # the augmentation probability is updated based on the dicriminator's
        # accuracy on real images
        accuracy_error = current_accuracy - self.target_accuracy
        self.probability.assign(
            tf.clip_by_value(
                self.probability + accuracy_error / self.integration_steps, 0.0, 1.0
            )
        )

    
class OverSamplingAE:
    def __init__(self, bs, z_dim, lr, models_dir, samples_dir,
                 training_steps, save_interval, plot_interval, bn_axis, gradient_penalty_weight,
                 disc_training_ratio, ae_loss_fn, gan_loss_fn, gp):
        self.dataset = ds
        self.bs = bs
        self.z_dim = z_dim
        self.lr = lr
        self.model_dir = models_dir
        self.samples_dir = samples_dir
        self.training_steps = training_steps
        self.save_interval = save_interval
        self.plot_interval = plot_interval
        self.bn_axis = bn_axis
        self.gradient_penalty_weight = gradient_penalty_weight
        self.disc_training_ratio = disc_training_ratio
        self.ae_loss_fn = ae_loss_fn
        self.gan_loss_fn = gan_loss_fn
        self.gp = gp
        self.kernel_initializer = tf.keras.initializers.GlorotNormal()
        self.bias_initializer = None
        self.enc_kernel_reg = None
        self.dec_kernel_reg = None
        self.enc_kernel_constraint = None
        self.dec_kernel_constraint = None
        self.disc_kernel_constraint = None
        self.gen_kernel_constraint = None
        self.lat_reg = None
        self.x_train, self.y_train, self.side_length, self.channels = trainS, trainL.flatten(), trainS.shape[1], trainS.shape[3]
        self.num_class = len(np.unique(self.y_train))
        
        self.augmenter = AdaptiveAugmenter(self.side_length, self.channels)
        self.encoder = build_encoder(self.side_length, self.side_length, self.channels, self.z_dim)
        self.decoder = build_decoder(self.side_length, self.side_length, self.channels, self.z_dim, self.num_class)
        self.discriminator = build_discriminator(self.side_length, self.side_length, self.channels, self.num_class)
        self.linear_probe = build_linear_probe(self.z_dim, self.num_class)
        self.mixer_net = build_mixer(self.z_dim, self.num_class, n_component=2)

        self.encoder_ema = tf.keras.models.clone_model(self.encoder)
        self.decoder_ema = tf.keras.models.clone_model(self.decoder)
        self.linear_probe_ema = tf.keras.models.clone_model(self.linear_probe)
        self.ema = 0.99

        self.enc_trainer = tfa.optimizers.AdamW(learning_rate=self.lr, weight_decay=0.001, name='ENC-opt')
        self.mixer_trainer = tf.keras.optimizers.Adam(learning_rate=self.lr, name='Mixer-opt')

        self.dec_trainer = tf.keras.optimizers.Adam(learning_rate=self.lr, name='DEC-opt')
        self.img_gen_trainer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=0.5, beta_2=0.9,
                                                        name='image-generator-opt')
        self.img_disc_trainer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=0.5, beta_2=0.9,
                                                         name='image-discriminator-opt')

    @tf.function
    def gen_train_step(self, x1, x2, y1, y2, y_onehot):
        with tf.GradientTape() as img_gen_tape:

            x = tf.concat([x1, x2], axis=0)

            z1_real = self.encoder(x1, training=True)
            z2_real = self.encoder(x2, training=True)

            z_vec1 = tf.concat([z1_real, z2_real], axis=0)
            z_vec2 = tf.concat([z2_real, z1_real], axis=0)
            lbl = tf.concat([y_onehot, y_onehot], axis=0)
            noise = tf.random.normal((z_vec1.shape[0], self.z_dim))
            alphas = self.mixer_net([z_vec1, z_vec2, noise, lbl], training=True)
            z_mix = tf.reshape(alphas[:, 0], (-1, 1)) * z_vec1 + tf.reshape(alphas[:, 1], (-1, 1)) * z_vec2
            y_mix_pred = self.linear_probe(z_mix, training=True)
            x_gen = self.decoder([z_mix, tf.concat([y_onehot, y_onehot], axis=0)], training=True)

            aug_images = self.augmenter(x, training=True)
            z_real = self.encoder(aug_images, training=True)
            y_pred = self.linear_probe(z_real, training=True)
            x_hat = self.decoder(
                [z_real, tf.concat([y_onehot, y_onehot], axis=0)], training=True)

            fake_images = self.augmenter(tf.concat([x_gen, x_hat], axis=0), training=True)

            z_fake = self.encoder(fake_images, training=True)
            y_fake_pred = self.linear_probe(z_fake, training=False)

            fake_img_logits = self.discriminator(
                [fake_images, tf.concat([y_onehot, y_onehot, y_onehot, y_onehot], axis=0)], training=True)

            # ssim_loss = tf.reduce_mean(1 - tf.image.ssim(x, x_hat, max_val=1.0))
            mae_loss = tf.keras.losses.MeanAbsoluteError()(x, x_hat)

            clf_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
                tf.concat([y1, y2, y1, y2], axis=0), tf.concat([y_pred, y_mix_pred], axis=0))

            fake_clf_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
                tf.concat([y1, y2, y1, y2], axis=0), y_fake_pred)

            img_gen_loss = generator_loss(fake_img_logits, self.gan_loss_fn)

            random_label = tf.gather(lbl, tf.random.shuffle(tf.range(lbl.shape[0])))
            mixer_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(
                random_label, y_mix_pred
            )

        img_gen_g = img_gen_tape.gradient(img_gen_loss, self.decoder.trainable_variables)

        self.img_gen_trainer.apply_gradients(zip(img_gen_g, self.decoder.trainable_variables))

        for weight, ema_weight in zip(
            self.decoder.weights, self.decoder_ema.weights
        ):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        return mae_loss, clf_loss, mixer_loss, img_gen_loss, fake_clf_loss

    @tf.function
    def clf_dec_train_step(self, x1, x2, y1, y2, y_onehot):
        with tf.GradientTape() as clf_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as mixer_tape:

            x = tf.concat([x1, x2], axis=0)

            z1_real = self.encoder(x1, training=True)
            z2_real = self.encoder(x2, training=True)

            z_vec1 = tf.concat([z1_real, z2_real], axis=0)
            z_vec2 = tf.concat([z2_real, z1_real], axis=0)
            lbl = tf.concat([y_onehot, y_onehot], axis=0)
            noise = tf.random.normal((z_vec1.shape[0], self.z_dim))
            alphas = self.mixer_net([z_vec1, z_vec2, noise, lbl], training=True)
            z_mix = tf.reshape(alphas[:, 0], (-1, 1)) * z_vec1 + tf.reshape(alphas[:, 1], (-1, 1)) * z_vec2
            y_mix_pred = self.linear_probe(z_mix, training=True)
            x_gen = self.decoder([z_mix, tf.concat([y_onehot, y_onehot], axis=0)], training=True)

            aug_images = self.augmenter(x, training=True)
            z_real = self.encoder(aug_images, training=True)
            y_pred = self.linear_probe(z_real, training=True)
            x_hat = self.decoder(
                [z_real, tf.concat([y_onehot, y_onehot], axis=0)], training=True)

            fake_images = self.augmenter(tf.concat([x_gen, x_hat], axis=0), training=True)

            z_fake = self.encoder(fake_images, training=True)
            y_fake_pred = self.linear_probe(z_fake, training=False)

            fake_img_logits = self.discriminator(
                [fake_images, tf.concat([y_onehot, y_onehot, y_onehot, y_onehot], axis=0)], training=True)

            # ssim_loss = tf.reduce_mean(1 - tf.image.ssim(x, x_hat, max_val=1.0))
            mae_loss = tf.keras.losses.MeanAbsoluteError()(x, x_hat)

            clf_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
                tf.concat([y1, y2, y1, y2], axis=0), tf.concat([y_pred, y_mix_pred], axis=0))

            fake_clf_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
                tf.concat([y1, y2, y1, y2], axis=0), y_fake_pred)

            img_gen_loss = generator_loss(fake_img_logits, self.gan_loss_fn)

            random_label = tf.gather(lbl, tf.random.shuffle(tf.range(lbl.shape[0])))
            mixer_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(
                random_label, y_mix_pred
            )

            enc_loss = 5 * clf_loss + 1.0 * mae_loss
            dec_loss = fake_clf_loss

        enc_g = clf_tape.gradient(enc_loss, self.encoder.trainable_variables + self.linear_probe.trainable_variables)
        dec_g = dec_tape.gradient(dec_loss, self.decoder.trainable_variables)
        mixer_g = mixer_tape.gradient(mixer_loss, self.mixer_net.trainable_variables)

        self.enc_trainer.apply_gradients(zip(enc_g, self.encoder.trainable_variables + self.linear_probe.trainable_variables))
        self.dec_trainer.apply_gradients(zip(dec_g, self.decoder.trainable_variables))
        self.mixer_trainer.apply_gradients(zip(mixer_g, self.mixer_net.trainable_variables))

        for weight, ema_weight in zip(
            self.encoder.weights, self.encoder_ema.weights
        ):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        for weight, ema_weight in zip(
            self.decoder.weights, self.decoder_ema.weights
        ):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        for weight, ema_weight in zip(
            self.linear_probe.weights, self.linear_probe_ema.weights
        ):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        return mae_loss, clf_loss, mixer_loss, img_gen_loss, fake_clf_loss

    @tf.function
    def disc_train_step(self, x1, x2, y1, y2, y_onehot):
        with tf.GradientTape() as img_disc_tape:

            x = tf.concat([x1, x2], axis=0)

            z1_real = self.encoder(x1, training=True)
            z2_real = self.encoder(x2, training=True)

            z_vec1 = tf.concat([z1_real, z2_real], axis=0)
            z_vec2 = tf.concat([z2_real, z1_real], axis=0)
            lbl = tf.concat([y_onehot, y_onehot], axis=0)
            noise = tf.random.normal((z_vec1.shape[0], self.z_dim))
            alphas = self.mixer_net([z_vec1, z_vec2, noise, lbl], training=True)
            z_mix = tf.reshape(alphas[:, 0], (-1, 1)) * z_vec1 + tf.reshape(alphas[:, 1], (-1, 1)) * z_vec2
            y_mix_pred = self.linear_probe(z_mix, training=True)
            x_gen = self.decoder([z_mix, tf.concat([y_onehot, y_onehot], axis=0)], training=True)

            aug_images = self.augmenter(x, training=True)
            z_real = self.encoder(aug_images, training=True)
            x_hat = self.decoder(
                [z_real, tf.concat([y_onehot, y_onehot], axis=0)], training=True)

            real_img_logits = self.discriminator(
                [aug_images, tf.concat([y_onehot, y_onehot], axis=0)], training=True)

            fake_images = self.augmenter(tf.concat([x_gen, x_hat], axis=0), training=True)
            fake_img_logits = self.discriminator(
                [fake_images, tf.concat([y_onehot, y_onehot, y_onehot, y_onehot], axis=0)], training=True)

            img_disc_loss = discriminator_loss(real_img_logits, fake_img_logits, self.gan_loss_fn)

            # Get the interpolated image
            alpha = tf.random.normal([4 * self.bs, 1, 1, 1], 0.0, 1.0)
            diff = fake_images - tf.concat([aug_images, aug_images], axis=0)
            interpolated = tf.concat([aug_images, aug_images], axis=0) + alpha * diff

            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                # 1. Get the discriminator output for this interpolated image.
                pred = self.discriminator(
                    [interpolated, tf.concat([y_onehot, y_onehot, y_onehot, y_onehot], axis=0)], training=True)

            # 2. Calculate the gradients w.r.t to this interpolated image.
            grads = gp_tape.gradient(pred, [interpolated])[0]
            # 3. Calculate the norm of the gradients.
            norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
            gp = tf.reduce_mean((norm - 1.0) ** 2)

            img_disc_loss += 10 * gp

        img_disc_g = img_disc_tape.gradient(img_disc_loss, self.discriminator.trainable_variables)
        
        self.augmenter.update(real_img_logits)

        self.img_disc_trainer.apply_gradients(zip(img_disc_g, self.discriminator.trainable_variables))

        return img_disc_loss

    @tf.function
    def reconstruct(self, x, y_onehot):
        z = self.encoder_ema(x, training=False)
        x_r = self.decoder_ema([z, y_onehot], training=False)
        return x_r

    @tf.function
    def generate(self, x1, x2, y_onehot):
        z1 = self.encoder_ema(x1, training=False)
        z2 = self.encoder_ema(x2, training=False)
        noise = tf.random.normal((z1.shape[0], self.z_dim))
        alphas = self.mixer_net([z1, z2, noise, y_onehot], training=False)
        z_mix = tf.reshape(alphas[:, 0], (-1, 1)) * z1 + tf.reshape(alphas[:, 1], (-1, 1)) * z2
        x_gen = self.decoder_ema([z_mix, y_onehot], training=False)
        return x_gen

    def train(self):
        mae_loss_buf = []
        clf_loss_buf = []
        fake_clf_loss_buf = []
        img_gen_loss_buf = []
        img_disc_loss_buf = []
        mixer_loss_buf = []
        acsa_buf = []
        steps_buf = []

        for step in range(self.training_steps):
            if step % 25 == 0:
                if step > 0:
                    del true_data_gen
                true_data_gen = TrainingDataGenerator(self.x_train, self.y_train, self.num_class, self.side_length,
                                                      self.channels)

            x1, x2, y = true_data_gen.get_batch(bs=self.bs, cl=-1)
            y_onehot = tf.keras.utils.to_categorical(y, self.num_class)

            if step % 5 != 0 or step == 0:
                mae_l, clf_l, mix_l, img_gen_l, fake_clf_l = \
                    self.gen_train_step(x1.astype(np.float32), x2.astype(np.float32), y, y, y_onehot)
                for _ in range(5):
                    x1, x2, y = true_data_gen.get_batch(bs=self.bs, cl=-1)
                    y_onehot = tf.keras.utils.to_categorical(y, self.num_class)
                    img_disc_l = self.disc_train_step(x1.astype(np.float32), x2.astype(np.float32), y, y, y_onehot)
            else:
                mae_l, clf_l, mix_l, img_gen_l, fake_clf_l = \
                    self.clf_dec_train_step(x1.astype(np.float32), x2.astype(np.float32), y, y, y_onehot)
            
            if step % (self.plot_interval // 10) == 0:
                reconstructed_images = self.reconstruct(x1, y_onehot)
                image_grid(images=np.concatenate((x1[0:8], reconstructed_images[0:8]), 0),
                           sv_path=f'{self.samples_dir}recon_{step}.png', img_per_row=8)
                gen = self.generate(x1, x2, y_onehot)
                image_grid(images=np.concatenate((x1[0:8], gen[0:8], x2[0:8]), 0).reshape(-1, self.side_length, self.side_length, self.channels),
                           sv_path=f'{self.samples_dir}gen_{step}.png', img_per_row=8)

                mae_loss_buf.append(mae_l.numpy())
                mixer_loss_buf.append(mix_l.numpy())
                clf_loss_buf.append(clf_l.numpy())
                fake_clf_loss_buf.append(fake_clf_l.numpy())
                img_gen_loss_buf.append(img_gen_l.numpy())
                img_disc_loss_buf.append(img_disc_l.numpy())
                steps_buf.append(step)
                acsa_buf.append(compute_acsa(self.encoder_ema, self.linear_probe_ema, testS, testL))


                print(HEADER +
                      f'Training an CAE ({self.gan_loss_fn.upper()}) Model on {self.dataset.upper()}\n' + END_C)
                print(HEADER + f'Latent Dim: {self.z_dim}; AE Loss Fn: {self.ae_loss_fn}, '
                               f'GAN Loss Fn: {self.gan_loss_fn}\n' + END_C)
                print(WARNING + f'Step: {steps_buf[-1]}, ' + END_C + OK_BLUE +
                      f'MAE Loss: {mae_loss_buf[-1]:.4f}, Mixer Loss: {mixer_loss_buf[-1]:.4f}, '
                      f'Clf Loss: {clf_loss_buf[-1]:.4f}, Fake Clf Loss: {fake_clf_loss_buf[-1]:.4f}, '
                      f'Image Gen Loss: {img_gen_loss_buf[-1]:.4f}, '
                      f'Image Disc Loss: {img_disc_loss_buf[-1]:.4f}, ACSA: {acsa_buf[-1]:.4f}\n'
                      + END_C)

            if step % self.plot_interval == 0 and step > 0:
                plot_graph(x=steps_buf, y=mae_loss_buf, x_label='Steps', y_label='MAE Loss',
                           samples_dir=self.samples_dir, img_name='mae_loss.png')
                plot_graph(x=steps_buf, y=mixer_loss_buf, x_label='Steps', y_label='Mixer Loss',
                           samples_dir=self.samples_dir, img_name='mixer_loss.png')
                plot_graph(x=steps_buf, y=clf_loss_buf, x_label='Steps', y_label='Clf Loss',
                           samples_dir=self.samples_dir, img_name='clf_loss.png')
                plot_graph(x=steps_buf, y=fake_clf_loss_buf, x_label='Steps', y_label='Fake Clf Loss',
                           samples_dir=self.samples_dir, img_name='fake_clf_loss.png')
                plot_graph(x=steps_buf, y=img_gen_loss_buf, x_label='Steps', y_label='Img. Gen. Loss',
                           samples_dir=self.samples_dir, img_name='img_gen_loss.png')
                plot_graph(x=steps_buf, y=img_disc_loss_buf, x_label='Steps', y_label='Img. Disc. Loss',
                           samples_dir=self.samples_dir, img_name='img_disc_loss.png')
                plot_graph(x=steps_buf, y=acsa_buf, x_label='Steps', y_label='ACSA',
                           samples_dir=self.samples_dir, img_name='acsa.png')

            if step % self.save_interval == 0 and step > 0:
                tf.keras.models.save_model(self.encoder, f'{self.model_dir}/encoder_{step}', overwrite=True,
                                           include_optimizer=True, save_format='tf', signatures=None, options=None)
                tf.keras.models.save_model(self.decoder, f'{self.model_dir}/decoder_{step}', overwrite=True,
                                           include_optimizer=True, save_format='tf', signatures=None, options=None)
                tf.keras.models.save_model(self.linear_probe, f'{self.model_dir}/lp_{step}', overwrite=True,
                                           include_optimizer=True, save_format='tf', signatures=None, options=None)
                tf.keras.models.save_model(self.encoder_ema, f'{self.model_dir}/encoder_ema_{step}', overwrite=True,
                                           include_optimizer=True, save_format='tf', signatures=None, options=None)
                tf.keras.models.save_model(self.decoder_ema, f'{self.model_dir}/decoder_ema_{step}', overwrite=True,
                                           include_optimizer=True, save_format='tf', signatures=None, options=None)
                tf.keras.models.save_model(self.linear_probe_ema, f'{self.model_dir}/lp_ema_{step}', overwrite=True,
                                           include_optimizer=True, save_format='tf', signatures=None, options=None)
                tf.keras.models.save_model(self.discriminator, f'{self.model_dir}/disc_{step}', overwrite=True,
                                           include_optimizer=True, save_format='tf', signatures=None, options=None)
                tf.keras.models.save_model(self.mixer_net, f'{self.model_dir}/mixer_net_{step}', overwrite=True,
                                           include_optimizer=True, save_format='tf', signatures=None, options=None)

        with open(self.samples_dir + 'mae_loss.pkl', 'wb') as fp:
            pickle.dump(mae_loss_buf, fp)
        with open(self.samples_dir + 'mixer_loss.pkl', 'wb') as fp:
            pickle.dump(mixer_loss_buf, fp)
        with open(self.samples_dir + 'clf_loss.pkl', 'wb') as fp:
            pickle.dump(clf_loss_buf, fp)
        with open(self.samples_dir + 'fake_clf_loss.pkl', 'wb') as fp:
            pickle.dump(fake_clf_loss_buf, fp)
        with open(self.samples_dir + 'img_gen_loss.pkl', 'wb') as fp:
            pickle.dump(img_gen_loss_buf, fp)
        with open(self.samples_dir + 'img_disc_loss.pkl', 'wb') as fp:
            pickle.dump(img_disc_loss_buf, fp)
        with open(self.samples_dir + 'plot_steps.pkl', 'wb') as fp:
            pickle.dump(steps_buf, fp)
        with open(self.samples_dir + 'acsa.pkl', 'wb') as fp:
            pickle.dump(acsa_buf, fp)

        return

tf.compat.v1.reset_default_graph()
model = OverSamplingAE(
    bs=args.batch_size, z_dim=args.z_dim, lr=args.lr,
    models_dir=trained_models_dir, samples_dir=training_data_dir,
    training_steps=args.training_steps, save_interval=args.save_interval,
    plot_interval=args.plot_interval, bn_axis=args.bn_axis,
    gradient_penalty_weight=args.gradient_penalty_weight, disc_training_ratio=args.disc_training_ratio,
    ae_loss_fn=args.ae_loss_fn, gan_loss_fn=args.gan_loss_fn, gp=args.gp
)

model.train()

print(HEADER + 'Training Complete' + END_C)
