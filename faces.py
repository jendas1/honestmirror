import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel

URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl

tflib.init_tf()
with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)

generator = Generator(Gs_network, batch_size=1, randomize_noise=False)


smile_direction = np.load('ffhq_dataset/latent_directions/smile.npy')
gender_direction = np.load('ffhq_dataset/latent_directions/gender.npy')
age_direction = np.load('ffhq_dataset/latent_directions/age.npy')


def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def latent_representation(image,image_size=256,learning_rate=1,iterations=1000,randomize_noise=False):
    ref_images = [image]
    print(ref_images)
    batch_size = 1
    tflib.init_tf()
    with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    generator = Generator(Gs_network, batch_size, randomize_noise=randomize_noise)
    perceptual_model = PerceptualModel(image_size, layer=9, batch_size=batch_size)
    perceptual_model.build_perceptual_model(generator.generated_image)

    # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
    for images_batch in tqdm(split_to_batches(ref_images, batch_size), total=len(ref_images) // batch_size):
        perceptual_model.set_reference_images(images_batch)
        op = perceptual_model.optimize(generator.dlatent_variable, iterations=iterations, learning_rate=learning_rate)
        pbar = tqdm(op, leave=False, total=iterations)
        for loss in pbar:
            pbar.set_description(' Loss: %.2f' % loss)
        print(' ', ' loss:', loss)
        generated_dlatents = generator.get_dlatents()
        generator.reset_dlatents()
    return generated_dlatents



def morph_latent(alpha,move_strength=1.0,steps=10,me_pt=None,direction=smile_direction+age_direction):
    z = np.empty((steps, 18, 512))
    for i, cur_alpha in enumerate(np.linspace(start=alpha, stop=alpha+move_strength, num=steps)):
        z[i] = me_pt + direction * alpha
    return z


def generate_gif(latent_vectors,filename):
    images = generate_images(latent_vectors)
    # Save into a GIF file that loops forever
    images[0].save(filename, format='GIF', append_images=images[1:], save_all=True, duration=40, loop=1)

def generate_images(latent_vectors):
    return [generate_image(vec) for vec in latent_vectors]

def generate_image(latent_vector):
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img


def align_image(filename):
    import os
    import sys
    import bz2
    from keras.utils import get_file
    from ffhq_dataset.face_alignment import image_align
    from ffhq_dataset.landmarks_detector import LandmarksDetector

    LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

    def unpack_bz2(src_path):
        data = bz2.BZ2File(src_path).read()
        dst_path = src_path[:-4]
        with open(dst_path, 'wb') as fp:
            fp.write(data)
        return dst_path
    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))

    landmarks_detector = LandmarksDetector(landmarks_model_path)
    for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(filename), start=1):
        face_img_name = '%s.png' % os.path.splitext(filename)[0]
        aligned_face_path = face_img_name
        image_align(filename, aligned_face_path, face_landmarks)
