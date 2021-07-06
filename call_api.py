import tweepy
import json
import pickle
import numpy as np
import datetime

import torch
import torchvision.utils as tvutils

import generate_names
import gan_model_definitions

with open("secrets.json") as handle:
    secrets = json.loads(handle.read())

auth = tweepy.OAuthHandler(
    secrets["TWITTER_TOKEN"],
    secrets["TWITTER_SECRET"],
)

auth.set_access_token(
    secrets["TWITTER_ACCESS_TOKEN"],
    secrets["TWITTER_ACCESS_SECRET"]
)

api = tweepy.API(auth)

with open("models/name_gen.pkl", "rb") as handle:
    name_gen = pickle.load(handle)

species_name = generate_names.generate("models/name_gen.pkl", 100)
status = f"Species: {species_name}"

device = torch.device("cuda:0")

image_content_types = ['Leaf', 'Flower', 'Entire', 'Fruit', 'Stem']

netG = gan_model_definitions.Generator(image_content_types)
netG.load_state_dict(torch.load("models/256x256.pkl"))
netG.to(device)
netG.eval()

def create_noise():
    content_type_choice = np.random.choice(range(len(image_content_types)))
    noise = torch.randn(1, gan_model_definitions.nz - len(image_content_types), 1, 1, device = device)
    random_content_types = np.eye(len(image_content_types))[content_type_choice]
    return image_content_types[content_type_choice], torch.cat((
        noise,
        torch.tensor(random_content_types).view((1, len(image_content_types), 1, 1)).to(device)
    ), dim = 1).float()

with torch.no_grad():
    img_class, noise = create_noise()
    image, _ = netG(noise)

# normalize
image = (image[0] - image[0].min()) / (image[0].max() - image[0].min()) 

image_path = f"output/{datetime.datetime.now()}.png"
tvutils.save_image(image, image_path)

# image type prediction is still not very good
# status += f"\nImage: {img_class}"

media = api.media_upload(image_path)

api.update_status(status = status, media_ids = [media.media_id])