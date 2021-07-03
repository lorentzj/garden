import tweepy
import json
import pickle
import generate_names
import torch
import torchvision.utils as tvutils
import gan_model_definitions
import datetime

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
netG.load_state_dict(torch.load("models/128x128.pkl"))
netG.to(device)
netG.eval()

with torch.no_grad():
    image, img_class = netG(torch.randn(1, gan_model_definitions.nz, 1, 1, device = device))
img_class = image_content_types[img_class[0].argmax(0)]
# normalize
image = (image[0] - image[0].min()) / (image[0].max() - image[0].min()) 

image_path = f"output/{datetime.datetime.now()}.png"
tvutils.save_image(image, image_path)
status += f"\nImage: {img_class}"

media = api.media_upload(image_path)

api.update_status(status = status, media_ids = [media.media_id])