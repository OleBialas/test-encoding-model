import owncloud

urls = [
    "https://uni-bonn.sciebo.de/s/GK6k7KgBRc4wWja",
    "https://uni-bonn.sciebo.de/s/3L5TPjbfqSStbai",
]

fnames = ["Vc_2975.mat", "2975_LickingLama_20250207_125807.mat"]

for url, fname in zip(urls, fnames):
    print("downloading")
    owncloud.Client.from_public_link(url).get_file("/", fname)
