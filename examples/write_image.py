import sys
sys.path.append(".")
import image_dataset
import requests
from PIL import Image
import io
from tqdm import tqdm

def download_image(url : str) -> Image.Image:
    resp = requests.get(url)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))

def main():
    data = [
        {
            "url": "https://cdn.shopify.com/s/files/1/0286/3900/2698/products/TVN_Huile-olive-infuse-et-s-227x300_e9a90ffd-b6d2-4118-95a1-29a5c7a05a49_800x.jpg?v=1616684087",
            "text": "Olive oil infused with Tuscany herbs",
            "width": 227,
            "height": 300
        },
        {
            "url": "https://cdn.tom-tailor.com/img/272_362/1016249_22689_5.jpg",
            "text": "Sweat blazer with a colour wash - 5 - TOM TAILOR" ,
            "width": 272,
            "height": 362
        },
        {
            "url": "https://img-fs-2.wnlimg.com/p/2c4/1df/d6c/6ff58cdab3a9faefaa1db57/x200-q90.jpg",
            "text": "Mojave Oasis Apple Watch Band",
            "width": 200,
            "height": 200
        }
    ]

    with image_dataset.make_dataset("example_dataset") as ds:
        for item in tqdm(data):
            ds.write(download_image(item["url"]), item)
    print("Done")


if __name__ == "__main__":
    main()
