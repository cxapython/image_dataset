import sys
sys.path.append(".")
import image_dataset
import json

def main():
    dataset = image_dataset.ImageDataset("example_dataset")
    for i, data in enumerate(dataset):
        data.image.save(f"img_{i}.jpg")
        json.dump(data.meta, open(f"data_{i}.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
