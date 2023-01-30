# Image Dataset

## 使用说明

### 1. 写入图片

```python
with image_dataset.make_dataset("path/to/dataset") as ds:
    for image, meta in data_source:
        # image 是 PIL 图片
        # meta 是一个字典
        ds.write(image, meta)
```

### 2. 读取图片

```python
dataset = image_dataset.ImageDataset("path/to/dataset")
for data in dataset:
    image = data.image
    meta = data.meta
```
