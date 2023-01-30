import setuptools


def main() -> None:
    setuptools.setup(
        name="image_dataset",
        version="0.1.0",
        author="zgy",
        author_email="qbjooo@qq.com",
        description="Image dataset",
        long_description="# image dataset\n",
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(exclude=("tools", )),
        python_requires=">=3.6",
        setup_requires=["wheel"],
        install_requires=[
            "ujson",
            "zstandard",
            "numpy",
            "Pillow"
        ],
    )


if __name__ == "__main__":
    main()
