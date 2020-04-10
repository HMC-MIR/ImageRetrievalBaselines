import setuptools


setuptools.setup(
    name="sheet-id-cirtorch",
    version=1.0,
    install_requires=[
        "numpy",
        "scipy",
        "torch",
        "torchvision",
    ],
    packages=setuptools.find_packages(),
)
