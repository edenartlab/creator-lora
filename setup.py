import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="creator_lora",
    version="0.0.0",
    author="Mayukh Deb",
    author_email="mayukhmainak2000@gmail.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/edenartlab/creator-lora",
    packages=setuptools.find_packages(),
    install_requires=None,
    python_requires=">=3.6",
    include_package_data=True,
)