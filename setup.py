from setuptools import find_packages, setup

__version__ = "1.0.1"

setup(
    name="mini_diffusion_tf",
    version=__version__,
    description="",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/kmkolasinski/mini-diffusion",
    author="Krzysztof Kolasinski",
    author_email="kmkolasinski@gmail.com",
    packages=find_packages(exclude=["tests", "notebooks"]),
    include_package_data=False,
    zip_safe=False,
    install_requires=[
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
