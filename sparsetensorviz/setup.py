from setuptools import find_packages, setup

import versioneer

with open("requirements.txt") as f:
    install_requires = f.read().strip().split("\n")

extras_require = {
    "test": ["pytest"],
    "viz": ["sphinxcontrib-svgbob"],
}
extras_require["complete"] = sorted({v for req in extras_require.values() for v in req})

with open("README.md") as f:
    long_description = f.read()

setup(
    name="sparsetensorviz",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Explore multidimensional sparse data structures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Erik Welch",
    author_email="erik.n.welch@gmail.com",
    url="https://github.com/GraphBLAS/binsparse-specification",
    packages=find_packages(),
    license="BSD",
    python_requires=">=3.8",
    setup_requires=[],
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha" "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    zip_safe=False,
)
