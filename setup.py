from setuptools import setup, find_packages

setup(
    name='ponita',
    version='0.1.0',
    description='A jax implementation of the Ponita library.',
    author='David Wessels',
    py_modules=['datasets'],
    # If your package has Python dependencies, list them here. For example:
    install_requires=[
        'jax',
        'jaxlib',
    ],
)