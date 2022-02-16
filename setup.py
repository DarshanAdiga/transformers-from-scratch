from setuptools import setup

setup(name='mini-self-attention',
      version='0.1',
      description='Self attention from scratch [Reference: http://www.peterbloem.nl/blog/transformers]',
      author='Darshan Adiga',
      author_email='3319126+DarshanAdiga@users.noreply.github.com',
      packages=['mini-self-attention'],
      install_requires=[
            'torch',
            'torchtext',
            'revtok', # For 'subword' tokenizer in the torchtext
            'tensorboard',
            'tqdm',
            'numpy'
      ],
      zip_safe=False)