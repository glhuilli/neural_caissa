from setuptools import find_packages, setup

setup(
      name='neural_caissa',
      version='0.0.2',
      description='Python package for Chess playing.',
      url='https://github.com/glhuilli/neural_caissa',
      author="Gaston L'Huillier",
      author_email='glhuilli@gmail.com',
      license='MIT License',
      packages=find_packages(),
      package_data={
            '': ['LICENSE']
      },
      zip_safe=False,
      install_requires=[x.strip() for x in open("requirements.txt").readlines()])
