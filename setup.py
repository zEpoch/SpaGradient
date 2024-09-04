from setuptools import setup,find_packages


setup(
    name='SpaGradient',
    version='1.0',
    description='A package for calculating spatial gradients',
    author='Tao Zhou',
    author_email='zhotoa@foxmail.com',
    url='https://github.com/zEpoch/SpaGradient',
    packages=find_packages(),
    install_requires=[
        'pyacvd==0.2.11',
        'pymeshfix',
        'pyvista',
        'pymcubes',
        'anndata',
        'seaborn',
        'scikit-learn',
        'statsmodels',
        'scanpy',
        'trame',
        'pyvista[jupyter]'
    ],
)