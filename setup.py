from setuptools import setup, find_packages

setup(
    name='gpt_batch',
<<<<<<< HEAD
    version='0.1.5',
=======
    version='0.1.3',
>>>>>>> 15a870c57eb93e1942068418d5d41079d054e8b7
    packages=find_packages(),
    install_requires=[
        'openai', 'tqdm'
    ],
    author='Ted Yu',
    author_email='liddlerain@gmail.com',
    description='A package for batch processing with OpenAI API.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/fengsxy/gpt_batch',
)
