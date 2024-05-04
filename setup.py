from setuptools import setup, find_packages

setup(
    name='gpt_batch',
    version='0.1.1',
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
