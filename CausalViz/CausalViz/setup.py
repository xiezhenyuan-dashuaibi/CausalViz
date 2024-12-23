from setuptools import setup, find_packages

setup(
    name="causalviz",
    version="0.2.0",
    description="一个用于低维数据的因果推断可视化的Python包，该包通过一层神经网络的拟合来还原X对Y的因果关系，并且对其进行可视化。同时，该包能够实现对调节效应的自动分析与可视化。",
    author="xiezhenyuan-dashuaibi",
    author_email="546091915@qq.com",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'seaborn',
        'torch',
        'matplotlib'
    ],
    python_requires='>=3.6',
)
