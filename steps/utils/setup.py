from setuptools import setup, find_packages

setup(
    name="utils",
    version="0.1.0",
    packages=find_packages(),  # 自动发现所有包和子包
    install_requires=[  # 依赖项（可选）
        # "numpy>=1.18.0",
        # "requests",
    ],
    author="Tipriest",
    author_email="a1503741059@163.com",
    description="A short description of your package",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mypackage",
)
