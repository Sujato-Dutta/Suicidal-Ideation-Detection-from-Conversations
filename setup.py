from setuptools import find_packages, setup

setup(
    name="suicidal_ideation_pipeline",
    version="0.1.0",
    author="Sujato Dutta",
    author_email="sujatodutta0204@gmail.com",
    description="Suicidal Ideation Detection Pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    include_package_data=True,
    zip_safe=False,
)
