# Python Packaging Project Tutorial with PyPI 
- Authors 
  - Lakshmi Sravya Chalapati <sravyach24@gwmail.gwu.edu> 
  - Tanaka, Ei <ei.tanaka@email.gwu.edu>

## 1. Overview 
This guide provides step-by-step instructions on how to package a basic Python project. It covers how to organize and prepare your project with the required files for packaging, how to compile the package, and how to publish it to the Python Package Index (PyPI).

***Important to Note:***
The process for publishing Python packages can evolve over time with updates to packaging tools, standards, and best practices within the Python community. It's crucial to refer to the most current documentation and resources for guidance on packaging and distributing Python software. Key sources include the official Python Packaging Authority (PyPA) guidelines, the Python Packaging User Guide, and documentation for specific tools like setuptools, pip, and twine. Staying informed about the latest recommendations ensures that your package is compatible, secure, and accessible to the Python community. Always check these resources for the most up-to-date information before publishing your Python packages.

## 2. What is the PyPI?
The Python Package Index (PyPI) is a software repository for the Python programming language. PyPI helps you find and install software developed and shared by the Python community. It is a central hub where developers can upload their Python projects in packages, making them available for others to download and use. PyPI supports the pip tool, which is a package installer for Python, allowing users to install, update, and manage packages from the repository easily. Through PyPI, developers can distribute their software globally, and users can access various tools and libraries to enhance their development projects.

## 3. Packaging Flow
1. Setup the PyPI account

2. Prepare the Source Tree: Start with a source tree of your package, usually obtained from a version control system.

3. Create Configuration File: Prepare a pyproject.toml file within the source tree. This file contains the package's metadata (e.g., name, version) and details on how to build the package.

4. Generate Build Artifacts: Use a build tool to create artifacts from the configuration file. This typically includes a source distribution (sdist) and one or more wheel distributions (wheels). For pure Python packages, a generic wheel is often sufficient.

5. Upload to Package Distribution Service: Upload the created build artifacts to a package distribution service, commonly PyPI, making the package available for others to download and use.

6. Download Package: End users download the package's build artifact from the distribution service.

7. Install Package: The downloaded package is installed into the user's Python environment, typically into the site-packages directory. This step is usually handled by pip and may include additional build or compile steps as specified by the package's metadata.

## 4. Six Steps to Building and Publishing Python Project

### 4.1 Set up the account and API Token in PyPI
Before working on your project, you need to set up the account in PyPi.

*Note:*
In this tutorial, we will use `TestPyPI`. `TestPyPI` is a separate instance of the Python Package Index (PyPI) that allows you to try out the distribution tools and process without worrying about affecting the real index.

Go to https://test.pypi.org/
<img src="https://drive.google.com/thumbnail?id=1hF2-EYL2VbrecWDaNvkGdNdAHQSMiWd0&sz=w1000">

Create an account on PyPI
<img src="https://drive.google.com/thumbnail?id=1u5MPxxFM80_TxgvjT2QTr_DMJKrq1g1S&sz=w1000">

Go to Account Settings.
<img src="https://drive.google.com/thumbnail?id=1Hrz9P0IH5bMwNXVfntSL2diZn2Mq7EIn&sz=w1000">

Generate revocery codes in the Two factor authentication section.
<img src="https://drive.google.com/thumbnail?id=1m3QaJi3zxMDND_a7XmM4vvmTREpcSL25&sz=w1000">

Save your Account recovery codes in your local machine.
<img src="https://drive.google.com/thumbnail?id=1r-mHA3Zebwvu_RhXGdy7mmgoGx2kwYW3&sz=w1000">

Add 2FA with authentication application at 2FA section in the account setting.
<img src="https://drive.google.com/thumbnail?id=1tGxQhMpOesm3a_9iztmr9Wdi52y2oz2U&sz=w1000">

Read QR code with your authenticaion app, then enter an authentication code to verify application.
<img src="https://drive.google.com/thumbnail?id=1q8CzfLyBF7ak2U6KUhKkXlKD1QlCSZTL&sz=w1000">

Next, add API token in your account setting.
<img src="https://drive.google.com/thumbnail?id=1uAO-UthbaxoQapcdAMx-lVa5jku6CE7W&sz=w1000">

Create an API token.
<img src="https://drive.google.com/thumbnail?id=1Tf6cRVMN9sOcL9JnZwnOmsXMfyeZNEFu&sz=w1000">

Store the API Token in your local machine safely and securely.
<img src="https://drive.google.com/thumbnail?id=1Oi7FgCDpO6z2LhNfPcHPtSvmbBZnPpNz&sz=w1000">

**We are going to use the above API token in the later section.**

### 4.2 Prepare the Source Tree
This guide employs a straightforward project referred to as example_package_YOUR_USERNAME_HERE. If your username happens to be 'me', the package would be named example_package_me, guaranteeing that your package name is distinctive and avoids clashes with packages uploaded by others using this guide. It's advisable to adhere to this guide precisely with this project before attempting to package your own project.

Create the following package structure locally:
```
packaging_tutorial/
└── src/
    └── example_package_YOUR_USERNAME_HERE/
        ├── __init__.py
        └── example.py
```

The directory containing the Python files should match the project name. This simplifies the configuration and is more obvious to users who install the package.

*Note:* `__init__.py` is recommended to import the directory as a regular package, even if as is our case for this tutorial that file is empty.

Open `example.py` and enter the following content:
```bash
def add_two(number):
    return number + 2
```

You will now add files that are used to prepare the project for distribution. When you’re done, the project structure will look like this:
```
packaging_tutorial/
├── LICENSE
├── pyproject.toml
├── README.md
├── src/
│   └── example_package_YOUR_USERNAME_HERE/
│       ├── __init__.py
│       └── example.py
└── tests/
```

An Example of Scree Image
<img src="https://drive.google.com/thumbnail?id=1TPFywe7fsWrvOM3UeD9nOxjgZEoHXGqW&sz=w600">

### 4.3 Create Configuration File

The setup file required varies based on the tool chosen to generate the build products. It's commonly accepted to employ a `pyproject.toml` file, which is in the TOML format.

The pyproject.toml file must contain a `[build-system]` section at the very least, which identifies the build tool you're using. Several build tools exist, such as [flit](https://packaging.python.org/en/latest/key_projects/#flit), [hatch](https://packaging.python.org/en/latest/key_projects/#hatch), [pdm](https://packaging.python.org/en/latest/key_projects/#pdm), [poetry](https://packaging.python.org/en/latest/key_projects/#poetry), [Setuptools](https://packaging.python.org/en/latest/key_projects/#setuptools), [trampolim](https://pypi.org/project/trampolim/), and [whey](https://pypi.org/project/whey/), among others. The specific content to include in the `[build-system]` section can be found in the documentation for each tool.

For this tutorial, here is a table for using [hatch](https://packaging.python.org/en/latest/key_projects/#hatch):

```
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```
<img src="https://drive.google.com/thumbnail?id=1bBYTR-kuu28y796vWe3U-SFMxLHQ0iqw&sz=w600">

To cofigure metadata, open `pyproject.toml` and enter the following content:

Change the 'name' to your package name

```
[project]
name = "example_package_YOUR_USERNAME_HERE"
version = "0.0.1"
authors = [
  { name="Example Author", email="author@example.com" },
]
description = "A small example package"
readme = "README.md"
requires-python = ">=3.8"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]

[project.urls]
Homepage = "https://github.com/pypa/sampleproject"
Issues = "https://github.com/pypa/sampleproject/issues"
```

<img src="https://drive.google.com/thumbnail?id=1cY50wzcSLFI9oZ-kySHGKmBNlh-i4Pwk&sz=w1000">

See the [pyproject.toml guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#writing-pyproject-toml) for details on these and other fields that can be defined in the `[project]` table.

Open `README.md` and enter the following content. You can customize this if you'd like.


```
# Example Package

This is a simple example package that performs addition. You can use
[GitHub-flavored Markdown](https://guides.github.com/features/mastering-markdown/)
to write your content.
```

It's also important for every package uploaded to the Python Package Index to include a **license**.

For help picking a license, see https://choosealicense.com/

Once you have chosen a license, oprn `LICENSE` and enter the license text. For example, if you had chosen the MIT license:

```
Copyright (c) 2018 The Python Packaging Authority

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### 4.4 Generate Build Artifacts

The subsequent action involves creating [distribution packages](https://packaging.python.org/en/latest/glossary/#term-Distribution-Package) for the package. These packages are compiled into archives that get uploaded to the Python Package Index, allowing them to be installed via pip.

Make sure you have the latest version of PyPA’s [build](https://packaging.python.org/en/latest/key_projects/#build) installed:

**Unix/macOS**     
```bash
python3 -m pip install --upgrade build
```

**Windows**
```bash
py -m pip install --upgrade build
```

Now run this command from the same directory where pyproject.toml is located:

**Unix/macOS**
```bash
python3 -m build
```

**Windows**
```bash
python -m build
```

This command should output a lot of text and once completed should generate two files in the dist directory:

```
dist/
├── example_package_YOUR_USERNAME_HERE-0.0.1-py3-none-any.whl
└── example_package_YOUR_USERNAME_HERE-0.0.1.tar.gz
```

### 4.5 Upload the Build Artifacts

Finally, it’s time to upload your package to the Python Package Index!

You can use twine to upload the distribution packages. You’ll need to install [Twine](https://packaging.python.org/en/latest/key_projects/#twine):


**Unix/macOS**
```bash
python3 -m pip install --upgrade twine
```

**Windows**
```bash
py -m pip install --upgrade twine
```

Once installed, run Twine to upload all of the archives under dist:

**Unix/macOS**
```bash
python3 -m twine upload --repository testpypi dist/*
```

**Windows**
```bash
py -m twine upload --repository testpypi dist/*
```

You will be prompted for a username and password.

For the username, use `__token__`. For the password, use the token value, including the pypi- prefix that we generated earlier.

After the command completes, you should see output similar to this:

```
Uploading distributions to https://test.pypi.org/legacy/
Enter your username: __token__
Uploading example_package_YOUR_USERNAME_HERE-0.0.1-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.2/8.2 kB • 00:01 • ?
Uploading example_package_YOUR_USERNAME_HERE-0.0.1.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.8/6.8 kB • 00:00 • ?
```

### 4.6 Install and test the package

You can use pip to install your package and verify that it works

```bash
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps example-package-YOUR-USERNAME-HERE
```

Make sure you’re still in your virtual environment, then run Python:


**Unix/macOS**
```
python3
```
and import the package:

```
>>> from example_package_YOUR_USERNAME_HERE import example
>>> example.add_two(2)
4
```

## **Next steps**

Keep in mind that this tutorial showed you how to upload your package to Test PyPI, which isn’t a permanent storage. The Test system occasionally deletes packages and accounts. It is best to use TestPyPI for testing and experiments like this tutorial.

When you are ready to upload a real package to the Python Package Index you can do much the same as you did in this tutorial, but with these important differences:

- Choose a memorable and unique name for your package. You don’t have to append your username as you did in the tutorial, but you can’t use an existing name.

- Register an account on https://pypi.org - note that these are two separate servers and the login details from the test server are not shared with the main server.

- Use `twine upload dist/*` to upload your package and enter your credentials for the account you registered on the real PyPI. Now that you’re uploading the package in production, you don’t need to specify `--repository`; the package will upload to https://pypi.org/ by default.

- Install your package from the real PyPI using `python3 -m pip install [your-package]`.

## References

1. [PyPI](https://pypi.org/): Official Website

2. [Python Packaging Glossary](https://packaging.python.org/en/latest/glossary/)

3. [Python Packaging User Guide](https://packaging.python.org/en/latest/): The official guide by Python Packaging Authority (PyPA)

4. [Packaging Work Flow](https://packaging.python.org/en/latest/flow/)

5. [Python Packaging User Guide (Tutorial)](https://packaging.python.org/en/latest/tutorials/packaging-projects/)

6. [Writing your pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)

7. [How to create a Python Package with `__init.py__`](https://timothybramlett.com/How_to_create_a_Python_Package_with___init__py.html)

## Other Resources & Videos
1. [How to Package Your Python Code](https://python-packaging.readthedocs.io/en/latest/index.html) | by Scott Torborg (article):

  This tutorial aims to put forth an opinionated and specific pattern to make trouble-free packages for community use. It doesn’t describe the only way of doing things, merely one specific approach that works well.

2. [How to package a Python](https://py-pkgs.org/03-how-to-package-a-python.html) | by Tomas Beuzen (article):

  In this chapter we will develop an entire example Python package from beginning-to-end to demonstrate the key steps involved in developing a package.

3. [How to Publish a Python Package to PyPI (pip)](https://youtu.be/Kz6IlDCyOUY?si=faRZMCoqD0fB95mR) | by pixegami:

  This tutorial covers setting up the project, configuring the setup.py file, building and testing the package, adding CLI functionality, publishing to PyPI, and installing using pip. Watch this video if you want to share your code and make it easily installable for others.