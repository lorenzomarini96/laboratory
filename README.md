# laboratory
Repository with all (or almost) my discoveries to do typical physics laboratory data analysis.

## Table of Contents
* [read_file](https://github.com/lorenzomarini96/model/blob/main/README.md#creating-the-repository-on-github)
* [fit](https://github.com/lorenzomarini96/model/blob/main/README.md#creation-of-basic-folders)
    * linear-fit
    * square-fit
    * exponential-fit
    * ...
* [histogram](https://github.com/lorenzomarini96/model/blob/main/README.md#creating-a-module)

* [Unittest](https://github.com/lorenzomarini96/model/blob/main/README.md#unittest)
    * [Code structure for unittest](https://github.com/lorenzomarini96/model/blob/main/README.md#code-structure-for-unittest)
    * [Info on assertEquale](https://github.com/lorenzomarini96/model/blob/main/README.md#info-on-assertequale)


> Also this time always in the same folder (```tests```) also add:
> - The ```__init__.py``` file
> - The ```Makefile``` file 

> (see: [Makefile](https://github.com/lorenzomarini96/model/blob/main/tests/Makefile) )

**CAUTION: Import modules into Python**

[Where does Python look for modules?](https://bic-berkeley.github.io/psych-214-fall-2016/sys_path.html)

Useful links: 
- https://bic-berkeley.github.io/psych-214-fall-2016/sys_path.html
- https://docs.python.org/3.5/library/sys.html#sys.path
- https://pymotw.com/3/sys/imports.html#import-path
- https://stackoverflow.com/questions/5137497/find-current-directory-and-files-directory
- https://stackoverflow.com/questions/2860153/how-do-i-get-the-parent-directory-in-python
- https://docs.python.org/3/library/pathlib.html


**Python looks for modules in “sys.path”**
- Python looks for modules in **sys.path**
- Python has a simple algorithm for finding a module with a given name, such as *a_module*. It looks for a file called *a_module.py* in the directories listed in the variable **sys.path**.
- The a_module.py file is in the code directory, and this directory is not in the sys.path list.
- Because **sys.path** is just a Python list, like any other, we can make the import work by appending the code directory to the list.

## Code structure for unittest

In particular, the initial code for each unittest should have, more or less, the following structure:

```python
import sys
import os
from pathlib import Path

import unittest

# Get the absolute path to the parent dir.
sys.path.insert(0, str(Path(os.getcwd()).parent))

# Import my function in my module.
from [name_repository].module1 import func1

class TestFunc1(unittest.TestCase):
“””Unittest for the module1 module”””

    def test1(self):
    “””Test with...”””
    self.assert...


if __name__ == "__main__":
    unittest.main()
```
# Travis CI

<img src="https://user-images.githubusercontent.com/55988954/103585215-f8485f00-4ee2-11eb-9e66-82029a119191.png" width="200" /> 

- Travis CI is a continuous integration service used to create and test software projects hosted on GitHub.
- Travis CI is configured by adding a file named .travis.yml, which is a text file in YAML format, to the root of the repository.
- This file specifies the programming language used, the desired build and test environment (including dependencies that must be installed before the software can be built and tested), and various other parameters.

See also: (https://travis-ci.org/github/lorenzomarini96)

# Repo template:

# Project Title

One Paragraph of project description goes here

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
