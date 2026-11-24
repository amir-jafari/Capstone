Author:Bingly, Arjun
Github: https://github.com/arjbingly/pythonPackaging/
# Example Python Packaging

```
.
├── LICENSE   
├── README.md     
├── main     
│   └── main.py     
├── pyproject.toml
├── src
│   └── mycalculator
│       ├── __init__.py
│       ├── funcs.py
│       └── greet.py
└── tests
    └── test_funcs.py
```

_Note that:_

1. `src/mycalulator` is the package
2. `main` contains all the main files, and unless the `mycalculator` package is installed it can not import its code._

## Steps to create your own package

1. Follow a similar directory structure.
2. Create a `pyproject.toml` file in the main directory, with all the necessary lines.
3. Run the below commands from the main directory.

### To install this package locally,

Run : `pip install .`

### If you want the package to automatically update its code,

Run: `pip -e install .`
