#!/bin/bash
python setup.py sdist bdist_wheel
python -m twine check dist/* && python -m twine upload dist/*
