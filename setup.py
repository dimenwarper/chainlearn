#! /usr/bin/env python

DESCRIPTION = 'chainlearn: A sprinkle of syntax sugar for pandas/sklearn'
LONG_DESCRIPTION = '''\
 Mini module with some syntax sugar utilities for pandas and sklearn, draws inspiration on some R tidyverse tools.
'''

DISTNAME = 'chainlearn'
MAINTAINER = 'Pablo Cordero'
MAINTAINER_EMAIL = 'dimenwarper@gmail.com'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/dimenwarper/chainlearn/'
VERSION = '0.0.2.dev'

INSTALL_REQUIRES = [
    'scikit-learn>=0.19.0',
    'pandas>=0.15.2',
]

PACKAGES = [
    'chainlearn',
]

try:
    from setuptools import setup
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

if __name__ == '__main__':

    setup(
        name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        url=DOWNLOAD_URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        install_requires=INSTALL_REQUIRES,
        packages=PACKAGES,
    )
