from distutils.core import setup

DESCRIPTION = "Gaussian Process for count data."
LONG_DESCRIPTION = DESCRIPTION
NAME = "GPcounts"
AUTHOR = "Nuha BinTayyash"
AUTHOR_EMAIL = "nuha.bintayyash@postgrad.manchester.ac.uk"
MAINTAINER = "Nuha BinTayyash"
MAINTAINER_EMAIL = "nuha.bintayyash@postgrad.manchester.ac.uk"
DOWNLOAD_URL = 'https://github.com/ManchesterBioinference/GPcounts'
LICENSE = 'MIT'

VERSION = '0.1'

requirements = [
]

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=DOWNLOAD_URL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      packages=['GPcounts'],
      package_data={},
      install_requires=requirements
      )
