import os
from infoclus import InfoClus


RELATIVE_DATA_PATH = os.path.join(os.path.pardir, 'data')
DATA_SET_NAME = 'german_socio_eco'

infoclus = InfoClus(DATA_SET_NAME, RELATIVE_DATA_PATH)
# infoclus.optimise()

