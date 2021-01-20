import cProfile
from utils.file_handler import FileHandler
import logging
import sys
import os
import globals as g
from top_level import TopLevel
timestamp = FileHandler.get_datetime()
g.TIMESTAMP = timestamp

fname = os.path.join(g.LOGS_PROFS, f'{g.TIMESTAMP}')

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler(fname+'.log'),
                              logging.StreamHandler(sys.stdout)])

logger = logging.getLogger(__name__)
logger.info('Initiate Logger')

override_params = {}

tl = TopLevel()
exec_str = 'tl.single_epoch_train()'

cProfile.run(exec_str, filename=fname+'.prof')


