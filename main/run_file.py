import cProfile
from utils.file_handler import FileHandler
import logging
import sys
import os
import main.globals as g
from top_level import TopLevel
timestamp = FileHandler.get_datetime()
g.TIMESTAMP = timestamp

fname = os.path.join(g.LOGS_PROFS, f'{g.TIMESTAMP}')
os.makedirs(g.LOGS_PROFS, exist_ok=True)

tl = TopLevel(override_params=None)
# exec_str = 'tl.single_batch_train()'
# exec_str = 'tl.single_epoch_train()'
# exec_str = 'tl.train()'
# exec_str = 'tl.evaluate_model()'
exec_str = 'tl.run_experiment()'


cProfile.run(exec_str, filename=fname+'.prof')


