import cProfile
import os

import main.globals as g
from top_level import TopLevel
from utils.file_handler import FileHandler
task_id = None

timestamp = FileHandler.get_datetime()
g.TIMESTAMP = timestamp
task = None
fname = os.path.join(g.LOGS_PROFS, f'{g.TIMESTAMP}')
os.makedirs(g.LOGS_PROFS, exist_ok=True)
# TODO: add task name
# task_id = '"2176639f89c84d2bb124c6d83d5df25d"'
tl = TopLevel()
# exec_str = 'tl.single_batch_train()'
# exec_str = f'tl.evaluate_model(task_id={task_id})'
exec_str = f'tl.train(task_id={task_id})'

cProfile.run(exec_str, filename=fname + '.prof')


