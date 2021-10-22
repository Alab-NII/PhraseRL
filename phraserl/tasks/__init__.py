from .multiwoz.multiwoz import MultiWozTask

TASKS = {"multiwoz": MultiWozTask}


def get_task_cls(name):
    return TASKS[name]
