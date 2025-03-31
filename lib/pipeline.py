import wandb
import os
import multiprocessing
import collections

import numpy as np

Worker = collections.namedtuple("Worker", ("queue", "process"))
WorkerInitData = collections.namedtuple(
    "WorkerInitData", ("num", "config")
)

def main(config, main_worker):
    # Spin up workers before calling wandb.init()
    # Workers will be blocked on a queue waiting to start

    sweep_run = wandb.init(project="MGRADA", config=config)
    config = dict(sweep_run.config)
    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = sweep_run.get_sweep_url()
    project_url = sweep_run.get_project_url()
    sweep_group_url = "{}/groups/{}".format(project_url, sweep_id)
    sweep_run.notes = sweep_group_url
    sweep_run.name = sweep_run.id

    sweep_q = multiprocessing.Queue()
    workers = []
    for num in range(config["folds"]):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=main_worker, kwargs=dict(sweep_q=sweep_q, worker_q=q)
        )
        p.start()
        workers.append(Worker(queue=q, process=p))

    metrics = []
    for num in range(config["folds"]):
        worker = workers[num]
        # start worker
        worker.queue.put(
            WorkerInitData(
                num=num,
                config=config,
            )
        )

    for num in range(config["folds"]):
        worker = workers[num]
        # get metric from worker
        result = sweep_q.get()
        # wait for worker to finish
        worker.process.join()
        # log metric to sweep_run
        metrics.append(result.best_accuracy)
    metrics = np.array(metrics)
    sweep_run.log(dict(accuracy_mean=metrics.mean(), accuracy_std=metrics.std()))
    wandb.join()

    print("*" * 40)
    print("Sweep URL:       ", sweep_url)
    print("Sweep Group URL: ", sweep_group_url)
    print("*" * 40)