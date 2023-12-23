from clearml import Task, Logger
from clearml.automation import (
DiscreteParameterRange, HyperParameterOptimizer,
RandomSearch, UniformIntegerParameterRange
)

task = Task.init(project_name='disaster_tweets',
                task_name='optimize_module_use_HP',
                task_type=Task.TaskTypes.optimizer,
                reuse_last_task_id=False)

args = {
        'template_task_id': "5ca2e2eca98d4634b0750f6fa72f27c1",
        'run_as_service': False,
}

an_optimizer = HyperParameterOptimizer(
    base_task_id=args['template_task_id'],
    hyper_parameters=[
    UniformIntegerParameterRange('General/batch_size', min_value=4, max_value=16, step_size=4),
    DiscreteParameterRange('General/learning_rate', values=[1e-05, 5e-05, 1e-06, 5e-06]),
    ],
    objective_metric_title='F1',
    objective_metric_series='F1',
    objective_metric_sign='max',
)

an_optimizer.start()
top_exp = an_optimizer.get_top_experiments(top_k=3)
print([t.id for t in top_exp])
task.upload_artifact('top_exp', top_exp)
an_optimizer.wait()
an_optimizer.stop()