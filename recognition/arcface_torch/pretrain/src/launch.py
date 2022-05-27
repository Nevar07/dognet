import logging

logging.basicConfig(level=logging.INFO)
from anylearn.applications.quickstart import quick_train

from anylearn.config import init_sdk

from anylearn.interfaces.resource import SyncResourceUploader

init_sdk('http://111.202.73.142/', "18211067198", "cbm19991111")

task, _, _, _ = quick_train(algorithm_name="dog_ct_pt_bs1024_lr0.004", algorithm_dir="../src",
                            entrypoint="python main.py",
                            output="checkpoints",
                            dataset_id="DSET9c66d44d11eca6b05ae046c6ae66", dataset_hyperparam_name='data_dir',
                            algorithm_force_update=True,
                            hyperparams={
                                'save_dir': 'checkpoints',
                            },
                            mirror_name="QUICKSTART_PYTORCH1.9.0_CUDA11",
                            resource_uploader=SyncResourceUploader(),
                            resource_request=[{'DL2022-1':
                                                   {'RTX-3090-unique':4,
                                                    'CPU': 16, 
                                                    'Memory': 64,}
                                               }],
                            )
print(task)
