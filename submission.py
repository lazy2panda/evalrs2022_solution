"""

    Entry point to run the submission code: this is a modified version from the template
    provided in: https://github.com/RecList/evalRS-CIKM-2022/blob/main/submission.py

    The main change is the model class init, since the custom model need some parameters.

    Check the original template and evalRS repo (https://github.com/RecList/evalRS-CIKM-2022) for
    the fully commented version, and detailed instructions on the competition.
"""

import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv('upload.env', verbose=True)


EMAIL = os.getenv('EMAIL')
assert EMAIL != '' and EMAIL is not None
BUCKET_NAME = os.getenv('BUCKET_NAME')
PARTICIPANT_ID = os.getenv('PARTICIPANT_ID')
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')

DATASET_SEED = os.getenv('DATASET_SEED')
MODEL_SEED = os.getenv('MODEL_SEED')

DATASET_SEED=int(DATASET_SEED) if DATASET_SEED is not None else DATASET_SEED
MODEL_SEED=int(MODEL_SEED) if MODEL_SEED is not None else MODEL_SEED



# run the evaluation loop when the script is called directly
if __name__ == '__main__':
    # import the basic classes
    from evaluation.EvalRSRunner import EvalRSRunner
    # import my custom recList, containing the custom test
    from evaluation.EvalRSRecList import myRecList
    from evaluation.EvalRSRunner import ChallengeDataset
    from submission.MyModel import MyModel
    print('\n\n==== Starting evaluation script at: {} ====\n'.format(datetime.utcnow()))
    # load the dataset
    print('\n\n==== Loading dataset at: {} ====\n'.format(datetime.utcnow()))
    dataset = ChallengeDataset(force_download=True,seed=DATASET_SEED)
    print('\n\n==== Init runner at: {} ====\n'.format(datetime.utcnow()))
    # run the evaluation loop
    runner = EvalRSRunner(
        dataset=dataset,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        participant_id=PARTICIPANT_ID,
        bucket_name=BUCKET_NAME,
        email=EMAIL
        )
    print('==== Runner loaded, starting loop at: {} ====\n'.format(datetime.utcnow()))
    my_model = MyModel(sample_num=8,
    cold_num=9000,
    seed=MODEL_SEED,
    drop_num=1,
    worker=7,
    diversity_flag=True,
    diversity_keeptop=10,
    diversity_history=500)

    # run evaluation with your model
    runner.evaluate(
        model=my_model,
        custom_RecList=myRecList # pass my custom reclist to the runner!
        ,upload=True
        )
    print('\n\n==== Evaluation ended at: {} ===='.format(datetime.utcnow()))
