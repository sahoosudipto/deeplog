import os
import json
import logging
from clearml import Task, Logger
from deeplog.deeplog import train, model_fn, input_fn, predict_fn, save_model
from example.preprocess import deeplog_df_transfer, deeplog_file_generator

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

# Define constants for directories and ClearML project/task names
MODEL_DIR = './model/'
DATA_DIR = './data/'
PROJECT_NAME = 'DeepLog Anomaly Detection'
TASK_NAME = 'Training & Inference'

# Create a ClearML task
Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)

def preprocess_logs(input_dir, output_dir):
    """
    Preprocess log files into structured datasets for training and testing.
    """
    from spellpy import spell

    log_format = '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>'
    log_main = 'open_stack'
    tau = 0.5

    parser = spell.LogParser(
        indir=input_dir, outdir=output_dir, log_format=log_format, logmain=log_main, tau=tau
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for log_name in ['openstack_abnormal.log', 'openstack_normal2.log', 'openstack_normal1.log']:
        parser.parse(log_name)

    logger.info("Log preprocessing completed.")


def prepare_datasets(output_dir):
    """
    Transform logs into datasets suitable for DeepLog.
    """
    df = pd.read_csv(f'{output_dir}/openstack_normal1.log_structured.csv')
    df_normal = pd.read_csv(f'{output_dir}/openstack_normal2.log_structured.csv')
    df_abnormal = pd.read_csv(f'{output_dir}/openstack_abnormal.log_structured.csv')

    event_id_map = {event_id: i for i, event_id in enumerate(df['EventId'].unique(), 1)}

    deeplog_train = deeplog_df_transfer(df, event_id_map)
    deeplog_file_generator('train', deeplog_train)

    deeplog_test_normal = deeplog_df_transfer(df_normal, event_id_map)
    deeplog_file_generator('test_normal', deeplog_test_normal)

    deeplog_test_abnormal = deeplog_df_transfer(df_abnormal, event_id_map)
    deeplog_file_generator('test_abnormal', deeplog_test_abnormal)

    logger.info("Datasets preparation completed.")


def train_and_save_model(args):
    """
    Train the DeepLog model and save it for future inference.
    """
    train(args)
    logger.info("Model training completed.")


def load_model_and_infer(model_dir, input_data, threshold):
    """
    Load a pre-trained model and perform inference.
    """
    model_info = model_fn(model_dir)
    prediction = predict_fn(input_data, model_info)

    # Check for anomalies
    anomalies = [idx for idx, value in enumerate(prediction['predict_list']) if value == 1]
    if len(anomalies) >= threshold:
        logger.warning(f"Anomaly detected in {len(anomalies)} sequences.")
    else:
        logger.info("No anomalies detected.")

    return prediction

if __name__ == '__main__':
    # Example usage for pipeline

    # Step 1: Preprocess logs
    preprocess_logs(input_dir='./data/OpenStack/', output_dir='./openstack_result/')

    # Step 2: Prepare datasets
    prepare_datasets(output_dir='./openstack_result/')

    # Step 3: Training
    class Args:
        batch_size = 64
        epochs = 50
        window_size = 10
        input_size = 1
        hidden_size = 64
        num_layers = 2
        num_classes = 5  # Adjust based on dataset
        num_candidates = 3
        seed = 42
        hosts = ['127.0.0.1']
        current_host = '127.0.0.1'
        model_dir = MODEL_DIR
        data_dir = DATA_DIR
        num_gpus = 0
        local = True

    args = Args()
    train_and_save_model(args)

    # Step 4: Inference
    test_data = {'line': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}  # Example input
    prediction = load_model_and_infer(MODEL_DIR, test_data, threshold=2)

    # Log inference results to ClearML
    Logger.current_logger().report_text(f"Inference results: {json.dumps(prediction)}")

    logger.info("Pipeline execution completed.")
