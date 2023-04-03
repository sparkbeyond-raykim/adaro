import pandas as pd
import sparkbeyond._api2.classes as sb
import numpy as np
import json
import logging
import argparse
import time
import urllib3
import pickle
import datetime
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(format='\n%(asctime)s %(message)s')
session_log = logging.getLogger('Pipeline Log')

# read config files
with open('SB_Credential.json', 'rb') as b:
    sb_config = json.load(b)
with open('NPHR_config.json', 'rb') as b:
    nphr_config = json.load(b)

dp_credential = sb_config['DP']
client = sb.SparkBeyondClient(base_url = dp_credential['base_url'], api_key= dp_credential['api_key'],
                              verify_ssl_certificate=False)
buckets = nphr_config['buckets']
buckets.append(np.inf)

class NPHR_pipeline_build():
    def __init__(self, args, unit_number):
        session_log.warning(f"Initiating Pipeline Building For the Plant Unit {unit_number}")
        self.unit_number = unit_number
        self.directory_path = f'preprocessed_data/{args.directory_path}'
        self.project_name = f'NPHR_Unit{unit_number}_{args.directory_path.split("nphr_")[-1]}'

    def ingest_file(self):
        train = pd.read_csv(f'{self.directory_path}/train_unit{self.unit_number}.csv')[['datetime', 'load', 'NPHR delta']]
        cont_boiler = pd.read_csv(f'{self.directory_path}/cont_boiler_unit{self.unit_number}.csv')
        train['datetime'] = pd.to_datetime(train['datetime'])
        cont_boiler['datetime'] = pd.to_datetime(cont_boiler['datetime'])

        return train, cont_boiler

    def file_upload(self, train, cont_boiler):
        train_sb = client.upload_dataframe_and_detect_settings(train, target_path='train.csv.gz', project_name=self.project_name, append_contents_hash=False, create_project_if_absent=True)
        cont_boiler_sb = client.upload_dataframe_and_detect_settings(cont_boiler, target_path=f'cont_boiler_unit{self.unit_number}.csv.gz', project_name=self.project_name, append_contents_hash=False, create_project_if_absent=True)

        return train_sb, cont_boiler_sb

    def define_contexts(self, cont_boiler_sb):
        ts_boiler = sb.AsTimeSeries(input=cont_boiler_sb, time_column='datetime', time_windows=[sb.AddTimeWindow(date_column='datetime', window_size=30, time_unit=sb.SBTimeUnit.Minutes)], name_override=f'controllable_boiler_unit_{self.unit_number}')

        return ts_boiler

    def feature_summary_report(self, revision_mapping):
        all_features = []
        for bucket, metadata in revision_mapping.items():
            try:
                revision = metadata['revision_id']
                model = client.revision(project_name=self.project_name, revision_id=revision)
                features = model.features()
                features['Load Bucket'] = bucket
                feature_data = pd.DataFrame(client._client.get(f'/api/v1/projects/{self.project_name}/revisions/{revision}/features').json()['wrappedFeatures'])
                features['TAG'] = feature_data['usedContexts'].apply(lambda x: x[0]['valueColumnName'])
                features['DP Project'] = self.project_name
                features['DP Train Date'] = metadata['dp_train_date']
                features['Training Data Start Date'] = metadata['trainset_start_date']
                features['Training Data End Date'] = metadata['trainset_end_date']
                features['PP Unit'] = metadata['pp_unit']

                all_features.append(features)

            except:
                session_log.warning(f"Learning Failed for the Following Model"
                                    f"proejct_name: {self.project_name}"
                                    f"revidion id: {revision}"
                                    f"bucket_id: {bucket}")
                continue

        all_features_df = pd.concat(all_features)
        feature_summary = all_features_df[['DP Project','PP Unit','DP Train Date', 'Training Data Start Date', 'Training Data End Date','Load Bucket', 'feature', 'Target Mean Shift', '% support', 'TAG']].reset_index(drop=True)
        feature_summary.rename(columns={'Target Mean Shift': 'NPHR Shfit'}, inplace=True)

        return feature_summary


    def build_pipeline(self):
        session_log.warning(f"Uploading Files to DP")

        # ingest file
        train, cont = self.ingest_file()

        # upload to dp
        train_sb, cont_boiler_sb = self.file_upload(train, cont)

        # define time series context
        ts_boiler = self.define_contexts(cont_boiler_sb)

        # sb pipeline configuration
        problem_definition = sb.ProblemDefinition(target_column='NPHR delta', temporal_split_column='datetime')
        feature_generator_settings = sb.FeatureGenerationSettings(
            solving_settings=sb.SolvingSettings(
                function_whitelist=['average', 'window'],
                function_blacklist=['minSlope', 'maxSlope', 'maxAbsSlope', 'frequencyOfMode', 'sumOfValues', 'maxAbsSlope', 'std']),
            generation_methods=sb.FeatureGenerationMethods(range=False, numeric_equality=False, equality=False),
            column_subset_settings=sb.ColumnSubsetSettings(expanded_time_series=False),
            feature_selection_method=sb.FeatureSelectionMethods.SimpleFeaturesFirst,
            deduplication_settings=sb.FeatureDeduplicationSettings(similarity_threshold=0.95),
            monotone_features_only=True,
            local_top_feature_count=1,
            ban_date_column_features=True,
            normalize_numeric_features=False
            )

        learning_settings = sb.LearningSettings(problem_definition=problem_definition, feature_generator_settings=feature_generator_settings)

        project_metadata = {}
        bucket_num = 1

        # start model building
        today_date = datetime.datetime.strftime(datetime.datetime.today().date(), '%Y-%m-%d')
        trainset_start_date = datetime.datetime.strftime(pd.to_datetime(train['datetime']).min(),'%Y-%m-%d')
        trainset_end_date = datetime.datetime.strftime(pd.to_datetime(train['datetime']).max(), '%Y-%m-%d')

        session_log.warning(f"Started Model Building")
        for low_lim, high_lim in zip(buckets, buckets[1:]):
            raw_train = client.create_pipeline(project_name=self.project_name, input_data=train_sb)

            if high_lim == np.inf:
                bucket_train = raw_train.filter_by_column_values(column_name='load', operator='>=', values=low_lim).exclude_columns('load')
            else:
                bucket_train = raw_train.filter_by_column_values(column_name='load', operator='>=', values=low_lim).filter_by_column_values(column_name='load', operator='<',
                                                                                                                                            values=high_lim).exclude_columns('load')

            model = client.learn(
                project_name=self.project_name,
                training_set=bucket_train,
                test_set=bucket_train,
                context_datasets=[ts_boiler],
                learning_settings=learning_settings,
                target_column='NPHR delta',
                revision_description=f'LB: {low_lim} to {high_lim}',
                print_notifications_log_until_done=False,
                print_job_state_until_done=False,
                open_in_browser=False
            )

            if high_lim == np.inf: high_lim = 'up'
            project_metadata[f'LB{bucket_num}_{low_lim}_{high_lim}'] = {'revision_id': model.revision_id,
                                                                        'dp_train_date': today_date,
                                                                        'pp_unit': self.unit_number,
                                                                        'trainset_start_date': trainset_start_date,
                                                                        'trainset_end_date': trainset_end_date,
                                                                        }
            bucket_num += 1

        with open (f'tmp/project_meta_data/{self.project_name}_project_meta.obj', 'wb') as f:
            pickle.dump(project_metadata, f)

        while model.is_complete() == False:
            time.sleep(5)

        session_log.warning(f"Generating feature summary report")
        feature_summary_report = self.feature_summary_report(project_metadata)
        feature_summary_report.to_csv(f'feature_reports/{self.project_name}_feature_summary_report.csv', index=False)



def main():
    # Take user defined input path
    parser = argparse.ArgumentParser(description="testing")
    parser.add_argument('--directory_path',  required=True, help="file_path_to_preprocessed_data")
    args = parser.parse_args()

#     args = {'directory_path': 'nphr_03272023'}

    # build pipeline for plant unit 1
    pipeline_builder = NPHR_pipeline_build(args, unit_number=1)
    pipeline_builder.build_pipeline()

    # build pipeline for plant unit 2
    pipeline_builder = NPHR_pipeline_build(args, unit_number=2)
    pipeline_builder.build_pipeline()


if __name__ == "__main__":
    main()



