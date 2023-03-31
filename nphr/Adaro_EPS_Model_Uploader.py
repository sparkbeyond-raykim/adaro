import sparkbeyond._api2.classes as sb
from sparkbeyond.predictionserver.api import PredictionServerClient
import pickle
import argparse
import urllib3
import json
import logging
import tqdm
import os

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(format='\n%(asctime)s %(message)s')
session_log = logging.getLogger('Pipeline Log')

with open('SB_Credential.json', 'rb') as b:
    sb_config = json.load(b)

dp_credential = sb_config['DP']
client = sb.SparkBeyondClient(base_url = dp_credential['base_url'], api_key= dp_credential['api_key'],
                              verify_ssl_certificate=False)

eps_credential = sb_config['EPS']
eps_client = PredictionServerClient(url=eps_credential['base_url'], refresh_token=eps_credential['api_key'], validate_ssl_certificate=False)

class eps_file_uploader():
    def __init__(self, args, unit_number):
        self.unit_number = unit_number
        self.project_name = f'NPHR_Unit{unit_number}_{args["directory_path"].split("nphr_")[-1]}'
        meta_data_path = f'{self.project_name}_project_meta.obj'
        with open(f'tmp/project_meta_data/{meta_data_path}', 'rb') as f:
            self.project_meta = pickle.load(f)
            
    def download_model_objects_from_dp(self):
        session_log.warning(f"Downloading Model Objects From DP...")
        for bucket_id, revision_id in self.project_meta.items():
            model_file_name = f'{self.project_name}_rev_{revision_id}_bucket_{bucket_id}.zip'
            model = client.revision(self.project_name, revision_id)
            model.download_model(local_path=f'tmp/model_objects/{model_file_name}', with_contexts=True)

    def upload_model_objects_to_dp(self):
        session_log.warning(f"Uploading Model Objects to EPS...")
        for bucket_id, revision_id in self.project_meta.items():
            model_file_name = f'{self.project_name}_rev_{revision_id}_bucket_{bucket_id}.zip'
            group_name = f'{self.project_name}_rev_{revision_id}_bucket_{bucket_id}'
            eps_client.upload_group(f'tmp/model_objects/{model_file_name}', group_name=group_name, force=True,
                                    enforce_version_compatibility=False)
def main():
    # Take user defined input path
    parser = argparse.ArgumentParser(description="testing")
    parser.add_argument('--directory_path',  required=True, help="file_path_to_preprocessed_data")
    args = {'directory_path': 'nphr_03272023'}

    # download and upload model objects for plant unit 1
    uploader = eps_file_uploader(args, 1)
    uploader.download_model_objects_from_dp()
    uploader.upload_model_objects_to_dp()

if __name__ == "__main__":
    main()


