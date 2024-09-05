'''
Weaviateを使用して、ベクトルDBを作成するためのスクリプト。
ベクトルDBの作成、データの追加を行う
'''

import weaviate
from weaviate.util import generate_uuid5
import weaviate.classes.config as wc
import os
import sys

import pandas as pd
from tqdm import tqdm

from RAG_Agriculture.utils.data_load import load_json


def main():
    config_dict = load_json('./config.json')

    if config_dict['embedding_model']=='openai':
        api_key = os.environ.get('OPENAI_API_KEY')
        headers = {"X-OpenAI-Api-Key": api_key}  # Replace with your OpenAI API key
        client = weaviate.connect_to_local(headers=headers)

    else:
        print('Invalid embedding model. Please check embedding_model in the config file')
        sys.exit()
    
    print('------connected------')

    try:
        client.collections.delete(config_dict['weaviate']['schema']['class'])
        client.collections.create_from_dict(config_dict['weaviate']['schema'])
        print('------collection created------')

        collection = client.collections.get(config_dict['weaviate']['schema']['class'])
        df = pd.read_csv(config_dict['weaviate']['data_path'])

        with collection.batch.dynamic() as batch:
            for i, batch_df in tqdm(df.iterrows()):
                obj = {k:batch_df[v] for k,v in config_dict['weaviate']['target_columns'].items()}

                batch.add_object(
                    properties=obj,
                    uuid=generate_uuid5(batch_df['作業名'])
                )

                # Note these are outside the `with` block - they are populated after the context manager exits
                failed_objs_a = client.batch.failed_objects  # Get failed objects from the first batch import
                failed_refs_a = client.batch.failed_references  # Get failed references from the first batch import

        # Check for failed objects
        if len(collection.batch.failed_objects) > 0:
            print(collection.batch.failed_objects)
            print(f"Failed to import {len(collection.batch.failed_objects)} objects")
    finally:
        client.close()

if __name__ == '__main__':
    main()
