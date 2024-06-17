import shutil
import os

from dagshub import get_repo_bucket_client

with open("dagshub-token.txt", "r") as f:
    token = f.read()
    if token:
        from dagshub.auth import add_app_token
        add_app_token(token)

repo = "mizoru/ORD"
s3 = get_repo_bucket_client(repo)

s3.download_file(
    Bucket="ORD",  
    Key="whole_dataset_0.9.csv",
    Filename="data/whole_dataset.csv",
)

s3.download_file(
    Bucket="ORD",  
    Key="unchecked_packed_dataset_0.7.zip",
    Filename="unchecked_packed_dataset.zip",
)

s3.download_file(
    Bucket="ORD",  
    Key="checked_packed_dataset_0.7.zip",
    Filename="checked_packed_dataset.zip",
)


extract_dir = './data/'
# !unzip -UU checked_packed_dataset.zip
# subprocess.run('unzip -UU checked_packed_dataset.zip'.split())
shutil.unpack_archive("checked_packed_dataset.zip", extract_dir)
# !unzip -UU unchecked_packed_dataset.zip
# subprocess.run('unzip -UU unchecked_packed_dataset.zip'.split())
shutil.unpack_archive("unchecked_packed_dataset.zip", extract_dir)


# !rm unchecked_packed_dataset.zip checked_packed_dataset.zip
os.remove("unchecked_packed_dataset.zip")
os.remove("checked_packed_dataset.zip")