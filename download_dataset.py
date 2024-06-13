from dagshub import get_repo_bucket_client

with open("dagshub-token.txt", "r") as f:
    token = f.read()

if not token:
    from dagshub.auth import get_token
    
    token = get_token()

    with open("dagshub-token.txt", "w") as f:
        f.write(token)


s3 = get_repo_bucket_client("mizoru/ORD")

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

s3.download_file(
    Bucket="ORD",  
    Key="whole_dataset_0.9.csv",
    Filename="whole_dataset.csv",
)

!unzip -UU -q checked_packed_dataset.zip
!unzip -UU -q unchecked_packed_dataset.zip

!rm unchecked_packed_dataset.zip checked_packed_dataset.zip