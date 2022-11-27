mkdir -p data
aws s3 cp s3://drivendata-competition-biomassters-public-us/features_metadata.csv data/ --no-sign-request
aws s3 cp s3://drivendata-competition-biomassters-public-us/train_agbm_metadata.csv data/ --no-sign-request
