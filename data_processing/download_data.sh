#! /bin/bash
OUT_DIR=$1
shift
REPO_SHARDS=$@
ID=0
for shard in ${REPO_SHARDS}; do
    echo "Processing data from ${shard}"
    # python convert_to_gh_downloader_format.py $shard $OUT_DIR
    ID=$(($ID + 1))
    # cp download_repo_text.py $OUT_DIR
    # cp Programming_Languages_Extensions.json $OUT_DIR
    # cd $OUT_DIR
    # python download_repo_text.py --verbose
    python download_repo_text.py $shard $OUT_DIR --verbose
    mv $OUT_DIR/"github_data" $OUT_DIR/"github_data_${ID}"
    # cd -
    echo "Finished processsing data from ${shard}"
done
# python convert_to_gh_downloader_format.py $COMBINED_REPOS $OUT_DIR

# cd $OUT_DIR
# python download_repo_text.py
# cd -
