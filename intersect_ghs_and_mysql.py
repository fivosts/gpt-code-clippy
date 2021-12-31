import glob
import pandas
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ghs_csv_files", nargs='+', required=True)
    parser.add_argument("--mysql_tsv_files", nargs='+', required=True)
    parser.add_argument("--output_csv_file", required=True)

    args = parser.parse_args()

    ghs = []
    ghs_df = pandas.concat([pandas.read_csv(fname) for glob_path in args.ghs_csv_files for fname in glob.glob(glob_path)], axis=0)

    ghs_df['name'] = ghs_df['name'].str.lower()

    mysql_df = pandas.concat([pandas.read_csv(fname, sep='\t') for glob_path in args.mysql_tsv_files for fname in glob.glob(glob_path)], axis=0)

    mysql_df['name'] = mysql_df['url'].apply(
        lambda url: '/'.join(url.split('/')[-2:]).lower(),
    )

    joined = ghs_df.merge(mysql_df, on='name', how='inner')

    # since we also have joined['main_language']
    del(joined['language'])
    del(joined['url'])

    joined['approximate_num_comments'] = joined['num_comments']
    del(joined['num_comments'])

    joined = joined.sort_values('approximate_num_comments', ascending=False)

    print(f"{len(joined):,} repos; {sum(joined['approximate_num_comments']):,} comments approx")

    joined.to_csv(args.output_csv_file, index=False)
