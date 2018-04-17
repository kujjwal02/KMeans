import argparse
from kmeans import KMeansRobust


def data_file(filename, delimiter=' '):
    '''
    Read the file and converts the text
    to list of list containing numbers
    '''
    result = []
    with open(filename) as file:
        for line in file:
            row = line.strip().split(delimiter)
            result.append([float(num) for num in row])

    return result


def write_file(filename, data, delimiter=' '):
    '''
    Write the list of lists to a file
    delimited by the specified delimiter
    '''
    with open(filename, 'w+') as file:
        for row in data:
            file.write(delimiter.join(str(num) for num in row))
            file.write('\n')


def get_cmd_arguments():
    parser = argparse.ArgumentParser(
        description='''Compute cluster centroids of the dataset
        and save it in output file''')
    parser.add_argument('input', type=str,
                        help='file that contains the dataset')
    parser.add_argument('n_clusters', type=int,
                        help='Number of clusters to find')
    parser.add_argument('--output', type=str, default='clusters.txt',
                        help='file in which to store the output')

    return parser.parse_args()


def run():
    args = get_cmd_arguments()

    # load data
    data = data_file(args.input)
    
    # Run kmeans on data
    kmeans = KMeansRobust(n_clusters=args.n_clusters)
    kmeans.fit(data)

    # Save the clusters to output file
    write_file(args.output, kmeans.cluster_centers_)


if __name__ == '__main__':
    run()
