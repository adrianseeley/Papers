class Dataset
{
    public string[] features;
    public List<Datapoint> datapoints;

    public Dataset(string path)
    {
        // get all lines of csv file
        string[] lines = File.ReadAllLines(path);

        // the first line contains the headers, we skip the timestamp column
        features = lines[0].Split("\",\"").Skip(1).ToArray();

        // each feature has a prefix of I have ... and the real question is in []
        for (int i = 0; i < features.Length; i++)
        {
            features[i] = features[i].Split('[')[1].Split(']')[0];
        }

        // create a list of parsed datapoints
        datapoints = new List<Datapoint>();

        // iterate through datapoint lines (skipping header line)
        for (int i = 1; i < lines.Length; i++)
        {
            string[] parts = lines[i].Split(',');

            // the index is just a number we assign to id the datapoint
            int index = i - 1;

            // we have a boolean vector for each feature (minus 1 for the timestamp)
            bool[] vector = new bool[parts.Length - 1];

            // "YES" is true and "NO" is false, note that quotes are in file
            for (int j = 1; j < parts.Length; j++)
            {
                vector[j - 1] = parts[j] == "\"YES\"";
            }

            // add parsed datapoint to list
            datapoints.Add(new Datapoint(index, vector));
        }
    }
}

class Datapoint
{
    public int index;
    public bool[] vector;

    public Datapoint(int index, bool[] vector)
    {
        this.index = index;
        this.vector = vector;
    }
}

class DifferenceTable : Dictionary<int, Dictionary<int, int>>
{
    public DifferenceTable(Dataset dataset)
    {
        // iterate through all pairs of datapoints
        for (int i = 0; i < dataset.datapoints.Count; i++)
        {
            for (int j = i + 1; j < dataset.datapoints.Count; j++)
            {
                // calculate the difference between the two datapoints
                int difference = 0;
                for (int k = 0; k < dataset.features.Length; k++)
                {
                    if (dataset.datapoints[i].vector[k] != dataset.datapoints[j].vector[k])
                    {
                        difference++;
                    }
                }

                // add the difference to the table (low, high, difference)
                if (!ContainsKey(i))
                {
                    Add(i, new Dictionary<int, int>());
                }
                this[i][j] = difference;

                // add the difference to the table (high, low, difference)
                if (!ContainsKey(j))
                {
                    Add(j, new Dictionary<int, int>());
                }
                this[j][i] = difference;

                // note: we add both directions as a small lookup optimization
            }
        }
    }
}

class Cluster
{
    public List<Datapoint> datapoints;
    public int differenceCount;
    public int differenceSum;
    public float differenceAverage;
    public float weightedDifferenceAverage;

    public Cluster()
    {
        datapoints = new List<Datapoint>();
        differenceCount = 0;
        differenceSum = 0;
        differenceAverage = 0f;
        weightedDifferenceAverage = 0f;
    }

    public float MockAddDatapoint(Datapoint datapoint, DifferenceTable differanceTable)
    {
        // copy cluster metrics to mock update them
        int mockDifferenceCount = differenceCount;
        int mockDifferenceSum = differenceSum;

        // iterate through existing datapoints in cluster to calculate mock metrics
        foreach (Datapoint existingDatapoint in datapoints)
        {
            // get the difference between the new datapoint and the existing datapoint
            int difference = differanceTable[datapoint.index][existingDatapoint.index];

            // update the mock metrics
            mockDifferenceCount++;
            mockDifferenceSum += difference;
        }

        // calculate the mock average
        float mockDifferenceAverage = ((float)mockDifferenceSum) / ((float)mockDifferenceCount);

        // calculate the mock weighted average
        float mockWeightedDifferenceAverage = mockDifferenceAverage * (float)(datapoints.Count + 1);

        // return the mock weighted average
        return mockWeightedDifferenceAverage;
    }

    public void AddDatapoint(Datapoint datapoint, DifferenceTable differanceTable)
    {
        // iterate through existing datapoints in cluster to update metrics
        foreach (Datapoint existingDatapoint in datapoints)
        {
            // get the difference between the new datapoint and the existing datapoint
            int difference = differanceTable[datapoint.index][existingDatapoint.index];

            // update the metrics
            differenceCount++;
            differenceSum += difference;
        }

        // add the datapoint to the cluster
        datapoints.Add(datapoint);

        // if we have more than one datapoint update averages
        if (datapoints.Count > 1)
        {
            // calculate the average
            differenceAverage = ((float)differenceSum) / ((float)differenceCount);

            // calculate the weighted average
            weightedDifferenceAverage = differenceAverage * ((float)datapoints.Count);
        }
    }
}

class Program
{
    static void ClusterSearch(int clusterCount, Dataset dataset, DifferenceTable differenceTable, out List<Cluster>? bestClusters, out float bestTotalWeightedDifferenceAverage, out int[] bestClusterSeedIterator)
    {
        // create a cluster seed iterator
        int[] clusterSeedIterator = new int[clusterCount];

        // record the best clusters so far
        bestClusters = null;
        bestTotalWeightedDifferenceAverage = float.MaxValue;
        bestClusterSeedIterator = new int[clusterCount];

        // iterate through all possible cluster seeds
        for (; ; )
        {
            // if the iterator has no duplicate values we can use it as a seed
            if (clusterSeedIterator.Distinct().Count() == clusterCount)
            {
                // build clusters
                List<Cluster> clusters = BuildClusters(clusterSeedIterator, dataset, differenceTable, out float totalWeightedDifferenceAverage);

                // if we have no best, or this is better than the best, record it
                if (bestClusters == null || totalWeightedDifferenceAverage < bestTotalWeightedDifferenceAverage)
                {
                    bestClusters = clusters;
                    bestTotalWeightedDifferenceAverage = totalWeightedDifferenceAverage;
                    Array.Copy(clusterSeedIterator, bestClusterSeedIterator, clusterCount);
                }
            }

            // iterate
            for (int i = 0; i < clusterCount; i++)
            {
                clusterSeedIterator[i]++;
                if (clusterSeedIterator[i] < dataset.datapoints.Count)
                {
                    break;
                }
                clusterSeedIterator[i] = 0;
            }
        }
    }

    static List<Cluster> BuildClusters(int[] clusterSeedIterator, Dataset dataset, DifferenceTable differenceTable, out float totalWeightedDifferenceAverage)
    {
        // create the clusters using the seeds
        List<Cluster> clusters = new List<Cluster>();
        for (int i = 0; i < clusterSeedIterator.Length; i++)
        {
            Cluster cluster = new Cluster();
            cluster.AddDatapoint(dataset.datapoints[clusterSeedIterator[i]], differenceTable);
            clusters.Add(cluster);
        }

        // add all datapoints, greedily minimizing the weighted average of all clusters
        // this value always starts at 0 since each cluster has 1 datapoint and therefore
        // no differences yet
        totalWeightedDifferenceAverage = 0f;
        int totalDatapoints = clusterSeedIterator.Length;
        for (int i = 0; i < dataset.datapoints.Count; i++)
        {
            // skip seed datapoints
            if (clusterSeedIterator.Contains(i))
            {
                continue;
            }

            // increment total datapoints
            totalDatapoints++;

            // find the best cluster to add it to
            float bestTotalWeightedDifferenceAverage = float.MaxValue;
            int bestCluster = -1;
            for (int j = 0; j < clusters.Count; j++)
            {
                // mock add the datapoint to the cluster
                float mockWeightedDifferenceAverage = clusters[j].MockAddDatapoint(dataset.datapoints[i], differenceTable);

                // calculate the total weighted difference average
                float mockTotalWeightedDifferenceAverage = mockWeightedDifferenceAverage;
                for (int k = 0; k < clusters.Count; k++)
                {
                    if (k != j)
                    {
                        mockTotalWeightedDifferenceAverage += clusters[k].weightedDifferenceAverage;
                    }
                }
                mockTotalWeightedDifferenceAverage /= ((float)totalDatapoints);

                // if this is the best cluster so far, record it
                if (bestCluster == -1 || mockTotalWeightedDifferenceAverage < bestTotalWeightedDifferenceAverage)
                {
                    bestTotalWeightedDifferenceAverage = mockTotalWeightedDifferenceAverage;
                    bestCluster = j;
                }
            }

            // add the datapoint to the best cluster
            clusters[bestCluster].AddDatapoint(dataset.datapoints[i], differenceTable);

            // update the total weighted difference average
            totalWeightedDifferenceAverage = bestTotalWeightedDifferenceAverage;
        }

        // return the clusters (note that the totalWeightedDifferenceAverage is an out parameter)
        return clusters;
    }


    static void Main(string[] args)
    {
        const string infile = "FT-163.csv";
        const int clusterCount = 3;

        Dataset dataset = new Dataset(infile);
        DifferenceTable differenceTable = new DifferenceTable(dataset);
        ClusterSearch(clusterCount, dataset, differenceTable, out List<Cluster>? bestClusters, out float bestTotalWeightedDifferenceAverage, out int[] bestClusterSeedIterator);

        if (bestClusters == null)
        {
            throw new Exception("No best clusters found");
        }
        Console.WriteLine($"Best total weighted difference average: {bestTotalWeightedDifferenceAverage}");
        for (int i = 0; i < bestClusters.Count; i++)
        {
            Cluster cluster = bestClusters[i];
            Console.WriteLine($"Cluster: {i}, Count: {cluster.datapoints.Count}, DA: {cluster.differenceAverage}");
        }
    }
}