using System.Text;

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

        // count the activations of each feature
        List<int> activations = new List<int>();
        for (int i = 0; i < features.Length; i++)
        {
            int activation = 0;
            for (int j = 0; j < datapoints.Count; j++)
            {
                if (datapoints[j].vector[i])
                {
                    activation++;
                }
            }
            activations.Add(activation);
        }

        // copy features into a list sorted by activations
        List<string> sortedFeatures = new List<string>();
        for (int i = 0; i < features.Length; i++)
        {
            int maxActivation = activations.Max();
            int maxIndex = activations.IndexOf(maxActivation);
            sortedFeatures.Add(features[maxIndex]);
            activations[maxIndex] = -1;
        }

        // create a new set of datapoints with sorted features
        List<Datapoint> sortedDatapoints = new List<Datapoint>();
        for (int i = 0; i < datapoints.Count; i++)
        {
            bool[] sortedVector = new bool[features.Length];
            for (int j = 0; j < features.Length; j++)
            {
                sortedVector[j] = datapoints[i].vector[Array.IndexOf(features, sortedFeatures[j])];
            }
            sortedDatapoints.Add(new Datapoint(datapoints[i].index, sortedVector));
        }

        // overwrite the original features and datapoints
        features = sortedFeatures.ToArray();
        datapoints = sortedDatapoints;
    }
}

class Datapoint
{
    public int index;
    public bool[] vector;
    public HashSet<int> featureSet;

    public Datapoint(int index, bool[] vector)
    {
        this.index = index;
        this.vector = vector;
        this.featureSet = new HashSet<int>();
        for (int i = 0; i < vector.Length; i++)
        {
            if (vector[i])
            {
                featureSet.Add(i);
            }
        }
    }

    public string Smear()
    {
        StringBuilder sb = new StringBuilder();
        sb.Append('[');
        for (int i = 0; i < vector.Length; i++)
        {
            sb.Append(vector[i] ? "▓" : "░");
        }
        sb.Append(']');
        return sb.ToString();
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

    public void AddDatapoint(Datapoint datapoint, DifferenceTable differanceTable, int i = -1)
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
        if (i == -1)
        {
            datapoints.Add(datapoint);
        }
        else
        {
            datapoints.Insert(i, datapoint);
        }

        // if we have more than one datapoint update averages
        if (datapoints.Count > 1)
        {
            // calculate the average
            differenceAverage = ((float)differenceSum) / ((float)differenceCount);

            // calculate the weighted average
            weightedDifferenceAverage = differenceAverage * ((float)datapoints.Count);
        }
    }

    public Datapoint RemoveDatapoint(int i, DifferenceTable differenceTable)
    {
        // get the i'th datapoint
        Datapoint removedDatapoint = datapoints[i];

        // remove it
        datapoints.RemoveAt(i);

        // if there are no datapoints left, zero metrics and return
        if (datapoints.Count == 0)
        {
            differenceCount = 0;
            differenceSum = 0;
            differenceAverage = 0f;
            weightedDifferenceAverage = 0f;
        }

        // iterate through the remaining datapoints to remove the differences
        foreach (Datapoint remainingDatapoint in datapoints)
        {
            // lookup the difference
            int difference = differenceTable[removedDatapoint.index][remainingDatapoint.index];

            // remove it from the metrics
            differenceCount--;
            differenceSum -= difference;
        }

        // update the averages
        differenceAverage = ((float)differenceSum) / ((float)differenceCount);
        weightedDifferenceAverage = differenceAverage * ((float)datapoints.Count);

        // return the datapoint
        return removedDatapoint;
    }
}

class Program
{
    static long BinomialCoefficient(int n, int k)
    {
        // calculate the binomial coefficient
        long result = 1;
        for (int i = 1; i <= k; i++)
        {
            result = result * (n - i + 1) / i;
        }
        return result;
    }

    static void ClusterSearch(int clusterCount, Dataset dataset, DifferenceTable differenceTable, out List<Cluster>? bestClusters, out float bestTotalWeightedDifferenceAverage, out int[] bestClusterSeedIterator)
    {
        // create a cluster seed iterator
        int[] clusterSeedIterator = new int[clusterCount * 2];

        // calculate the number of possible seeds dataset.datapoints choose clusterCount * 2
        long seedCount = BinomialCoefficient(dataset.datapoints.Count, clusterCount * 2);

        // record the best clusters so far
        bestClusters = null;
        bestTotalWeightedDifferenceAverage = float.MaxValue;
        bestClusterSeedIterator = new int[clusterCount * 2];

        // iterate through all possible cluster seeds
        long currentSeedIndex = 0;
        for (; ; )
        {
            // if the iterator has no duplicate values we can use it as a seed
            if (clusterSeedIterator.Distinct().Count() == clusterCount * 2)
            {
                currentSeedIndex++;
                if (currentSeedIndex % 10000 == 0)
                {
                    double complete = (double)currentSeedIndex / (double)seedCount;
                    Console.WriteLine($"Progress: {complete:P}");
                }

                // build clusters
                List<Cluster> clusters = ClusterSearchBuildClusters(clusterCount, clusterSeedIterator, dataset, differenceTable, out float totalWeightedDifferenceAverage);

                // if we have no best, or this is better than the best, record it
                if (bestClusters == null || totalWeightedDifferenceAverage < bestTotalWeightedDifferenceAverage)
                {
                    bestClusters = clusters;
                    bestTotalWeightedDifferenceAverage = totalWeightedDifferenceAverage;
                    Array.Copy(clusterSeedIterator, bestClusterSeedIterator, clusterCount);
                }
            }

            // iterate
            for (int i = 0; i < clusterSeedIterator.Length; i++)
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

    static List<Cluster> ClusterSearchBuildClusters(int clusterCount, int[] clusterSeedIterator, Dataset dataset, DifferenceTable differenceTable, out float totalWeightedDifferenceAverage)
    {
        // create the clusters using the seeds
        List<Cluster> clusters = new List<Cluster>();
        for (int i = 0; i < clusterCount; i++)
        {
            Cluster cluster = new Cluster();
            cluster.AddDatapoint(dataset.datapoints[clusterSeedIterator[2 * i]], differenceTable);
            cluster.AddDatapoint(dataset.datapoints[clusterSeedIterator[2 * i + 1]], differenceTable);
            clusters.Add(cluster);
        }

        // iterate through all datapoints and greedily add them to the best cluster
        int totalDatapoints = clusterSeedIterator.Length;
        totalWeightedDifferenceAverage = float.NaN;
        for (int i = 0; i < dataset.datapoints.Count; i++)
        {
            // skip seed datapoints
            if (clusterSeedIterator.Contains(i))
            {
                continue;
            }

            // increment total datapoints in clusters
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

    static void MigrateOptimizeClusters(List<Cluster> clusters, DifferenceTable differenceTable)
    {
        // count datapoints
        int totalDatapoints = 0;
        for (int i = 0; i < clusters.Count; i++)
        {
            totalDatapoints += clusters[i].datapoints.Count;
        }

        // calculate the total weighted distance average)
        float totalWeightedDifferenceAverage = 0f;
        for (int i = 0; i < clusters.Count; i++)
        {
            totalWeightedDifferenceAverage += clusters[i].weightedDifferenceAverage;
        }
        totalWeightedDifferenceAverage /= ((float)totalDatapoints);

        Console.WriteLine("Starting: " + totalWeightedDifferenceAverage);

        // loop until improvements stop
        bool improved = true;
        while (improved)
        {
            improved = false;

            // iterate for the host cluster
            for (int hostClusterIndex = 0; hostClusterIndex < clusters.Count; hostClusterIndex++)
            {
                // get the host cluster
                Cluster hostCluster = clusters[hostClusterIndex];

                // if the host cluster only has 2 datapoints, it cant give one away
                if (hostCluster.datapoints.Count == 2)
                {
                    continue;
                }

                // iterate for the destination clusters
                for (int destinationClusterIndex = 0; destinationClusterIndex < clusters.Count; destinationClusterIndex++)
                {
                    // cant migrate a datapoint to self
                    if (hostClusterIndex == destinationClusterIndex)
                    {
                        continue;
                    }

                    // get the destination cluster
                    Cluster destinationCluster = clusters[destinationClusterIndex];

                    // iterate for the host datapoint
                    for (int hostDatapointIndex = 0; hostDatapointIndex < hostCluster.datapoints.Count; hostDatapointIndex++)
                    {
                        // pull the host datapoint
                        Datapoint hostDatapoint = hostCluster.RemoveDatapoint(hostDatapointIndex, differenceTable);

                        // add the host datapoint to the destination cluster
                        destinationCluster.AddDatapoint(hostDatapoint, differenceTable);

                        // calculate the new total weighted difference average
                        float newTotalWeightedDifferenceAverage = 0f;
                        for (int i = 0; i < clusters.Count; i++)
                        {
                            newTotalWeightedDifferenceAverage += clusters[i].weightedDifferenceAverage;
                        }
                        newTotalWeightedDifferenceAverage /= ((float)totalDatapoints);

                        // if this migration is an improvement, keep it and continue
                        if (newTotalWeightedDifferenceAverage < totalWeightedDifferenceAverage)
                        {
                            totalWeightedDifferenceAverage = newTotalWeightedDifferenceAverage;
                            improved = true;
                            hostDatapointIndex--; // jog back to cover removed datapoint
                            continue;
                        }

                        // reaching here implies the migration was not an improvement, so we revert it

                        // pull the datapoint from the end of the destination
                        Datapoint revertedDatapoint = destinationCluster.RemoveDatapoint(destinationCluster.datapoints.Count - 1, differenceTable);

                        // add it back in place to the host
                        hostCluster.AddDatapoint(revertedDatapoint, differenceTable, hostDatapointIndex);

                        // continue to next
                        continue;
                    }
                }
            }
        }

        Console.WriteLine("Ending: " + totalWeightedDifferenceAverage);
    }


    static void Main(string[] args)
    {
        const string infile = "FT-163.csv";
        //const int clusterCount = 2;


        // create trajectory tgf streams
        using (StreamWriter forward = new StreamWriter("trajectory-forward.tgf"))
        using (StreamWriter backward = new StreamWriter("trajectory-backward.tgf"))
        using (StreamWriter both = new StreamWriter("trajectory-both.tgf"))
        {
            Dataset dataset = new Dataset(infile);
            DifferenceTable differenceTable = new DifferenceTable(dataset);

            // write nodes
            for (int i = 0; i < dataset.datapoints.Count; i++)
            {
                forward.WriteLine($"{i} s: {dataset.datapoints[i].Smear()} i: {dataset.datapoints[i].index} a: {dataset.datapoints[i].vector.Count(x => x)} ");
                backward.WriteLine($"{i} s: {dataset.datapoints[i].Smear()} i: {dataset.datapoints[i].index} a: {dataset.datapoints[i].vector.Count(x => x)} ");
                both.WriteLine($"{i} s: {dataset.datapoints[i].Smear()} i: {dataset.datapoints[i].index} a: {dataset.datapoints[i].vector.Count(x => x)} ");
            }

            // write divider
            forward.WriteLine("#");
            backward.WriteLine("#");
            both.WriteLine("#");

            // write edges
            for (int i = 0; i < dataset.datapoints.Count; i++)
            {
                Datapoint datapoint = dataset.datapoints[i];

                // get activation count of this datapoint
                int activationCount = datapoint.vector.Count(x => x);

                // find datapoints with more activations
                List<Datapoint> datapointsWithMoreActivations = dataset.datapoints.Where(x => x.vector.Count(y => y) > activationCount).ToList();
                if (datapointsWithMoreActivations.Count > 0)
                {
                    // find the closest
                    Datapoint closest = datapointsWithMoreActivations.OrderBy(x => differenceTable[datapoint.index][x.index]).First();

                    // make an edge from this to next most activated
                    forward.WriteLine($"{datapoint.index} {closest.index}");
                    both.WriteLine($"{datapoint.index} {closest.index}");
                }

                // find datapoints with less activations
                List<Datapoint> datapointsWithLessActivations = dataset.datapoints.Where(x => x.vector.Count(y => y) < activationCount).ToList();
                if (datapointsWithLessActivations.Count > 0)
                {
                    // find the closest
                    Datapoint closest = datapointsWithLessActivations.OrderBy(x => differenceTable[datapoint.index][x.index]).First();

                    // make an edge from the lesser activated to this
                    backward.WriteLine($"{closest.index} {datapoint.index}");
                    both.WriteLine($"{closest.index} {datapoint.index}");
                }
            }
        }

        return;

        // migrate optimize
        // create a writer
        for (int clusterCount = 2; clusterCount <= 50; clusterCount++)
        {
            Dataset dataset = new Dataset(infile);
            DifferenceTable differenceTable = new DifferenceTable(dataset);

            // shuffle dataset
            Random random = new Random();
            dataset.datapoints = dataset.datapoints.OrderBy(x => random.Next()).ToList();

            Console.WriteLine("Cluster Count: " + clusterCount);

            // round robin build clusters
            List<Cluster> clusters = new List<Cluster>();
            for (int i = 0; i < clusterCount; i++)
            {
                clusters.Add(new Cluster());
            }
            for (int i = 0; i < dataset.datapoints.Count; i++)
            {
                clusters[i % clusterCount].AddDatapoint(dataset.datapoints[i], differenceTable);
            }

            // migrate optimize
            MigrateOptimizeClusters(clusters, differenceTable);

            // compute total weighted difference average
            float totalWeightedDifferenceAverage = 0f;
            for (int i = 0; i < clusters.Count; i++)
            {
                totalWeightedDifferenceAverage += clusters[i].weightedDifferenceAverage;
            }
            totalWeightedDifferenceAverage /= ((float)dataset.datapoints.Count);

            // sort each cluster
            for (int i = 0; i < clusters.Count; i++)
            {
                // get this cluster
                Cluster cluster = clusters[i];

                // sort datapoints by most to least activation
                cluster.datapoints = cluster.datapoints.OrderBy(x => x.vector.Count(y => y)).Reverse().ToList();
            }

            // sort all clusters by most to least highest activation
            clusters = clusters.OrderBy(cluster => cluster.datapoints[0].vector.Count(v => v)).Reverse().ToList();

            // write smear via write stream
            using (StreamWriter writer = new StreamWriter($"cluster{clusterCount}-verti-smear.txt"))
            {
                writer.WriteLine($"Cluster Count: {clusterCount}");
                writer.WriteLine($"Total Weighted Difference Average: {totalWeightedDifferenceAverage}");
                for (int i = 0; i < clusters.Count; i++)
                {
                    writer.WriteLine($"Cluster: {i}, Count: {clusters[i].datapoints.Count}, Difference Average: {clusters[i].differenceAverage}");
                    foreach (Datapoint datapoint in clusters[i].datapoints)
                    {
                        writer.WriteLine(datapoint.Smear());
                    }
                }
            }

            // write a horizontal smear
            int longestHeader = dataset.features.Max(x => x.Length);
            using (StreamWriter writer = new StreamWriter($"cluster{clusterCount}-hori-smear.txt"))
            {
                for (int featureIndex = 0; featureIndex < dataset.features.Length; featureIndex++)
                {
                    // write header with padding
                    writer.Write(dataset.features[featureIndex].PadLeft(longestHeader) + " | ");
                    // iterate through cluster
                    for (int clusterIndex = 0; clusterIndex < clusters.Count; clusterIndex++)
                    {
                        Cluster cluster = clusters[clusterIndex];
                        // .. datapoints
                        for (int datapointIndex = 0; datapointIndex < cluster.datapoints.Count; datapointIndex++)
                        {
                            writer.Write(cluster.datapoints[datapointIndex].vector[featureIndex] ? "█" : " ");
                        }

                        if (clusterIndex != clusters.Count - 1)
                        {
                            // write cluster separator
                            writer.Write(" | ");
                        }
                        else
                        {
                            // next line
                            writer.WriteLine();
                        }
                    }
                }
            }
        }

        /*
        // cluster search
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
        */
    }
}