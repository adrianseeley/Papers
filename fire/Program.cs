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
    public int activation;

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
        this.activation = featureSet.Count;
    }

    public static List<float> GetActivationProbabilities(List<Datapoint> datapoints)
    {
        // count the activations of each feature
        List<int> activations = new List<int>();
        for (int i = 0; i < datapoints[0].vector.Length; i++)
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

        // calculate the probabilities
        List<float> probabilities = new List<float>();
        for (int i = 0; i < activations.Count; i++)
        {
            probabilities.Add(((float)activations[i]) / ((float)datapoints.Count));
        }

        return probabilities;
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

    public static float ComputeTotalWeightedDifferenceAverage(List<Cluster> clusters, int totalDatapoints)
    {
        float totalWeightedDifferenceAverage = 0f;
        for (int i = 0; i < clusters.Count; i++)
        {
            totalWeightedDifferenceAverage += clusters[i].weightedDifferenceAverage;
        }
        totalWeightedDifferenceAverage /= ((float)totalDatapoints);
        return totalWeightedDifferenceAverage;
    }

    public static List<Cluster> ByMigration(Dataset dataset, DifferenceTable differenceTable, int clusterCount)
    {
        // shuffle dataset
        Random random = new Random();
        List<Datapoint> shuffledDatapoints = dataset.datapoints.OrderBy(x => random.Next()).ToList();

        // round robin build clusters
        List<Cluster> clusters = new List<Cluster>();
        for (int i = 0; i < clusterCount; i++)
        {
            clusters.Add(new Cluster());
        }
        for (int i = 0; i < shuffledDatapoints.Count; i++)
        {
            clusters[i % clusterCount].AddDatapoint(shuffledDatapoints[i], differenceTable);
        }

        // calculate the total weighted distance average)
        float totalWeightedDifferenceAverage = ComputeTotalWeightedDifferenceAverage(clusters, shuffledDatapoints.Count);

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
                        float newTotalWeightedDifferenceAverage = ComputeTotalWeightedDifferenceAverage(clusters, shuffledDatapoints.Count);

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

        return clusters;
    }

    public Cluster()
    {
        datapoints = new List<Datapoint>();
        differenceCount = 0;
        differenceSum = 0;
        differenceAverage = 0f;
        weightedDifferenceAverage = 0f;
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
    static void Main(string[] args)
    {
        const string infile = "FT-163.csv";
        Dataset dataset = new Dataset(infile);
        DifferenceTable differenceTable = new DifferenceTable(dataset);


        using (StreamWriter writer = new StreamWriter($"cluster-smear.txt"))
        {
            // find longest header size
            int longestHeader = dataset.features.Max(x => x.Length);

            // iterate through cluster counts
            for (int clusterCount = 2; clusterCount <= 50; clusterCount++)
            {
                // create clusters
                List<Cluster> clusters = Cluster.ByMigration(dataset, differenceTable, clusterCount);

                // compute total weighted difference average
                float totalWeightedDifferenceAverage = Cluster.ComputeTotalWeightedDifferenceAverage(clusters, dataset.datapoints.Count);

                // sort datapoints in each cluster by high to low activation
                clusters.ForEach(cluster => cluster.datapoints = cluster.datapoints.OrderBy(datapoint => datapoint.activation).Reverse().ToList());

                // sort clusters by high to low activation
                clusters = clusters.OrderBy(cluster => cluster.datapoints[0].activation).Reverse().ToList();

                // get activation probabilities for each cluster
                List<List<float>> activationProbabilities = new List<List<float>>();
                for (int i = 0; i < clusters.Count; i++)
                {
                    activationProbabilities.Add(Datapoint.GetActivationProbabilities(clusters[i].datapoints));
                }

                // title smear
                writer.WriteLine($"Cluster Count: {clusterCount} | Total Weighted Difference Average: {totalWeightedDifferenceAverage}");

                // write a horizontal smear of the clusters
                for (int featureIndex = 0; featureIndex < dataset.features.Length; featureIndex++)
                {
                    // write header with padding
                    writer.Write(dataset.features[featureIndex].PadLeft(longestHeader) + " | ");

                    // iterate through cluster
                    for (int clusterIndex = 0; clusterIndex < clusters.Count; clusterIndex++)
                    {
                        Cluster cluster = clusters[clusterIndex];

                        // iterate through datapoints
                        for (int datapointIndex = 0; datapointIndex < cluster.datapoints.Count; datapointIndex++)
                        {
                            writer.Write(cluster.datapoints[datapointIndex].vector[featureIndex] ? "█" : " ");
                        }

                        // separator or line ender
                        if (clusterIndex != clusters.Count - 1)
                        {
                            // write cluster separator
                            writer.Write(" | ");
                        }
                        else
                        {
                            // write end of cluster
                            writer.Write(" | ");

                            // write activation probabilities
                            for (int i = 0; i < activationProbabilities.Count; i++)
                            {
                                writer.Write(activationProbabilities[i][featureIndex].ToString("0.00") + " | ");
                            }

                            // rewrite the feature name
                            writer.Write(dataset.features[featureIndex]);

                            // next line
                            writer.WriteLine();
                        }
                    }
                }

                // add a padding line
                writer.WriteLine();
            }
        }
    }
}