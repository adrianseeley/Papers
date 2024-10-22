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
}

class Cluster
{
    public List<Datapoint> datapoints;
    public List<float> activationProbabilities;

    public Cluster()
    {
        datapoints = new List<Datapoint>();
        activationProbabilities = new List<float>();
    }

    public void CalculateActivationProbabilities()
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
        activationProbabilities.Clear();
        for (int i = 0; i < activations.Count; i++)
        {
            activationProbabilities.Add(((float)activations[i]) / ((float)datapoints.Count));
        }
    }

    public static float Fitness(List<Cluster> clusters)
    {
        // the goal is to maximize the euclidean distance between activation probabilities across clusters
        float fitness = 0.0f;
        for (int i = 0; i < clusters.Count; i++)
        {
            Cluster clusterA = clusters[i];
            for (int j = i + 1; j < clusters.Count; j++)
            {
                Cluster clusterB = clusters[j];
                for (int k = 0; k < clusterA.activationProbabilities.Count; k++)
                {
                    fitness += (float)Math.Pow(clusterA.activationProbabilities[k] - clusterB.activationProbabilities[k], 2);
                }
            }
        }
        fitness = (float)Math.Sqrt(fitness);
        return fitness;
    }

    public static List<Cluster> ByMigration(Dataset dataset, int clusterCount, int minClusterSize)
    {
        // shuffle dataset
        Random random = new Random();
        List<Datapoint> shuffledDatapoints = dataset.datapoints.OrderBy(x => random.Next()).ToList();

        // create clusters
        List<Cluster> clusters = new List<Cluster>();
        for (int i = 0; i < clusterCount; i++)
        {
            clusters.Add(new Cluster());
        }

        // add datapoints round robin style
        for (int i = 0; i < shuffledDatapoints.Count; i++)
        {
            clusters[i % clusterCount].datapoints.Add(shuffledDatapoints[i]);
        }

        // update probabilities
        clusters.ForEach(cluster => cluster.CalculateActivationProbabilities());

        // find starting fitness
        float bestFitness = Fitness(clusters);

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

                // if the host cluster doesnt have enough points to give away skip
                if (hostCluster.datapoints.Count <= minClusterSize)
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
                        // if the host cluster doesnt have enough points to give away skip
                        if (hostCluster.datapoints.Count <= minClusterSize)
                        {
                            break;
                        }

                        // pull the host datapoint, update probabilities
                        Datapoint hostDatapoint = hostCluster.datapoints[hostDatapointIndex];
                        hostCluster.datapoints.RemoveAt(hostDatapointIndex);
                        hostCluster.CalculateActivationProbabilities();

                        // add the host datapoint to the destination cluster, update probabilities
                        destinationCluster.datapoints.Add(hostDatapoint);
                        destinationCluster.CalculateActivationProbabilities();

                        // calculate the new fitness
                        float newFitness = Fitness(clusters);

                        // if this migration is an improvement, keep it and continue
                        if (newFitness > bestFitness)
                        {
                            bestFitness = newFitness;
                            improved = true;
                            hostDatapointIndex--; // jog back to cover removed datapoint
                            continue;
                        }

                        // reaching here implies the migration was not an improvement, so we revert it

                        // pull the datapoint from the end of the destination
                        destinationCluster.datapoints.RemoveAt(destinationCluster.datapoints.Count - 1);
                        destinationCluster.CalculateActivationProbabilities();

                        // add it back in place to the host
                        hostCluster.datapoints.Insert(hostDatapointIndex, hostDatapoint);
                        hostCluster.CalculateActivationProbabilities();

                        // continue to next
                        continue;
                    }
                }
            }
        }

        return clusters;
    }

}

class Program
{
    static void Main(string[] args)
    {
        const string infile = "FT-163.csv";
        Dataset dataset = new Dataset(infile);
        const int minClusterSize = 8;

        using (StreamWriter writer = new StreamWriter($"cluster-smear.txt"))
        {
            // find longest header size
            int longestHeader = dataset.features.Max(x => x.Length);

            // iterate through cluster counts
            for (int clusterCount = 2; clusterCount <= 10; clusterCount++)
            {
                // create clusters
                List<Cluster> clusters = Cluster.ByMigration(dataset, clusterCount, minClusterSize);

                // sort datapoints in each cluster by high to low activation
                clusters.ForEach(cluster => cluster.datapoints = cluster.datapoints.OrderBy(datapoint => datapoint.activation).Reverse().ToList());

                // sort clusters by high to low activation
                clusters = clusters.OrderBy(cluster => cluster.datapoints[0].activation).Reverse().ToList();

                // get activation probabilities for each cluster
                List<List<float>> activationProbabilities = new List<List<float>>();
                for (int i = 0; i < clusters.Count; i++)
                {
                    activationProbabilities.Add(clusters[i].activationProbabilities);
                }

                // calcualte fitness
                float fitness = Cluster.Fitness(clusters);

                // title smear
                writer.WriteLine($"Cluster Count: {clusterCount} | Fitness: {fitness}");

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