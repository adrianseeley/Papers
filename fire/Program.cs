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
    public List<int> activationCounts;
    public List<float> activationProbabilities;

    public Cluster()
    {
        datapoints = new List<Datapoint>();
        activationCounts = new List<int>();
        activationProbabilities = new List<float>();
    }

    public void CalculateMetrics()
    {
        // count the activations of each feature
        activationCounts.Clear();
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
            activationCounts.Add(activation);
        }

        // calculate the probabilities
        activationProbabilities.Clear();
        for (int i = 0; i < activationCounts.Count; i++)
        {
            activationProbabilities.Add(((float)activationCounts[i]) / ((float)datapoints.Count));
        }
    }

    public List<float> CalculateProbabilityDeltas(Cluster other)
    {
        List<float> probabilityDeltas = new List<float>();
        for (int i = 0; i < activationProbabilities.Count; i++)
        {
            probabilityDeltas.Add(other.activationProbabilities[i] - activationProbabilities[i]);
        }
        return probabilityDeltas;
    }

    public List<float> CalculateOddsRatios(Cluster other)
    {
        List<float> oddsRatios = new List<float>();
        for (int i = 0; i < activationProbabilities.Count; i++)
        {
            oddsRatios.Add(other.activationProbabilities[i] / activationProbabilities[i]);
        }
        return oddsRatios;
    }

    class Program
    {
        static void Main(string[] args)
        {
            const string infile = "FT-163.csv";
            Dataset dataset = new Dataset(infile);
            int longestFeatureNameLength = dataset.features.Max(f => f.Length);

            Cluster all = new Cluster();
            all.datapoints.AddRange(dataset.datapoints);
            all.CalculateMetrics();

            StreamWriter tablesWriter = new StreamWriter("tables.txt");

            tablesWriter.WriteLine("All Probabilities");
            tablesWriter.Write("P(ALL)".PadLeft(7));
            tablesWriter.Write("N(ALL)".PadLeft(9));
            tablesWriter.Write("  ");
            tablesWriter.Write("Feature".PadRight(longestFeatureNameLength));
            tablesWriter.WriteLine();
            for (int i = 0; i < dataset.features.Length; i++)
            {
                float activationProbability = all.activationProbabilities[i];
                int activationCount = all.activationCounts[i];
                tablesWriter.Write(activationProbability.ToString("0.000").PadLeft(7));
                tablesWriter.Write(activationCount.ToString().PadLeft(5));
                tablesWriter.Write("/");
                tablesWriter.Write(all.datapoints.Count.ToString().PadLeft(3));
                tablesWriter.Write("  ");
                tablesWriter.Write(dataset.features[i].PadRight(longestFeatureNameLength));
                tablesWriter.WriteLine();
            }
            tablesWriter.WriteLine();

            for (int featureIndex = 0; featureIndex < dataset.features.Length; featureIndex++)
            {
                Cluster noGroup = new Cluster();
                Cluster yesGroup = new Cluster();
                foreach (Datapoint datapoint in dataset.datapoints)
                {
                    if (datapoint.vector[featureIndex])
                    {
                        yesGroup.datapoints.Add(datapoint);
                    }
                    else
                    {
                        noGroup.datapoints.Add(datapoint);
                    }
                }
                noGroup.CalculateMetrics();
                yesGroup.CalculateMetrics();
                List<float> probabilityDeltas = noGroup.CalculateProbabilityDeltas(yesGroup);
                List<float> oddsRatios = noGroup.CalculateOddsRatios(yesGroup);

                tablesWriter.WriteLine($"Feature: {dataset.features[featureIndex]}");
                tablesWriter.Write("P(NO)".PadLeft(7));
                tablesWriter.Write("P(YES)".PadLeft(7));
                tablesWriter.Write("PΔ".PadLeft(7));
                tablesWriter.Write("OR".PadLeft(7));
                tablesWriter.Write("N(NO)".PadLeft(8));
                tablesWriter.Write("N(YES)".PadLeft(9));
                tablesWriter.Write("  ");
                tablesWriter.Write("Feature".PadRight(longestFeatureNameLength));
                tablesWriter.WriteLine();
                for (int otherFeatureIndex = 0; otherFeatureIndex < dataset.features.Length; otherFeatureIndex++)
                {
                    float noGroupActivationProbability = noGroup.activationProbabilities[otherFeatureIndex];
                    float yesGroupActivationProbability = yesGroup.activationProbabilities[otherFeatureIndex];
                    float probabilityDelta = probabilityDeltas[otherFeatureIndex];
                    float oddsRatio = oddsRatios[otherFeatureIndex];
                    int noGroupActivationCount = noGroup.activationCounts[otherFeatureIndex];
                    int noGroupCount = noGroup.datapoints.Count;
                    int yesGroupActivationCount = yesGroup.activationCounts[otherFeatureIndex];
                    int yesGroupCount = yesGroup.datapoints.Count;
                    tablesWriter.Write(noGroupActivationProbability.ToString("0.000").PadLeft(7));
                    tablesWriter.Write(yesGroupActivationProbability.ToString("0.000").PadLeft(7));
                    tablesWriter.Write(probabilityDelta.ToString("0.000").PadLeft(7));
                    tablesWriter.Write(oddsRatio.ToString("0.000").PadLeft(7));
                    tablesWriter.Write(noGroupActivationCount.ToString().PadLeft(4));
                    tablesWriter.Write("/");
                    tablesWriter.Write(noGroupCount.ToString().PadLeft(3));
                    tablesWriter.Write(yesGroupActivationCount.ToString().PadLeft(5));
                    tablesWriter.Write("/");
                    tablesWriter.Write(yesGroupCount.ToString().PadLeft(3));
                    tablesWriter.Write("  ");
                    tablesWriter.Write(dataset.features[otherFeatureIndex].PadRight(longestFeatureNameLength));
                    tablesWriter.WriteLine();
                }
                tablesWriter.WriteLine();
            }

            tablesWriter.Close();
            tablesWriter.Dispose();
        }
    }
}