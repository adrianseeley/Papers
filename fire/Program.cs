class Dataset
{
    public static Dictionary<string, string> shortFeatures = new Dictionary<string, string>()
    {
        {"Cooked with fire.", "COOKED"},
        {"Played with a lighter.", "LIGHTER"},
        {"Used a charcoal BBQ.", "CHARCOAL"},
        {"Worked in a kitchen.", "KITCHEN"},
        {"Played with matches.", "MATCHES"},
        {"Started a camp fire.", "CAMPFIRE"},
        {"Received fire safety training.", "SAFETY"},
        {"Used a gas BBQ.", "GASBBQ"},
        {"Lit fireworks.", "FIREWORKS"},
        {"Started a small fire.", "SMALL"},
        {"Burned leaves and/or brush.", "BRUSH"},
        {"Started a fire with friends.", "FRIENDS"},
        {"Started a fire intentionally.", "INTENTIONAL"},
        {"Seen a house, apartment, or building on fire.", "SEENHOUSE"},
        {"Used an accelerant such as gasoline or lighter fluid to start a fire.", "ACCELERANT"},
        {"Tried to start a fire with a magnifying glass.", "MAGNIFY"},
        {"Seen a fire get out of control.", "SEENOCONTROL"},
        {"Started a fire alone.", "ALONE"},
        {"Started an average size fire.", "AVERAGE"},
        {"Smoked marijuana (weed) or other drugs.", "WEED"},
        {"Hurt myself with fire.", "HURTSELF"},
        {"Smoked cigarettes.", "SMOKED"},
        {"Owned a Zippo brand lighter.", "ZIPPO"},
        {"Burned garbage.", "GARBAGE"},
        {"Smoked while under the legal age.", "SMOKEDYOUNG"},
        {"Had to evacuate because of a fire.", "EVACUATE"},
        {"Used a fire torch.", "TORCH"},
        {"Burned items on a BBQ that weren't food.", "BADBBQ"},
        {"Started a fire accidently.", "ACCIGENERAL"},
        {"Worked with Fire.", "WORKED"},
        {"Used fire as part of a ritual or practice.", "RITUAL"},
        {"Used fire to burn my own belongings.", "BURNEDMINE"},
        {"Thrown or placed dangerous items into a fire.", "THROWNBAD"},
        {"Accidently started a fire in the house.", "ACCIINHOUSE"},
        {"Had a fire get out of control.", "OUTCONTROL"},
        {"Attempted to conceal a fire.", "CONCEAGEN"},
        {"Started a large fire.", "LARGE"},
        {"Tried to hide a fire that I started.", "TRYHIDE"},
        {"Burned toys or stuffed animals/toys.", "BURNTOYS"},
        {"Been in trouble because of fire.", "TROUBLE"},
        {"Worked as a welder.", "WELDER"},
        {"Started an extremely large fire.", "XLARGE"},
        {"Used fire as a cry for help.", "CRYHELP"},
        {"Hurt someone else with fire.", "HURTOTHER"},
        {"Used fire to burn someone else's belongings.", "BURNEDTHEIRS"},
        {"Lied about a fire I started.", "LIED"},
        {"Used fire to destroy or conceal evidence of a crime.", "CONCEALCRIME"},
        {"Been in legal or criminal trouble because of a fire.", "LEGAL"},
        {"Used fire to get attention.", "ATTENTION"},
        {"Used fire to manipulate or control someone.", "MANIPULATE"},
        {"Set a fire to get revenge.", "REVENGE"},
        {"Used fire as a weapon.", "WEAPON"},
    };

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
    public float[,] jaccardIndexes;
    public float[,] jointProbabilities;
    public float[,] conditionalProbabilities;

    public Cluster(int featureCount)
    {
        datapoints = new List<Datapoint>();
        activationCounts = new List<int>();
        activationProbabilities = new List<float>();
        jaccardIndexes = new float[featureCount, featureCount];
        jointProbabilities = new float[featureCount, featureCount];
        conditionalProbabilities = new float[featureCount, featureCount];
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

        // calculate the Jaccard indexes
        for (int i = 0; i < datapoints[0].vector.Length; i++)
        {
            for (int j = 0; j < datapoints[0].vector.Length; j++)
            {
                int both = 0;
                int either = 0;
                for (int k = 0; k < datapoints.Count; k++)
                {
                    if (datapoints[k].vector[i] && datapoints[k].vector[j])
                    {
                        both++;
                    }
                    if (datapoints[k].vector[i] || datapoints[k].vector[j])
                    {
                        either++;
                    }
                }
                jaccardIndexes[i, j] = ((float)both) / ((float)either);
            }
        }

        // calculate the joint probabilities
        for (int i = 0; i < datapoints[0].vector.Length; i++)
        {
            for (int j = 0; j < datapoints[0].vector.Length; j++)
            {
                int both = 0;
                for (int k = 0; k < datapoints.Count; k++)
                {
                    if (datapoints[k].vector[i] && datapoints[k].vector[j])
                    {
                        both++;
                    }
                }
                jointProbabilities[i, j] = ((float)both) / ((float)datapoints.Count);
            }
        }

        // calculate the conditional probabilities
        for (int i = 0; i < datapoints[0].vector.Length; i++)
        {
            for (int j = 0; j < datapoints[0].vector.Length; j++)
            {
                conditionalProbabilities[i, j] = jointProbabilities[i, j] / activationProbabilities[j];
            }
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
}
class Program
{
    static void Main(string[] args)
    {
        const string infile = "FT-177.csv";
        Dataset dataset = new Dataset(infile);
        int longestFeatureNameLength = dataset.features.Max(f => f.Length);
        int longestShortFeatureNameLength = dataset.features.Max(f => Dataset.shortFeatures[f].Length);

        Cluster all = new Cluster(dataset.features.Length);
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
            Cluster noGroup = new Cluster(dataset.features.Length);
            Cluster yesGroup = new Cluster(dataset.features.Length);
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
            tablesWriter.Write("JACC".PadLeft(7));
            tablesWriter.Write("JOINT".PadLeft(7));
            tablesWriter.Write("COND.".PadLeft(7));
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
                tablesWriter.Write(all.jaccardIndexes[featureIndex, otherFeatureIndex].ToString("0.000").PadLeft(7));
                tablesWriter.Write(all.jointProbabilities[featureIndex, otherFeatureIndex].ToString("0.000").PadLeft(7));
                tablesWriter.Write(all.conditionalProbabilities[featureIndex, otherFeatureIndex].ToString("0.000").PadLeft(7));
                tablesWriter.Write("  ");
                tablesWriter.Write(dataset.features[otherFeatureIndex].PadRight(longestFeatureNameLength));
                tablesWriter.WriteLine();
            }

            tablesWriter.WriteLine("High OR >5.0");
            for (int otherFeatureIndex = 0; otherFeatureIndex < dataset.features.Length; otherFeatureIndex++)
            {
                if (otherFeatureIndex == featureIndex)
                {
                    continue;
                }
                float noGroupActivationProbability = noGroup.activationProbabilities[otherFeatureIndex];
                float yesGroupActivationProbability = yesGroup.activationProbabilities[otherFeatureIndex];
                float probabilityDelta = probabilityDeltas[otherFeatureIndex];
                float oddsRatio = oddsRatios[otherFeatureIndex];
                int noGroupActivationCount = noGroup.activationCounts[otherFeatureIndex];
                int noGroupCount = noGroup.datapoints.Count;
                int yesGroupActivationCount = yesGroup.activationCounts[otherFeatureIndex];
                int yesGroupCount = yesGroup.datapoints.Count;
                if (oddsRatio < 5.0f)
                {
                    continue;
                }
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
                tablesWriter.Write(all.jaccardIndexes[featureIndex, otherFeatureIndex].ToString("0.000").PadLeft(7));
                tablesWriter.Write(all.jointProbabilities[featureIndex, otherFeatureIndex].ToString("0.000").PadLeft(7));
                tablesWriter.Write(all.conditionalProbabilities[featureIndex, otherFeatureIndex].ToString("0.000").PadLeft(7));
                tablesWriter.Write("  ");
                tablesWriter.Write(dataset.features[otherFeatureIndex].PadRight(longestFeatureNameLength));
                tablesWriter.WriteLine();
            }
            tablesWriter.WriteLine("Low OR <= 0.5");
            for (int otherFeatureIndex = 0; otherFeatureIndex < dataset.features.Length; otherFeatureIndex++)
            {
                if (otherFeatureIndex == featureIndex)
                {
                    continue;
                }
                float noGroupActivationProbability = noGroup.activationProbabilities[otherFeatureIndex];
                float yesGroupActivationProbability = yesGroup.activationProbabilities[otherFeatureIndex];
                float probabilityDelta = probabilityDeltas[otherFeatureIndex];
                float oddsRatio = oddsRatios[otherFeatureIndex];
                int noGroupActivationCount = noGroup.activationCounts[otherFeatureIndex];
                int noGroupCount = noGroup.datapoints.Count;
                int yesGroupActivationCount = yesGroup.activationCounts[otherFeatureIndex];
                int yesGroupCount = yesGroup.datapoints.Count;
                if (oddsRatio > 0.5f)
                {
                    continue;
                }
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
                tablesWriter.Write(all.jaccardIndexes[featureIndex, otherFeatureIndex].ToString("0.000").PadLeft(7));
                tablesWriter.Write(all.jointProbabilities[featureIndex, otherFeatureIndex].ToString("0.000").PadLeft(7));
                tablesWriter.Write(all.conditionalProbabilities[featureIndex, otherFeatureIndex].ToString("0.000").PadLeft(7));
                tablesWriter.Write("  ");
                tablesWriter.Write(dataset.features[otherFeatureIndex].PadRight(longestFeatureNameLength));
                tablesWriter.WriteLine();
            }

            tablesWriter.WriteLine();
        }

        // write contingency tables for all features vs each other
        for (int a = 0; a < dataset.features.Length; a++)
        {
            for (int b = 0; b < dataset.features.Length; b++)
            {
                if (a == b)
                {
                    continue;
                }
                int A_B = 0;
                int NOTA_B = 0;
                int A_NOTB = 0;
                int NOTA_NOTB = 0;
                for (int i = 0; i < dataset.datapoints.Count; i++)
                {
                    if (dataset.datapoints[i].vector[a] && dataset.datapoints[i].vector[b])
                    {
                        A_B++;
                    }
                    else if (!dataset.datapoints[i].vector[a] && dataset.datapoints[i].vector[b])
                    {
                        NOTA_B++;
                    }
                    else if (dataset.datapoints[i].vector[a] && !dataset.datapoints[i].vector[b])
                    {
                        A_NOTB++;
                    }
                    else
                    {
                        NOTA_NOTB++;
                    }
                }
                tablesWriter.WriteLine($"Contingency Table: A: {dataset.features[a]} vs B: {dataset.features[b]}");
                tablesWriter.WriteLine($"A\\B\tB\t¬B");
                tablesWriter.WriteLine($"A\t{A_B}\t{A_NOTB}");
                tablesWriter.WriteLine($"¬A\t{NOTA_B}\t{NOTA_NOTB}");
                tablesWriter.WriteLine();
            }
        }

        tablesWriter.Close();
        tablesWriter.Dispose();

        HashSet<string> dangerous = new HashSet<string>()
        {
            "Hurt myself with fire.",
            "Started a fire accidently.",
            "Thrown or placed dangerous items into a fire.",
            "Accidently started a fire in the house.",
            "Had a fire get out of control.",
            "Attempted to conceal a fire.",
            "Tried to hide a fire that I started.",
            "Been in trouble because of fire.",
            "Started an extremely large fire.",
            "Used fire as a cry for help.",
            "Hurt someone else with fire.",
            "Lied about a fire I started.",
            "Used fire to destroy or conceal evidence of a crime.",
            "Been in legal or criminal trouble because of a fire.",
            "Used fire to get attention.",
            "Used fire to manipulate or control someone.",
            "Set a fire to get revenge.",
            "Used fire as a weapon.",
        };

        StreamWriter scatterWriter = new StreamWriter("scatter.csv");
        scatterWriter.WriteLine("SafeCount,DangereousCount");
        for (int i = 0; i < dataset.datapoints.Count; i++)
        {
            int safeCount = 0;
            int dangerousCount = 0;
            foreach (int featureIndex in dataset.datapoints[i].featureSet)
            {
                if (dangerous.Contains(dataset.features[featureIndex]))
                {
                    dangerousCount++;
                }
                else
                {
                    safeCount++;
                }
            }
            scatterWriter.WriteLine($"{safeCount},{dangerousCount}");
        }
        scatterWriter.Close();
        scatterWriter.Dispose();

    }
}