class Dataset
{
    public string[] features;
    public List<Datapoint> datapoints;

    public Dataset(string[] features, List<Datapoint> datapoints)
    {
        this.features = features;
        this.datapoints = datapoints;
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

class Program
{
    static Dataset Parse(string path)
    {
        // get all lines of csv file
        string[] lines = File.ReadAllLines(path);

        // the first line contains the headers, we skip the timestamp column
        string[] features = lines[0].Split("\",\"").Skip(1).ToArray();

        // each feature has a prefix of I have ... and the real question is in []
        for (int i = 0; i < features.Length; i++)
        {
            features[i] = features[i].Split('[')[1].Split(']')[0];
        }

        // create a list of parsed datapoints
        List<Datapoint> datapoints = new List<Datapoint>();

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

        // return created dataset
        return new Dataset(features, datapoints);
    }

    static void Main(string[] args)
    {
        const string infile = "FT-163.csv";
        Dataset dataset = Parse(infile);


    }
}