using MNISTLib;
using NeuralNetworkPrev.Controller;
using NeuralNetworkPrev.Model;
using NNv1;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Path = System.IO.Path;

namespace NeuralNetworkPrev.Views
{
    /// <summary>
    /// Interaction logic for MNISTPage.xaml
    /// </summary>
    public partial class MNISTPage : Page
    {
        private MNISTCore data;
        private NeuralNetwork nn;
        private ExManager manager;

        private double eta;
        private double momentum;

        private bool dataIsLoaded = false;
        private int trainSize;
        private int testSize;
        private int currTrainIndex = 0;
        private int currTestIndex = 0;

        private bool nnIsCreated = false;

        private BackgroundWorker bw;
        bool isTraining = false;
        private delegate void DisplayStatus(int itteration, double error, int mismatched);

        public MNISTPage()
        {
            InitializeComponent();

            bw = new BackgroundWorker();
            bw.WorkerSupportsCancellation = true;
            bw.DoWork += bw_DoWork;
            bw.RunWorkerCompleted += bw_RunWorkerComplited;

            // add relative path
            string projectFolder = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.FullName;
            txtBoxPath.Text = Path.GetFullPath(projectFolder +  @"\..\data\");
        }

        private void bw_RunWorkerComplited(object sender, RunWorkerCompletedEventArgs e)
        {
            isTraining = false;
            btnTrain.Content = "Train";
        }

        private void bw_DoWork(object sender, DoWorkEventArgs e)
        {
            double err;
            int mismatched = trainSize;
            int itteration = 0;

            int maxMismatched = 20;
            double errorLimit = 5.0;

            do
            {
                if (bw.CancellationPending)
                {
                    e.Cancel = true;
                    return;
                }

                itteration++;

                manager.eta = eta;
                manager.momentum = momentum;
                err = manager.Train(PrepareMiniBatch());

                mismatched = GetMismatched();

                DisplayStatus displayer = new DisplayStatus(ShowNNStatus);
                if (itteration % 2 == 0)
                    lblNNStatus.Dispatcher.BeginInvoke(displayer, itteration, err, mismatched);

            } while (mismatched > maxMismatched || err > errorLimit);
        }

        private void ShowNNStatus(int itteration, double error, int mismatched)
        {
            string status = "";
            status += "itteraion: " + itteration + ", ";
            status += "error: " + error + ", ";
            status += "mismatched: " + mismatched + " / " + trainSize;
            lblTrainStatus.Content = status;
        }

        private (double[], double[])[] PrepareMiniBatch()
        {
            (double[], double[])[] miniBatch = new (double[], double[])[trainSize];

            for (int i = 0; i < trainSize; i++)
            {
                double[] x = data.TrainingImages[i].RawImage;
                double[] y = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                y[data.TrainingImages[i].Label] = 1;

                miniBatch[i] = (x, y);
            }

            return miniBatch;
        }

        private int NNPredict(double[] x)
        {
            double[] res = nn.Forward(x);

            double max = 0;
            int index = 0;
            for (int j = 0; j < 10; j++)
            {
                if (max < res[j])
                {
                    max = res[j];
                    index = j;
                }
            }

            return index;
        }

        private int GetMismatched()
        {
            int mismatched = 0;
            for (int i = 0; i < trainSize; i++)
            {
                if (NNPredict(data.TrainingImages[i].RawImage) != data.TrainingImages[i].Label) mismatched++;
            }
            return mismatched;
        }

        private void BtnLoad_Click(object sender, RoutedEventArgs e)
        {
            int train = Convert.ToInt32(txtBoxTrain.Text);
            int test = Convert.ToInt32(txtBoxTest.Text);
            string path = txtBoxPath.Text;

            bool success = false;

            (data, success) = Models.GetMNISTData(train, test, path);

            if (!success)
            {
                dataIsLoaded = false;

                MessageBox.Show("An Error Occured", "Alert", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            else
            {
                trainSize = train;
                testSize = test;

                dataIsLoaded = true;
                currTrainIndex = 0;
                currTestIndex = 0;

                // set up screen for mnist preview
                lblTrainCounter.Content = currTrainIndex + 1 + " / " + trainSize;
                lblTestCounter.Content = currTestIndex + 1 + " / " + testSize;
                PreviewImage(data.TrainingImages[currTrainIndex], txtBlockTrain);
                PreviewImage(data.TrainingImages[currTestIndex], txtBlockDisplay);
                PreviewImage(data.TestImages[currTrainIndex], txtBlockTest);
                lblTestLabel.Content = data.TestImages[currTestIndex].Label;

                if (nnIsCreated)
                {
                    PreviewTestImage();
                }
            }

        }

        private void PreviewImage(DigitImage im, TextBlock txtBlock)
        {
            byte[][] px = im.Pixels;
            string output = "";
            for (int i = 0; i < px.Length; i++)
            {
                for (int j = 0; j < px[i].Length; j++)
                {
                    output += px[i][j] / 254;
                }
                output += '\n';
            }
            txtBlock.Text = output;
        }

        private void BtnTrainingPrev_Click(object sender, RoutedEventArgs e)
        {
            if (dataIsLoaded)
            {
                if (currTrainIndex > 0 && currTrainIndex < trainSize)
                {
                    currTrainIndex--;
                    lblTrainCounter.Content = currTrainIndex + 1 + " / " + trainSize;
                    PreviewImage(data.TrainingImages[currTrainIndex], txtBlockTrain);
                    PreviewImage(data.TrainingImages[currTrainIndex], txtBlockDisplay);
                }
            }
        }

        private void BtnTrainingNext_Click(object sender, RoutedEventArgs e)
        {
            if (dataIsLoaded)
            {
                if (currTrainIndex >= 0 && currTrainIndex < trainSize - 1)
                {
                    currTrainIndex++;
                    lblTrainCounter.Content = currTrainIndex + 1 + " / " + trainSize;
                    PreviewImage(data.TrainingImages[currTrainIndex], txtBlockTrain);
                    PreviewImage(data.TrainingImages[currTrainIndex], txtBlockDisplay);
                }
            }
        }

        private void BtnTestPrev_Click(object sender, RoutedEventArgs e)
        {
            if (dataIsLoaded)
            {
                if (currTestIndex > 0 && currTestIndex < testSize)
                {
                    currTestIndex--;
                    lblTestCounter.Content = currTestIndex + 1 + " / " + testSize;
                    PreviewImage(data.TestImages[currTestIndex], txtBlockTest);
                    lblTestLabel.Content = data.TestImages[currTestIndex].Label;
                    PreviewImage(data.TestImages[currTestIndex], txtBlockDisplay);
                }
                if (nnIsCreated)
                    PreviewTestImage();
            }
        }

        private void BtnTestNext_Click(object sender, RoutedEventArgs e)
        {
            if (dataIsLoaded)
            {
                if (currTestIndex >= 0 && currTestIndex < testSize - 1)
                {
                    currTestIndex++;
                    lblTestCounter.Content = currTestIndex + 1 + " / " + testSize;
                    PreviewImage(data.TestImages[currTestIndex], txtBlockTest);
                    lblTestLabel.Content = data.TestImages[currTestIndex].Label;
                    PreviewImage(data.TestImages[currTestIndex], txtBlockDisplay);
                }
                if (nnIsCreated)
                    PreviewTestImage();
            }
        }

        private void CmboxItem1_Selected(object sender, RoutedEventArgs e)
        {
            if (pageMNIST.IsLoaded)
            {
                stPanelH1.IsEnabled = true;
                stPanelH2.IsEnabled = false;
                stPanelH3.IsEnabled = false;
            }
        }

        private void CmboxItem2_Selected(object sender, RoutedEventArgs e)
        {
            stPanelH1.IsEnabled = true;
            stPanelH2.IsEnabled = true;
            stPanelH3.IsEnabled = false;
        }

        private void CmboxItem3_Selected(object sender, RoutedEventArgs e)
        {
            stPanelH1.IsEnabled = true;
            stPanelH2.IsEnabled = true;
            stPanelH3.IsEnabled = true;
        }

        private void BtnCreateNN_Click(object sender, RoutedEventArgs e)
        {
            int hiddenLayers = cmboxHiddenLayers.SelectedIndex + 1;
            int[] sizes = new int[hiddenLayers + 2];
            IActivation[] activations = new IActivation[hiddenLayers + 2];
            TextBox[] boxes = { txtBoxH1, txtBoxH2, txtBoxH3 };
            ComboBox[] cboxes = { cboxH1Activation, cboxH2Activation, cboxH3Activation };
            Sigmoid sigmoid = new Sigmoid();
            Relu relu = new Relu();
            IActivation[] activationOptions = { sigmoid, relu };

            sizes[0] = 28 * 28;
            activations[0] = null;

            sizes[hiddenLayers + 1] = 10;
            activations[hiddenLayers + 1] = sigmoid;

            try
            {
                for (int i = 1; i <= hiddenLayers; i++)
                {
                    sizes[i] = Convert.ToInt32(boxes[i].Text);
                    activations[i] = activationOptions[cboxes[i - 1].SelectedIndex];
                }

                double factor = 0.0357142857142857;
                //factor = 1.0 / 75;

                nn = new NeuralNetwork(sizes, new Random(), factor, activations);
                manager = new ExManager(nn);

                lblNNStatus.Content = "NN Created!";

                nnIsCreated = true;

                if (dataIsLoaded)
                {
                    PreviewTestImage();
                }
            }
            catch (Exception exception)
            {
                MessageBox.Show(exception.ToString(), "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void PreviewTestImage()
        {
            double[] res = nn.Forward(data.TestImages[currTestIndex].RawImage);
            lblNNOut0.Content = string.Format("{0:N2}", res[0]);
            lblNNOut1.Content = string.Format("{0:N2}", res[1]);
            lblNNOut2.Content = string.Format("{0:N2}", res[2]);
            lblNNOut3.Content = string.Format("{0:N2}", res[3]);
            lblNNOut4.Content = string.Format("{0:N2}", res[4]);
            lblNNOut5.Content = string.Format("{0:N2}", res[5]);
            lblNNOut6.Content = string.Format("{0:N2}", res[6]);
            lblNNOut7.Content = string.Format("{0:N2}", res[7]);
            lblNNOut8.Content = string.Format("{0:N2}", res[8]);
            lblNNOut9.Content = string.Format("{0:N2}", res[9]);

            lblTestNN.Content = NNPredict(data.TestImages[currTestIndex].RawImage);
        }

        private void BtnTrain_Click(object sender, RoutedEventArgs e)
        {
            if (!dataIsLoaded || !nnIsCreated)
            {
                MessageBox.Show("Plaese create a NN and load data", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }


            if (!isTraining && !bw.IsBusy)
            {
                try
                {
                    eta = double.Parse(txtBoxEta.Text, CultureInfo.InvariantCulture);
                    momentum = double.Parse(txtBoxMomentum.Text, CultureInfo.InvariantCulture);
                }
                catch (Exception exception)
                {
                    MessageBox.Show(exception.ToString(), "Alert", MessageBoxButton.OK, MessageBoxImage.Error);
                    return;
                }

                isTraining = true;
                bw.RunWorkerAsync();
                btnTrain.Content = "Stop";
            }
            else if (bw.IsBusy)
            {
                bw.CancelAsync();
            }
        }
    }
}
