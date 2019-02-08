using NeuralNetworkPrev.Controller;
using NeuralNetworkPrev.Model;
using NNv1;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
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

namespace NeuralNetworkPrev.Views
{
    /// <summary>
    /// Interaction logic for BinaryPage.xaml
    /// </summary>
    public partial class BinaryPage : Page
    {
        private (double[], double[])[] data;
        private NeuralNetwork nn;
        private SGDManager manager;

        bool isTraining = false;

        private double eta = 2;
        private double momentum = .6;
        private double errorBreak = 0.001;

        private BackgroundWorker bw;
        private delegate void ShowError(double err);
        private delegate void RenderNN(int itteration);

        public BinaryPage()
        {
            InitializeComponent();

            nn = new NeuralNetwork(new int[] { 2, 2, 1 }, 0.7);
            manager = new SGDManager(nn);

            bw = new BackgroundWorker();
            bw.WorkerSupportsCancellation = true;
            bw.DoWork += bw_DoWork;
            bw.RunWorkerCompleted += bw_RunWorkerComplited;

            lblNN.Content = nn;
        }

        private void bw_RunWorkerComplited(object sender, RunWorkerCompletedEventArgs e)
        {
            isTraining = false;
            btnTrain.Content = "Train";
        }

        private void bw_DoWork(object sender, DoWorkEventArgs e)
        {
            double err;
            int itteration = 0;
            do
            {
                if (bw.CancellationPending)
                {
                    e.Cancel = true;
                    return;
                }

                itteration++;

                System.Threading.Thread.Sleep(1);
                //err = manager.UpdateMiniBatch(nn, data, eta, momentum);
                manager.eta = eta;
                manager.momentum = momentum;
                err = manager.Train(data);
                ShowError shower = new ShowError(DisplayError);
                lblError.Dispatcher.BeginInvoke(shower, err);

                System.Threading.Thread.Sleep(1);
                RenderNN renderer = new RenderNN(DisplayNN);
                lblNN.Dispatcher.BeginInvoke(renderer, itteration);

            } while (err > errorBreak);
        }

        private void DisplayError(double error)
        {
            lblError.Content = string.Format("{0:N10}", error);
        }

        private void DisplayNN(int itteration)
        {
            lblNN.Content = nn + "\n\nItteration: " + itteration;
        }

        private void BtnTrain_Click(object sender, RoutedEventArgs e)
        {
            if (!isTraining && !bw.IsBusy)
            {
                try
                {
                    eta = double.Parse(txtBoxEta.Text, CultureInfo.InvariantCulture);
                    momentum = double.Parse(txtBoxMomentum.Text, CultureInfo.InvariantCulture);
                    errorBreak = double.Parse(txtBoxErrorBreak.Text, CultureInfo.InvariantCulture);

                    //eta = Convert.ToDouble(txtBoxEta.Text);
                    //momentum = Convert.ToDouble(txtBoxMomentum.Text);
                    //errorBreak = Convert.ToDouble(txtBoxErrorBreak.Text);
                }
                catch (Exception exception)
                {
                    MessageBox.Show(exception.ToString(), "Alert", MessageBoxButton.OK, MessageBoxImage.Error);
                }

                bw.RunWorkerAsync();
                isTraining = true;
                btnTrain.Content = "Stop";
            }
            else if (bw.IsBusy)
            {
                bw.CancelAsync();
            }
        }

        private void BtnTest_Click(object sender, RoutedEventArgs e)
        {
            double res00 = nn.Forward(data[0].Item1)[0];
            double res01 = nn.Forward(data[1].Item1)[0];
            double res10 = nn.Forward(data[2].Item1)[0];
            double res11 = nn.Forward(data[3].Item1)[0];

            lblOutput00.Content = string.Format("{0:N0}    ({0:N2})", res00);
            lblOutput01.Content = string.Format("{0:N0}    ({0:N2})", res01);
            lblOutput10.Content = string.Format("{0:N0}    ({0:N2})", res10);
            lblOutput11.Content = string.Format("{0:N0}    ({0:N2})", res11);
        }

        private void RadBtnOr_Checked(object sender, RoutedEventArgs e)
        {
            data = Models.GetBinaryData(0);
        }

        private void RadBtnAnd_Checked(object sender, RoutedEventArgs e)
        {
            data = Models.GetBinaryData(1);
        }

        private void RadBtnXor_Checked(object sender, RoutedEventArgs e)
        {
            data = Models.GetBinaryData(2);
        }

        private void BtnRefresh_Click(object sender, RoutedEventArgs e)
        {
            nn = new NeuralNetwork(new int[] { 2, 2, 1 }, 0.7);
            manager = new SGDManager(nn);
            lblNN.Content = nn;
        }
    }
}
