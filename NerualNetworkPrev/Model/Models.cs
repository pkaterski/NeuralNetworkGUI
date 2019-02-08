using MNISTLib;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkPrev.Model
{
    class Models
    {
        public static (double[], double[])[] GetBinaryData(int operation)
        {
            double[] i00 = new double[2] { 0, 0 };
            double[] i01 = new double[2] { 0, 1 };
            double[] i10 = new double[2] { 1, 0 };
            double[] i11 = new double[2] { 1, 1 };

            double[] o00;
            double[] o01;
            double[] o10;
            double[] o11;

            switch (operation)
            {
                case 0:
                    o00 = new double[1] { 0 };
                    o01 = new double[1] { 1 };
                    o10 = new double[1] { 1 };
                    o11 = new double[1] { 1 };
                    break;
                case 1:
                    o00 = new double[1] { 0 };
                    o01 = new double[1] { 0 };
                    o10 = new double[1] { 0 };
                    o11 = new double[1] { 1 };
                    break;
                case 2:
                    o00 = new double[1] { 0 };
                    o01 = new double[1] { 1 };
                    o10 = new double[1] { 1 };
                    o11 = new double[1] { 0 };
                    break;
                default:
                    o00 = new double[1] { 0 };
                    o01 = new double[1] { 0 };
                    o10 = new double[1] { 0 };
                    o11 = new double[1] { 0 };
                    break;
            }

            (double[], double[])[] miniBatch = new (double[], double[])[4] { (i00, o00), (i01, o01), (i10, o10), (i11, o11) };

            return miniBatch;
        }

        // returns data and if data loading was successful
        public static (MNISTCore, bool) GetMNISTData(int train, int test, string path)
        {
            MNISTCore mnist = new MNISTCore();
            bool loaded = mnist.LoadDB(path, train, test);

            if (loaded)
            {
                return (mnist, true);
            }
            else return (null, false);
        }

    }
}
