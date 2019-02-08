using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNISTLib
{
    public class DigitImage
    {
        private const int DIM_SIZE = 28;
        public static int SIZE = DIM_SIZE * DIM_SIZE;

        private byte[][] _pixels;
        private byte _label;
        private int[,] v;
        private int idx;

        public byte Label
        {
            get { return _label; }
            set { _label = value; }
        }

        public byte[][] Pixels
        {
            get { return _pixels; }
            set { _pixels = value; }
        }

        public double[] RawImage
        {
            get
            {
                double[] res = new double[SIZE];

                for (int i = 0; i < DIM_SIZE; i++)
                {
                    for (int j = 0; j < DIM_SIZE; j++)
                    {
                        res[j * DIM_SIZE + i] = _pixels[j][i] / 254.0;
                    }
                }

                return res;
            }
        }

        public DigitImage(byte[][] pixels, byte label)
        {
            _pixels = new byte[28][];
            for (int i = 0; i < _pixels.Length; i++)
                _pixels[i] = new byte[28];

            for (int i = 0; i < 28; i++)
                for (int j = 0; j < 28; j++)
                    _pixels[i][j] = pixels[i][j];

            _label = label;
        }

        public DigitImage(int[][] pixels, int label)
        {
            _pixels = new byte[28][];
            for (int i = 0; i < _pixels.Length; i++)
                _pixels[i] = new byte[28];

            for (int i = 0; i < 28; i++)
                for (int j = 0; j < 28; j++)
                    _pixels[i][j] = (byte)pixels[j][i]; //transpose j & i

            _label = (byte)label;
        }

        public DigitImage(int[,] pixels, int label)
        {
            _pixels = new byte[28][];
            for (int i = 0; i < _pixels.Length; i++)
                _pixels[i] = new byte[28];

            for (int i = 0; i < 28; i++)
                for (int j = 0; j < 28; j++)
                    _pixels[i][j] = (byte)pixels[j, i]; //transpose j & i

            _label = (byte)label;
        }

        public override string ToString()
        {
            string s = "";
            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    if (_pixels[i][j] == 0)
                        s += " "; //white
                    else if (_pixels[i][j] == 255)
                        s += "0"; //black
                    else
                        s += "."; //gray
                }
                s += "\n";
            }
            s += _label.ToString();
            return s;
        }
    }
}
