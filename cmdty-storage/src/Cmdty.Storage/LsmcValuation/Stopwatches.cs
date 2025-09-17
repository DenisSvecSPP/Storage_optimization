#region License
// Copyright (c) 2020 Jake Fowler
//
// Permission is hereby granted, free of charge, to any person 
// obtaining a copy of this software and associated documentation 
// files (the "Software"), to deal in the Software without 
// restriction, including without limitation the rights to use, 
// copy, modify, merge, publish, distribute, sublicense, and/or sell 
// copies of the Software, and to permit persons to whom the 
// Software is furnished to do so, subject to the following 
// conditions:
//
// The above copyright notice and this permission notice shall be 
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
// OTHER DEALINGS IN THE SOFTWARE.
#endregion

using System;
using System.Diagnostics;
using System.Globalization;
using System.Text;

namespace Cmdty.Storage
{
    internal sealed class Stopwatches
    {
        public Stopwatch All { get; }
        public Stopwatch RegressionPriceSimulation { get; }
        public Stopwatch ValuationPriceSimulation { get; }
        public Stopwatch BackwardInduction { get; }
        public Stopwatch PseudoInverse { get; }
        public Stopwatch ForwardSimulation { get; }

        public Stopwatches()
        {
            All = new Stopwatch();
            RegressionPriceSimulation = new Stopwatch();
            ValuationPriceSimulation = new Stopwatch();
            BackwardInduction = new Stopwatch();
            PseudoInverse = new Stopwatch();
            ForwardSimulation = new Stopwatch();
        }

        public string GenerateProfileReport()
        {
            var stringBuilder = new StringBuilder();
            TimeSpan otherAll = All.Elapsed - RegressionPriceSimulation.Elapsed - BackwardInduction.Elapsed - ValuationPriceSimulation.Elapsed -
                                ForwardSimulation.Elapsed;
            TimeSpan otherBackwardInduction = BackwardInduction.Elapsed - PseudoInverse.Elapsed;

            string regressPriceSimPercent =
                (RegressionPriceSimulation.Elapsed.Ticks / (double)All.Elapsed.Ticks).ToString("P2", CultureInfo.InvariantCulture);
            string valuationPriceSimPercent =
                (ValuationPriceSimulation.Elapsed.Ticks / (double)All.Elapsed.Ticks).ToString("P2", CultureInfo.InvariantCulture);
            string pseudoInversePercent =
                (PseudoInverse.Elapsed.Ticks / (double)All.Elapsed.Ticks).ToString("P2", CultureInfo.InvariantCulture);
            string otherBackInductionPercent =
                (otherBackwardInduction.Ticks / (double)All.Elapsed.Ticks).ToString("P2", CultureInfo.InvariantCulture);
            string forwardSimPercent =
                (ForwardSimulation.Elapsed.Ticks / (double)All.Elapsed.Ticks).ToString("P2", CultureInfo.InvariantCulture);
            string otherPercent = (otherAll.Ticks / (double)All.Elapsed.Ticks).ToString("P2", CultureInfo.InvariantCulture);

            stringBuilder.AppendLine("Total:\t\t\t" + All.Elapsed.ToString("g", CultureInfo.InvariantCulture));
            stringBuilder.AppendLine($"Regress price sim:\t{RegressionPriceSimulation.Elapsed.ToString("g", CultureInfo.InvariantCulture)}\t({regressPriceSimPercent})");
            stringBuilder.AppendLine($"Val price sim:\t\t{ValuationPriceSimulation.Elapsed.ToString("g", CultureInfo.InvariantCulture)}\t({valuationPriceSimPercent})");
            stringBuilder.AppendLine($"Pseudo-inverse:\t\t{PseudoInverse.Elapsed.ToString("g", CultureInfo.InvariantCulture)}\t({pseudoInversePercent})");
            stringBuilder.AppendLine($"Other back ind:\t\t{otherBackwardInduction.ToString("g", CultureInfo.InvariantCulture)}\t({otherBackInductionPercent})");
            stringBuilder.AppendLine($"Fwd sim:\t\t{ForwardSimulation.Elapsed.ToString("g", CultureInfo.InvariantCulture)}\t({forwardSimPercent})");
            stringBuilder.AppendLine($"Other:\t\t\t{otherAll.ToString("g", CultureInfo.InvariantCulture)}\t({otherPercent})");

            return stringBuilder.ToString();
        }

    }
}