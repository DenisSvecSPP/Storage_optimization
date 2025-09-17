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
using System.Collections.Generic;
using System.Linq;
using Xunit;
using static Cmdty.Storage.Sim;

namespace Cmdty.Storage.Test
{
    public sealed class BasisFunctionsBuilderTest
    {

        [Fact]
        [Trait("Category", "Lsmc.BasisFunctions")]
        public void Parse_ValidInput_AsExpected()
        {
            const string expression = "1 + s + s*s + s*s**3 + x0 + x1**3 + s*x2**4";
            BasisFunction[] basisFunctions = BasisFunctionsBuilder.Parse(expression);
            
            Assert.Equal(7, basisFunctions.Length);

            var spotPriceSims = new[] { 25.69, 21.88, 16.78 };
            var markovFactors = new ReadOnlyMemory<double>[]
            {
                new[]{ 0.56, 0.12, 1.55},
                new[]{ 1.08, 2.088, 0.988},
                new[]{ 2.808, 0.998, 5.84}
            };

            // 1s
            AssertBasisFunction(basisFunctions[0], spotPriceSims, markovFactors, new[] {1.0, 1.0, 1.0});
            // Spot price
            AssertBasisFunction(basisFunctions[1], spotPriceSims, markovFactors, spotPriceSims);
            // Spot price squared
            AssertBasisFunction(basisFunctions[2], spotPriceSims, markovFactors, spotPriceSims.Select(s=>s*s));
            // Spot price to power of 4
            AssertBasisFunction(basisFunctions[3], spotPriceSims, markovFactors, spotPriceSims.Select(s => s*s*s*s));
            // First factor
            AssertBasisFunction(basisFunctions[4], spotPriceSims, markovFactors, markovFactors[0].ToArray());
            // Second factor cubed
            AssertBasisFunction(basisFunctions[5], spotPriceSims, markovFactors, markovFactors[1].ToArray().Select(x=>x*x*x));
            // Spot price times third factor to power of 4
            double[] thirdFactor = markovFactors[2].ToArray();
            AssertBasisFunction(basisFunctions[6], spotPriceSims, markovFactors, 
                spotPriceSims.Select((s, i) => s* Math.Pow(thirdFactor[i], 4)));
        }

        private static void AssertBasisFunction(BasisFunction basisFunction, double[] spotSims, ReadOnlyMemory<double>[] markovSims, 
            IEnumerable<double> expectedResults)
        {
            var results = new double[spotSims.Length];
            basisFunction(markovSims, spotSims, results);
            Assert.Equal(expectedResults, results);
        }


        [Fact]
        [Trait("Category", "Lsmc.BasisFunctions")]
        public void CombineWithAddOperator_AsExpected()
        {
            BasisFunction[] basisFunctions = BasisFunctionsBuilder.Ones +
                Spot + Spot * Spot + Spot * Spot.Pow(3) + X0 + X1.Pow(3) + S * X2.Pow(4);

            Assert.Equal(7, basisFunctions.Length);

            var spotPriceSims = new[] { 25.69, 21.88, 16.78 };
            var markovFactors = new ReadOnlyMemory<double>[]
            {
                new[]{ 0.56, 0.12, 1.55},
                new[]{ 1.08, 2.088, 0.988},
                new[]{ 2.808, 0.998, 5.84}
            };

            // 1s
            AssertBasisFunction(basisFunctions[0], spotPriceSims, markovFactors, new[] { 1.0, 1.0, 1.0 });
            // Spot price
            AssertBasisFunction(basisFunctions[1], spotPriceSims, markovFactors, spotPriceSims);
            // Spot price squared
            AssertBasisFunction(basisFunctions[2], spotPriceSims, markovFactors, spotPriceSims.Select(s => s * s));
            // Spot price to power of 4
            AssertBasisFunction(basisFunctions[3], spotPriceSims, markovFactors, spotPriceSims.Select(s => s * s * s * s));
            // First factor
            AssertBasisFunction(basisFunctions[4], spotPriceSims, markovFactors, markovFactors[0].ToArray());
            // Second factor cubed
            AssertBasisFunction(basisFunctions[5], spotPriceSims, markovFactors, markovFactors[1].ToArray().Select(x => x * x * x));
            // Spot price times third factor to power of 4
            double[] thirdFactor = markovFactors[2].ToArray();
            AssertBasisFunction(basisFunctions[6], spotPriceSims, markovFactors,
                spotPriceSims.Select((s, i) => s * Math.Pow(thirdFactor[i], 4)));
        }

    }
}
