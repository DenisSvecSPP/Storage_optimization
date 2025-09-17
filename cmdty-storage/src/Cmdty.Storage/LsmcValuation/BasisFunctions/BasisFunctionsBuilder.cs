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
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;
using JetBrains.Annotations;
using Microsoft.CodeAnalysis.CSharp.Scripting;
using Microsoft.CodeAnalysis.Scripting;

namespace Cmdty.Storage
{
    /// <summary>
    /// Used for combining basis functions.
    /// </summary>
    public sealed class BasisFunctionsBuilder : IEnumerable<BasisFunction>
    {
        private readonly IEnumerable<BasisFunction> _functions;

        public BasisFunctionsBuilder(BasisFunction basisFunction) =>
            _functions = new[] { basisFunction };

        public BasisFunctionsBuilder(IEnumerable<BasisFunction> basisFunctions) =>
            _functions = basisFunctions;

        public static implicit operator BasisFunctionsBuilder(BasisFunction basisFunction) => 
            new BasisFunctionsBuilder(basisFunction);

        public static implicit operator BasisFunctionsBuilder(BasisFunction[] basisFunctions) =>
            new BasisFunctionsBuilder(basisFunctions);

        public static implicit operator BasisFunctionsBuilder(List<BasisFunction> basisFunctions) =>
            new BasisFunctionsBuilder(basisFunctions);

        public static implicit operator BasisFunction[](BasisFunctionsBuilder builder) => builder._functions.ToArray();

        public static implicit operator List<BasisFunction>(BasisFunctionsBuilder builder) => builder._functions.ToList();

        public static BasisFunctionsBuilder operator +(BasisFunctionsBuilder builder1, BasisFunctionsBuilder builder2) 
            => Combine(builder1, builder2);

        public static BasisFunctionsBuilder operator +(BasisFunctionsBuilder builder, BasisFunction basisFunction)
            => new BasisFunctionsBuilder(builder._functions.Concat(new [] {basisFunction}));

        public static BasisFunctionsBuilder operator +(BasisFunction basisFunction, BasisFunctionsBuilder builder)
            => new BasisFunctionsBuilder(builder._functions.Concat(new[] { basisFunction }));
        
        public static BasisFunctionsBuilder Combine(BasisFunctionsBuilder builder1, BasisFunctionsBuilder builder2)
            => new BasisFunctionsBuilder(builder1._functions.Concat(builder2._functions));

        public static BasisFunctionsBuilder Ones => new BasisFunctionsBuilder(BasisFunctions.Ones);

        public static BasisFunctionsBuilder SpotPricePower(int power) => new BasisFunctionsBuilder(BasisFunctions.SpotPricePower(power));

        public static BasisFunctionsBuilder MarkovFactorPower(int markovFactor, int power) 
            => new BasisFunctionsBuilder(BasisFunctions.MarkovFactorPower(markovFactor, power));

        public static BasisFunctionsBuilder MarkovFactorAllPositiveIntegerPowersUpTo(int markovFactor, int maxPower) 
            => new BasisFunctionsBuilder(BasisFunctions.MarkovFactorAllPositiveIntegerPowersUpTo(markovFactor, maxPower));
        
        public static BasisFunctionsBuilder AllMarkovFactorAllPositiveIntegerPowersUpTo(int maxPower, int numMarkovFactors)
            => new BasisFunctionsBuilder(BasisFunctions.AllMarkovFactorAllPositiveIntegerPowersUpTo(maxPower, numMarkovFactors));

        public static BasisFunction[] Parse([NotNull] string basisFunctionExpression)
        {
            if (basisFunctionExpression == null) throw new ArgumentNullException(nameof(basisFunctionExpression));
            return BasisFunctionsCache.GetOrAdd(basisFunctionExpression, expression =>
            {
                string[] monomials = expression.Split('+').Select(s => s.Trim()).ToArray();
                if (monomials.Length == 0)
                    throw new ArgumentException("Basis function expression contains no monomials.", nameof(expression));
                if (monomials.Distinct().Count() < monomials.Length)
                    throw new ArgumentException("Basis function expression contains repeated monomials.");
                return monomials.Select(ParseMonomial).ToArray();
            });
        }

        private static readonly ScriptOptions ParserScriptOptions;
        private static readonly ConcurrentDictionary<string, BasisFunction[]> BasisFunctionsCache;
        private static readonly ConcurrentDictionary<string, BasisFunction> MonomialsCache;

        static BasisFunctionsBuilder()
        {
            ParserScriptOptions = ScriptOptions.Default.WithImports("Cmdty.Storage.Sim")
                .WithReferences(Assembly.GetExecutingAssembly());
            BasisFunctionsCache = new ConcurrentDictionary<string, BasisFunction[]>();
            MonomialsCache = new ConcurrentDictionary<string, BasisFunction>();
        }

        private static BasisFunction ParseMonomial(string monomialExpression)
        {
            return MonomialsCache.GetOrAdd(monomialExpression, expression =>
            {
                if (monomialExpression == "1")
                    return BasisFunctions.Ones;
                monomialExpression = monomialExpression.Replace('s', 'S');
                // Replace xi with Factor(i)
                monomialExpression = Regex.Replace(monomialExpression, @"x(?<FactorNum>\d+)", match =>
                    $"Factor({match.Groups["FactorNum"].Value})");
                // Replace i**j with i.Pow(j)
                monomialExpression = Regex.Replace(monomialExpression, @"\*\*(?<Power>\d+)", match =>
                    $".Pow({match.Groups["Power"].Value})");
                return CSharpScript.EvaluateAsync<BasisFunction>(monomialExpression, ParserScriptOptions).Result;
            });
        }

        public IEnumerator<BasisFunction> GetEnumerator()
        {
            return _functions.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable) _functions).GetEnumerator();
        }
    }
}
