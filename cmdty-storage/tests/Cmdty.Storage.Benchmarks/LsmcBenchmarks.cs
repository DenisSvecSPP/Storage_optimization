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
using BenchmarkDotNet.Attributes;
using Cmdty.Core.Simulation.MultiFactor;
using Cmdty.TimePeriodValueTypes;
using MathNet.Numerics;
using TimeSeriesFactory = Cmdty.TimeSeries.TimeSeries;

namespace Cmdty.Storage.Benchmarks
{
    public class LsmcBenchmarks
    {
        private const int NumSims = 1_000;
        private const int RandomSeed = 11;
        private const int RegressMaxDegree = 2;
        // TODO make these static?
        private readonly LsmcValuationParameters<Day> _valuationParameters;
        public LsmcBenchmarks()
        {
            var valDate = new Day(2019, 8, 29);

            #region Set Up Simple Storage
            var storageStart = new Day(2019, 12, 1);
            var storageEnd = new Day(2020, 4, 1);
            const double maxWithdrawalRate = 850.0;
            const double maxInjectionRate = 625.0;
            const double maxInventory = 52_500.0;
            const double constantInjectionCost = 1.25;
            const double constantWithdrawalCost = 0.93;

            var simpleDailyStorage = CmdtyStorage<Day>.Builder
                .WithActiveTimePeriod(storageStart, storageEnd)
                .WithConstantInjectWithdrawRange(-maxWithdrawalRate, maxInjectionRate)
                .WithZeroMinInventory()
                .WithConstantMaxInventory(maxInventory)
                .WithPerUnitInjectionCost(constantInjectionCost, injectionDate => injectionDate)
                .WithNoCmdtyConsumedOnInject()
                .WithPerUnitWithdrawalCost(constantWithdrawalCost, withdrawalDate => withdrawalDate)
                .WithNoCmdtyConsumedOnWithdraw()
                .WithNoCmdtyInventoryLoss()
                .WithNoInventoryCost()
                .MustBeEmptyAtEnd()
                .Build();
            #endregion Set Up Simple Storage

            const double oneFactorMeanReversion = 12.5;
            const double oneFactorSpotVol = 0.95;
            var oneFDailyMultiFactorParams = MultiFactorParameters.For1Factor(oneFactorMeanReversion, 
                    TimeSeriesFactory.ForConstantData(valDate, storageEnd, oneFactorSpotVol));
            const double flatInterestRate = 0.055;

            const int numInventorySpacePoints = 100;

            const double baseForwardPrice = 53.5;
            const double forwardSeasonalFactor = 24.6;
            var forwardCurve = TimeSeriesFactory.FromMap(valDate, storageEnd, day =>
            {
                int daysForward = day.OffsetFrom(valDate);
                return baseForwardPrice + Math.Sin(2.0 * Math.PI / 365.0 * daysForward) * forwardSeasonalFactor;
            });

            _valuationParameters = new LsmcValuationParameters<Day>.Builder
                            {
                                BasisFunctions = BasisFunctionsBuilder.Ones +
                                                 BasisFunctionsBuilder.AllMarkovFactorAllPositiveIntegerPowersUpTo(RegressMaxDegree, 1),
                                CurrentPeriod = new Day(2019, 8, 29),
                                DiscountFactors = StorageHelper.CreateAct65ContCompDiscounter(flatInterestRate),
                                ForwardCurve = forwardCurve,
                                GridCalc = FixedSpacingStateSpaceGridCalc.CreateForFixedNumberOfPointsOnGlobalInventoryRange(simpleDailyStorage, numInventorySpacePoints),
                                Inventory = 5_685,
                                Storage = simpleDailyStorage,
                                SettleDateRule = deliveryDate => Month.FromDateTime(deliveryDate.Start).Offset(1).First<Day>() + 19, // Settlement on 20th of following month
                            }
                            .SimulateWithMultiFactorModelAndMersenneTwister(oneFDailyMultiFactorParams, NumSims, RandomSeed)
                            .Build();
        }

        //[Benchmark]
        public double ValueSimpleDailyStorageOneFactor_ManagedNumerics()
        {
            Control.UseManaged();
            LsmcStorageValuationResults<Day> results = LsmcStorageValuation.WithNoLogger.Calculate(_valuationParameters);
            return results.Npv;
        }

        [Benchmark]
        public double ValueSimpleDailyStorageOneFactor_MklNumerics()
        {
            Control.UseNativeMKL();
            LsmcStorageValuationResults<Day> results = LsmcStorageValuation.WithNoLogger.Calculate(_valuationParameters);
            return results.Npv;
        }

        // TODO:
        // Different number of simulations
        // Different grid spacing
        // Different number of factors

    }
}
