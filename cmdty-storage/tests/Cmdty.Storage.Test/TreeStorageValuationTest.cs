﻿#region License
// Copyright (c) 2019 Jake Fowler
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
using Cmdty.TimePeriodValueTypes;
using Cmdty.TimeSeries;
using Xunit;

namespace Cmdty.Storage.Test
{
    public sealed class TreeStorageValuationTest
    {
        [Fact]
        public void Calculate_StorageLooksLikeCallOptions_NpvEqualsBlack76()
        {
            const double percentTolerance = 0.005; // 0.5% tolerance
            var currentDate = new Day(2019, 8, 29);

            (DoubleTimeSeries<Day> forwardCurve, DoubleTimeSeries<Day> spotVolCurve) = 
                TestHelper.CreateDailyTestForwardAndSpotVolCurves(currentDate, new Day(2020, 4, 1));
            const double meanReversion = 16.5;
            const double timeDelta = 1.0 / 365.0;
            const double interestRate = 0.09;

            TestHelper.CallOptionLikeTestData testData = TestHelper.CreateThreeCallsLikeStorageTestData(forwardCurve);
            
            TreeStorageValuationResults<Day> valuationResults = 
                TreeStorageValuation<Day>.ForStorage(testData.Storage)
                .WithStartingInventory(testData.Inventory)
                .ForCurrentPeriod(currentDate)
                .WithForwardCurve(forwardCurve)
                .WithOneFactorTrinomialTree(spotVolCurve, meanReversion, timeDelta)
                .WithMonthlySettlement(testData.SettleDates)
                .WithAct365ContinuouslyCompoundedInterestRate(day => interestRate) // Constant interest rate
                .WithFixedNumberOfPointsOnGlobalInventoryRange(100)
                .WithLinearInventorySpaceInterpolation()
                .WithNumericalTolerance(1E-10)
                .Calculate();

            // Calculate value of equivalent call options
            double expectStorageValue = 0.0;
            foreach (TestHelper.CallOption option in testData.CallOptions)
            {
                double impliedVol = TestHelper.OneFactorImpliedVol(currentDate, option.ExpiryDate, spotVolCurve, meanReversion);
                double forwardPrice = forwardCurve[option.ExpiryDate];
                double black76Value = TestHelper.Black76CallOptionValue(currentDate, forwardPrice,
                                          impliedVol, interestRate, option.StrikePrice, option.ExpiryDate,
                                          option.SettleDate) * option.NotionalVolume;
                expectStorageValue += black76Value;
            }

            double percentError = (valuationResults.NetPresentValue - expectStorageValue) / expectStorageValue;

            Assert.InRange(percentError, -percentTolerance, percentTolerance);
        }
        
        [Fact]
        public void Calculate_StorageWithForcedInjectAndWithdraw_NpvEqualsTrivialIntrinsicCalc()
        {
            var currentDate = new Day(2019, 8, 29);

            var storageStart = new Day(2019, 12, 1);
            var storageEnd = new Day(2020, 4, 1);

            const double storageStartingInventory = 0.0;
            const double minInventory = 0.0;
            const double maxInventory = 10_000.0;

            const double forcedInjectionRate = 211.5;
            const int forcedInjectionNumDays = 20;
            var forcedInjectionStart = new Day(2019, 12, 20);

            const double injectionPerUnitCost = 1.23;
            const double injectionCmdtyConsumed = 0.01;

            const double forcedWithdrawalRate = 187.54;
            const int forcedWithdrawalNumDays = 15;
            var forcedWithdrawalStart = new Day(2020, 2, 5);

            const double withdrawalPerUnitCost = 0.98;
            const double withdrawalCmdtyConsumed = 0.015;
            
            (DoubleTimeSeries<Day> forwardCurve, DoubleTimeSeries<Day> spotVolCurve) = TestHelper.CreateDailyTestForwardAndSpotVolCurves(currentDate, storageEnd);
            const double meanReversion = 16.5;
            const double timeDelta = 1.0 / 365.0;
            const double interestRate = 0.09;

            TimeSeries<Month, Day> settlementDates = new TimeSeries<Month, Day>.Builder()
            {
                { new Month(2019, 12),  new Day(2020, 1, 20)},
                { new Month(2020, 1),  new Day(2020, 2, 18)},
                { new Month(2020, 2),  new Day(2020, 3, 21)},
                { new Month(2020, 3),  new Day(2020, 4, 22)}
            }.Build();

            var injectWithdrawConstraints = new List<InjectWithdrawRangeByInventoryAndPeriod<Day>>
            {
                (period: storageStart, injectWithdrawRanges: new List<InjectWithdrawRangeByInventory>
                {
                    (inventory: minInventory, (minInjectWithdrawRate: 0.0, maxInjectWithdrawRate: 0.0)),
                    (inventory: maxInventory, (minInjectWithdrawRate: 0.0, maxInjectWithdrawRate: 0.0))
                }),
                (period: forcedInjectionStart, injectWithdrawRanges: new List<InjectWithdrawRangeByInventory>
                {
                    (inventory: minInventory, (minInjectWithdrawRate: forcedInjectionRate, maxInjectWithdrawRate: forcedInjectionRate)),
                    (inventory: maxInventory, (minInjectWithdrawRate: forcedInjectionRate, maxInjectWithdrawRate: forcedInjectionRate))
                }),
                (period: forcedInjectionStart.Offset(forcedInjectionNumDays), injectWithdrawRanges: new List<InjectWithdrawRangeByInventory>
                {
                    (inventory: minInventory, (minInjectWithdrawRate: 0.0, maxInjectWithdrawRate: 0.0)),
                    (inventory: maxInventory, (minInjectWithdrawRate: 0.0, maxInjectWithdrawRate: 0.0))
                }),
                (period: forcedWithdrawalStart, injectWithdrawRanges: new List<InjectWithdrawRangeByInventory>
                {
                    (inventory: minInventory, (minInjectWithdrawRate: -forcedWithdrawalRate, maxInjectWithdrawRate: -forcedWithdrawalRate)),
                    (inventory: maxInventory, (minInjectWithdrawRate: -forcedWithdrawalRate, maxInjectWithdrawRate: -forcedWithdrawalRate))
                }),
                (period: forcedWithdrawalStart.Offset(forcedWithdrawalNumDays), injectWithdrawRanges: new List<InjectWithdrawRangeByInventory>
                {
                    (inventory: minInventory, (minInjectWithdrawRate: 0.0, maxInjectWithdrawRate: 0.0)),
                    (inventory: maxInventory, (minInjectWithdrawRate: 0.0, maxInjectWithdrawRate: 0.0))
                }),
            };

            Day InjectionCostPaymentTerms(Day injectionDate)
            {
                return injectionDate.Offset(10);
            }

            Day WithdrawalCostPaymentTerms(Day withdrawalDate)
            {
                return withdrawalDate.Offset(4);
            }
            
            CmdtyStorage<Day> storage = CmdtyStorage<Day>.Builder
                .WithActiveTimePeriod(storageStart, storageEnd)
                .WithTimeAndInventoryVaryingInjectWithdrawRatesPolynomial(injectWithdrawConstraints)
                .WithPerUnitInjectionCost(injectionPerUnitCost, InjectionCostPaymentTerms)
                .WithFixedPercentCmdtyConsumedOnInject(injectionCmdtyConsumed)
                .WithPerUnitWithdrawalCost(withdrawalPerUnitCost, WithdrawalCostPaymentTerms)
                .WithFixedPercentCmdtyConsumedOnWithdraw(withdrawalCmdtyConsumed)
                .WithNoCmdtyInventoryLoss()
                .WithNoInventoryCost()
                .WithTerminalInventoryNpv((cmdtySpotPrice, inventory) => 0.0)
                .Build();

            TreeStorageValuationResults<Day> valuationResults = TreeStorageValuation<Day>.ForStorage(storage)
                .WithStartingInventory(storageStartingInventory)
                .ForCurrentPeriod(currentDate)
                .WithForwardCurve(forwardCurve)
                .WithOneFactorTrinomialTree(spotVolCurve, meanReversion, timeDelta)
                .WithMonthlySettlement(settlementDates)
                .WithAct365ContinuouslyCompoundedInterestRate(day => interestRate) // Constant interest rate
                .WithFixedNumberOfPointsOnGlobalInventoryRange(100)
                .WithLinearInventorySpaceInterpolation()
                .WithNumericalTolerance(1E-10)
                .Calculate();
            
            // Calculate the NPV Manually

            // Period of forced injection
            double injectionPv = 0.0;
            for (int i = 0; i < forcedInjectionNumDays; i++)
            {
                Day injectionDate = forcedInjectionStart.Offset(i);
                double forwardPrice = forwardCurve[injectionDate];

                Day cmdtySettlementDate = settlementDates[Month.FromDateTime(injectionDate.Start)];
                double cmdtyDiscountFactor =
                    Act365ContCompoundDiscountFactor(currentDate, cmdtySettlementDate, interestRate);

                Day injectionCostSettlementDate = InjectionCostPaymentTerms(injectionDate);
                double injectCostDiscountFactor =
                    Act365ContCompoundDiscountFactor(currentDate, injectionCostSettlementDate, interestRate);

                double cmdtyBoughtPv = -forwardPrice * forcedInjectionRate * (1 + injectionCmdtyConsumed) * cmdtyDiscountFactor;
                double injectCostPv = -injectionPerUnitCost * forcedInjectionRate * injectCostDiscountFactor;

                injectionPv += cmdtyBoughtPv + injectCostPv;
            }

            // Period of forced withdrawal
            double withdrawalPv = 0.0;
            for (int i = 0; i < forcedWithdrawalNumDays; i++)
            {
                Day withdrawalDate = forcedWithdrawalStart.Offset(i);
                double forwardPrice = forwardCurve[withdrawalDate];

                Day cmdtySettlementDate = settlementDates[Month.FromDateTime(withdrawalDate.Start)];
                double cmdtyDiscountFactor =
                    Act365ContCompoundDiscountFactor(currentDate, cmdtySettlementDate, interestRate);

                Day withdrawalCostSettlementDate = WithdrawalCostPaymentTerms(withdrawalDate);
                double withdrawalCostDiscountFactor =
                    Act365ContCompoundDiscountFactor(currentDate, withdrawalCostSettlementDate, interestRate);

                double cmdtySoldPv = forwardPrice * forcedWithdrawalRate * (1 - withdrawalCmdtyConsumed) * cmdtyDiscountFactor;
                double withdrawalCostPv = -withdrawalPerUnitCost * forcedWithdrawalRate * withdrawalCostDiscountFactor;

                withdrawalPv += cmdtySoldPv + withdrawalCostPv;
            }

            double expectedNpv = injectionPv + withdrawalPv;

            Assert.Equal(expectedNpv, valuationResults.NetPresentValue, 10);
        }

        private static double Act365ContCompoundDiscountFactor(Day currentDate, Day paymentDate, double interestRate)
        {
            return Math.Exp(-paymentDate.OffsetFrom(currentDate) / 365.0 * interestRate);
        }

        [Fact(Skip = "Trying to figure out why this isn't passing.")] // TODO this isn't passing because in the injection period there is still flexibility to shift are when the injection occurs, generating extrinsic value
        public void Calculate_DeepInTheMoney_NpvEqualsTrivialIntrinsicCalc()
        {
            var currentDate = new Day(2019, 8, 29);

            var storageStart = new Day(2019, 12, 1);
            var storageEnd = new Day(2020, 4, 1);

            const double storageStartingInventory = 0.0;
            const double minInventory = 0.0;
            const double maxInventory = 100_000.0;

            const double injectionPerUnitCost = 1.23;
            const double injectionCmdtyConsumed = 0.01;

            const double withdrawalPerUnitCost = 0.98;
            const double withdrawalCmdtyConsumed = 0.015;

            const double meanReversion = 16.5;
            const double timeDelta = 1.0 / 365.0;

            const double lowPrice = 23.87;
            const double highPrice = 150.32;
            const int numDaysAtHighPrice = 20;

            const double injectionRate = 700.0;
            const double withdrawalRate = 700.0;

            Day dateToSwitchToHighForwardPrice = storageEnd.Offset(-numDaysAtHighPrice);

            var forwardCurveBuilder = new TimeSeries<Day, double>.Builder(storageEnd - currentDate + 1);

            foreach (Day day in currentDate.EnumerateTo(storageEnd))
            {
                double forwardPrice = day < dateToSwitchToHighForwardPrice ? lowPrice : highPrice;
                forwardCurveBuilder.Add(day, forwardPrice);
            }

            TimeSeries<Day, double> forwardCurve = forwardCurveBuilder.Build();
            DoubleTimeSeries<Day> spotVolCurve = TestHelper.CreateDailyTestForwardAndSpotVolCurves(currentDate, storageEnd).spotVolCurve;
            
            Day InjectionCostPaymentTerms(Day injectionDate)
            {
                return injectionDate.Offset(10);
            }

            Day WithdrawalCostPaymentTerms(Day withdrawalDate)
            {
                return withdrawalDate.Offset(4);
            }

            CmdtyStorage<Day> storage = CmdtyStorage<Day>.Builder
                .WithActiveTimePeriod(storageStart, storageEnd)
                .WithConstantInjectWithdrawRange(-withdrawalRate, injectionRate)
                .WithConstantMinInventory(minInventory)
                .WithConstantMaxInventory(maxInventory)
                .WithPerUnitInjectionCost(injectionPerUnitCost, InjectionCostPaymentTerms)
                .WithFixedPercentCmdtyConsumedOnInject(injectionCmdtyConsumed)
                .WithPerUnitWithdrawalCost(withdrawalPerUnitCost, WithdrawalCostPaymentTerms)
                .WithFixedPercentCmdtyConsumedOnWithdraw(withdrawalCmdtyConsumed)
                .WithNoCmdtyInventoryLoss()
                .WithNoInventoryCost()
                .MustBeEmptyAtEnd()
                .Build();

            TreeStorageValuationResults<Day> valuationResults = TreeStorageValuation<Day>.ForStorage(storage)
                .WithStartingInventory(storageStartingInventory)
                .ForCurrentPeriod(currentDate)
                .WithForwardCurve(forwardCurve)
                .WithOneFactorTrinomialTree(spotVolCurve, meanReversion, timeDelta)
                .WithCmdtySettlementRule(day => day)                     // No discounting 
                .WithDiscountFactorFunc((presentDate, cashFlowDate) => 1.0)
                .WithFixedNumberOfPointsOnGlobalInventoryRange(100)
                .WithLinearInventorySpaceInterpolation()
                .WithNumericalTolerance(1E-10)
                .Calculate();
            
            IntrinsicStorageValuationResults<Day> intrinsicResults = IntrinsicStorageValuation<Day>.ForStorage(storage)
                .WithStartingInventory(storageStartingInventory)
                .ForCurrentPeriod(currentDate)
                .WithForwardCurve(forwardCurve)
                .WithCmdtySettlementRule(day => day)                     // No discounting 
                .WithDiscountFactorFunc((valuationDate, cashFlowDate) => 1.0)         // No discounting
                .WithFixedGridSpacing(100)
                .WithLinearInventorySpaceInterpolation()
                .WithNumericalTolerance(1E-10)
                .Calculate();


            // Trivial intrinsic calc
            double volumeCycled = numDaysAtHighPrice * withdrawalRate;
            double withdrawPv = highPrice * volumeCycled * (1 - withdrawalCmdtyConsumed) - volumeCycled * withdrawalPerUnitCost;

            double injectionPv = -lowPrice * volumeCycled * (1 + injectionCmdtyConsumed) - volumeCycled * injectionPerUnitCost;

            double totalExpectedPv = withdrawPv + injectionPv;

            Assert.Equal(totalExpectedPv, valuationResults.NetPresentValue);
        }

        // TODO this test fails when injectionRate and withdrawalRate are not multiples- investigate this
        [Fact]
        public void Calculate_DeepInTheMoneyWithIntrinsicTree_NpvEqualsTrivialIntrinsicCalc()
        {
            var currentDate = new Day(2019, 8, 29);

            var storageStart = new Day(2019, 12, 1);
            var storageEnd = new Day(2020, 4, 1);

            const double storageStartingInventory = 0.0;
            const double minInventory = 0.0;
            const double maxInventory = 100_000.0;

            const double injectionPerUnitCost = 1.23;
            const double injectionCmdtyConsumed = 0.01;

            const double withdrawalPerUnitCost = 0.98;
            const double withdrawalCmdtyConsumed = 0.015;

            const double lowPrice = 23.87;
            const double highPrice = 150.32;
            const int numDaysAtHighPrice = 20;

            const double injectionRate = 400.0;
            const double withdrawalRate = 800.0;

            Day dateToSwitchToHighForwardPrice = storageEnd.Offset(-numDaysAtHighPrice);

            var forwardCurveBuilder = new TimeSeries<Day, double>.Builder(storageEnd - currentDate + 1);

            foreach (Day day in currentDate.EnumerateTo(storageEnd))
            {
                double forwardPrice = day < dateToSwitchToHighForwardPrice ? lowPrice : highPrice;
                forwardCurveBuilder.Add(day, forwardPrice);
            }

            TimeSeries<Day, double> forwardCurve = forwardCurveBuilder.Build();

            Day InjectionCostPaymentTerms(Day injectionDate)
            {
                return injectionDate.Offset(10);
            }

            Day WithdrawalCostPaymentTerms(Day withdrawalDate)
            {
                return withdrawalDate.Offset(4);
            }

            CmdtyStorage<Day> storage = CmdtyStorage<Day>.Builder
                .WithActiveTimePeriod(storageStart, storageEnd)
                .WithConstantInjectWithdrawRange(-withdrawalRate, injectionRate)
                .WithConstantMinInventory(minInventory)
                .WithConstantMaxInventory(maxInventory)
                .WithPerUnitInjectionCost(injectionPerUnitCost, InjectionCostPaymentTerms)
                .WithFixedPercentCmdtyConsumedOnInject(injectionCmdtyConsumed)
                .WithPerUnitWithdrawalCost(withdrawalPerUnitCost, WithdrawalCostPaymentTerms)
                .WithFixedPercentCmdtyConsumedOnWithdraw(withdrawalCmdtyConsumed)
                .WithNoCmdtyInventoryLoss()
                .WithNoInventoryCost()
                .MustBeEmptyAtEnd()
                .Build();

            TreeStorageValuationResults<Day> valuationResults = TreeStorageValuation<Day>.ForStorage(storage)
                .WithStartingInventory(storageStartingInventory)
                .ForCurrentPeriod(currentDate)
                .WithForwardCurve(forwardCurve)
                .WithIntrinsicTree()
                .WithCmdtySettlementRule(day => day)
                .WithDiscountFactorFunc((presentDate, cashFlowDate) => 1.0)
                .WithFixedGridSpacing(100)
                .WithLinearInventorySpaceInterpolation()
                .WithNumericalTolerance(1E-10)
                .Calculate();


            // Trivial intrinsic calc
            double volumeCycled = numDaysAtHighPrice * withdrawalRate;
            double withdrawPv = highPrice * volumeCycled * (1 - withdrawalCmdtyConsumed) - volumeCycled * withdrawalPerUnitCost;

            double injectionPv = -lowPrice * volumeCycled * (1 + injectionCmdtyConsumed) - volumeCycled * injectionPerUnitCost;

            double totalExpectedPv = withdrawPv + injectionPv;

            Assert.Equal(totalExpectedPv, valuationResults.NetPresentValue, 8);
        }

        [Fact]
        public void CalculateWithDecisionSimulator_DeepInTheMoneyWithIntrinsicTree_NpvAndDecisionProfileEqualsTrivialIntrinsicCalc()
        {
            var currentDate = new Day(2019, 8, 29);

            var storageStart = new Day(2019, 12, 1);
            var storageEnd = new Day(2020, 4, 1);

            const double storageStartingInventory = 0.0;
            const double minInventory = 0.0;
            const double maxInventory = 100_000.0;

            const double injectionPerUnitCost = 1.23;
            const double injectionCmdtyConsumed = 0.01;

            const double withdrawalPerUnitCost = 0.98;
            const double withdrawalCmdtyConsumed = 0.015;

            const double lowPrice = 23.87;
            const double highPrice = 150.32;
            const int numDaysAtHighPrice = 20;

            const double injectionRate = 400.0;
            const double withdrawalRate = 800.0;

            Day dateToSwitchToHighForwardPrice = storageEnd.Offset(-numDaysAtHighPrice);

            var forwardCurveBuilder = new TimeSeries<Day, double>.Builder(storageEnd - currentDate + 1);

            foreach (Day day in currentDate.EnumerateTo(storageEnd))
            {
                double forwardPrice = day < dateToSwitchToHighForwardPrice ? lowPrice : highPrice;
                forwardCurveBuilder.Add(day, forwardPrice);
            }

            TimeSeries<Day, double> forwardCurve = forwardCurveBuilder.Build();

            Day InjectionCostPaymentTerms(Day injectionDate)
            {
                return injectionDate.Offset(10);
            }

            Day WithdrawalCostPaymentTerms(Day withdrawalDate)
            {
                return withdrawalDate.Offset(4);
            }

            CmdtyStorage<Day> storage = CmdtyStorage<Day>.Builder
                .WithActiveTimePeriod(storageStart, storageEnd)
                .WithConstantInjectWithdrawRange(-withdrawalRate, injectionRate)
                .WithConstantMinInventory(minInventory)
                .WithConstantMaxInventory(maxInventory)
                .WithPerUnitInjectionCost(injectionPerUnitCost, InjectionCostPaymentTerms)
                .WithFixedPercentCmdtyConsumedOnInject(injectionCmdtyConsumed)
                .WithPerUnitWithdrawalCost(withdrawalPerUnitCost, WithdrawalCostPaymentTerms)
                .WithFixedPercentCmdtyConsumedOnWithdraw(withdrawalCmdtyConsumed)
                .WithNoCmdtyInventoryLoss()
                .WithNoInventoryCost()
                .MustBeEmptyAtEnd()
                .Build();

            (TreeStorageValuationResults<Day> valuationResults, ITreeDecisionSimulator<Day> decisionSimulator)
                                = TreeStorageValuation<Day>.ForStorage(storage)
                .WithStartingInventory(storageStartingInventory)
                .ForCurrentPeriod(currentDate)
                .WithForwardCurve(forwardCurve)
                .WithIntrinsicTree()
                .WithCmdtySettlementRule(day => day)                     // No discounting 
                .WithDiscountFactorFunc((presentDate, cashFlowDate) => 1.0)
                .WithFixedGridSpacing(100)
                .WithLinearInventorySpaceInterpolation()
                .WithNumericalTolerance(1E-10)
                .CalculateWithDecisionSimulator();

            // Calculate intrinsic decision profile
            var tree = valuationResults.Tree;
            var intrinsicSpotPath = new TimeSeries<Day, int>(tree.Indices, tree.Data.Select(x => 0));

            TreeSimulationResults<Day> simulateDecisions = decisionSimulator.SimulateDecisions(intrinsicSpotPath);
            
            // Trivial intrinsic calc
            double volumeCycled = numDaysAtHighPrice * withdrawalRate;
            double withdrawPv = highPrice * volumeCycled * (1 - withdrawalCmdtyConsumed) - volumeCycled * withdrawalPerUnitCost;

            double injectionPv = -lowPrice * volumeCycled * (1 + injectionCmdtyConsumed) - volumeCycled * injectionPerUnitCost;

            double totalExpectedPv = withdrawPv + injectionPv;

            DoubleTimeSeries<Day> intrinsicDecisionProfile = simulateDecisions.DecisionProfile;
            Assert.Equal(storageStart, intrinsicDecisionProfile.Start);
            Assert.Equal(storageEnd.Offset(-1), intrinsicDecisionProfile.End);

            double totalVolumeInLowPricePeriod = 0;
            foreach (Day day in storageStart.EnumerateTo(dateToSwitchToHighForwardPrice.Offset(-1)))
            {
                double decisionVolume = intrinsicDecisionProfile[day];
                Assert.True(decisionVolume >= 0);
                if (decisionVolume > 0)
                {
                    Assert.Equal(injectionRate, decisionVolume);
                }
                else
                {
                    Assert.Equal(0.0, decisionVolume);
                }
                totalVolumeInLowPricePeriod += decisionVolume;
            }
            Assert.Equal(volumeCycled, totalVolumeInLowPricePeriod);

            foreach (Day day in dateToSwitchToHighForwardPrice.EnumerateTo(storageEnd.Offset(-1)))
            {
                Assert.Equal(-withdrawalRate, intrinsicDecisionProfile[day]);
            }

            DoubleTimeSeries<Day> cmdtyVolumeConsumed = simulateDecisions.CmdtyVolumeConsumed;
            Assert.Equal(storageStart, cmdtyVolumeConsumed.Start);
            Assert.Equal(storageEnd.Offset(-1), cmdtyVolumeConsumed.End);

            for (int i = 0; i < cmdtyVolumeConsumed.Count; i++)
            {
                double cmdtyConsumed = cmdtyVolumeConsumed[i];
                double decisionVolume = intrinsicDecisionProfile[i];
                if (decisionVolume > 0) // Inject
                {
                    Assert.Equal(decisionVolume*injectionCmdtyConsumed, cmdtyConsumed);
                }
                else  // Withdraw
                {
                    Assert.Equal(-decisionVolume * withdrawalCmdtyConsumed, cmdtyConsumed);
                }
            }

            Assert.Equal(totalExpectedPv, simulateDecisions.StorageNpv, 8);
            Assert.Equal(totalExpectedPv, valuationResults.NetPresentValue, 8);
        }


        [Fact]
        public void Calculate_CurrentPeriodAfterEndPeriod_ResultsWithZeroNpvAndEmptyTimeSeries()
        {
            var storageStart = new Day(2019, 12, 1);
            var storageEnd = new Day(2020, 4, 1);

            var currentPeriod = storageEnd.Offset(1);

            CmdtyStorage<Day> storage = CmdtyStorage<Day>.Builder
                            .WithActiveTimePeriod(storageStart, storageEnd)
                            .WithConstantInjectWithdrawRange(-12, 43.5)
                            .WithZeroMinInventory()
                            .WithConstantMaxInventory(1000.0)
                            .WithPerUnitInjectionCost(1.0, day => day)
                            .WithNoCmdtyConsumedOnInject()
                            .WithPerUnitWithdrawalCost(2.0, day => day)
                            .WithNoCmdtyConsumedOnWithdraw()
                            .WithNoCmdtyInventoryLoss()
                            .WithNoInventoryCost()
                            .MustBeEmptyAtEnd()
                            .Build();

            DoubleTimeSeries<Day> forwardCurve = DoubleTimeSeries<Day>.Empty;
            DoubleTimeSeries<Day> spotVolCurve = DoubleTimeSeries<Day>.Empty; 
            
            const double meanReversion = 16.5;
            const double timeDelta = 1.0 / 365.0;

            TreeStorageValuationResults<Day> valuationResults = TreeStorageValuation<Day>.ForStorage(storage)
                            .WithStartingInventory(0.0)
                            .ForCurrentPeriod(currentPeriod)
                            .WithForwardCurve(forwardCurve)
                            .WithOneFactorTrinomialTree(spotVolCurve, meanReversion, timeDelta)
                            .WithCmdtySettlementRule(day => day) // No discounting 
                            .WithDiscountFactorFunc((presentDate, cashFlowDate) => 1.0)
                            .WithFixedGridSpacing(100)
                            .WithLinearInventorySpaceInterpolation()
                            .WithNumericalTolerance(1E-10)
                            .Calculate();

            Assert.Equal(0.0, valuationResults.NetPresentValue);
            Assert.True(valuationResults.Tree.IsEmpty);
            Assert.True(valuationResults.StorageNpvByInventory.IsEmpty);
            Assert.True(valuationResults.InventorySpaceGrids.IsEmpty);
            Assert.True(valuationResults.StorageNpvs.IsEmpty);
            Assert.True(valuationResults.InjectWithdrawDecisions.IsEmpty);
            Assert.True(valuationResults.InventorySpace.IsEmpty);
        }
    }
}
