#region License
// Copyright (c) 2021 Jake Fowler
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
using Xunit;

namespace Cmdty.Storage.Test
{
    public sealed class StepInjectWithdrawConstraintTest
    {
        private readonly StepInjectWithdrawConstraint _stepConstraint;
        private readonly List<InjectWithdrawRangeByInventory> _injectWithdrawRanges;

        public StepInjectWithdrawConstraintTest()
        {
            _injectWithdrawRanges = new List<InjectWithdrawRangeByInventory>
            {
                (inventory: 0.0, (minInjectWithdrawRate: -44.85, maxInjectWithdrawRate: 56.8)),
                (inventory: 100.0, (minInjectWithdrawRate: -45.01, maxInjectWithdrawRate: 54.5)),
                (inventory: 300.0, (minInjectWithdrawRate: -45.78, maxInjectWithdrawRate: 52.01)),
                (inventory: 600.0, (minInjectWithdrawRate: -46.17, maxInjectWithdrawRate: 51.9)),
                (inventory: 800.0, (minInjectWithdrawRate: -46.99, maxInjectWithdrawRate: 50.8)),
                (inventory: 1000.0, (minInjectWithdrawRate: -46.99, maxInjectWithdrawRate: 50.8))
            };
            _stepConstraint = new StepInjectWithdrawConstraint(_injectWithdrawRanges);
        }

        [Fact]
        public void InventorySpaceUpperBound_ConstantInjectWithdrawRate_EqualsNextPeriodInventoryPlusMaxWithdrawalRateAdjustedForLoss()
        {
            const double maxInjectionRate = 56.8;
            const double maxWithdrawalRate = 47.12;

            var injectWithdrawRange = new InjectWithdrawRange(-maxWithdrawalRate, maxInjectionRate);

            const double inventoryPercentLoss = 0.03;
            const double minInventory = 0.0;
            const double maxInventory = 1000.0;

            var injectWithdrawalRanges = new List<InjectWithdrawRangeByInventory>
            {
                (inventory: 0.0, injectWithdrawRange),
                (inventory: 100.0, injectWithdrawRange),
                (inventory: 300.0, injectWithdrawRange),
                (inventory: 600.0, injectWithdrawRange),
                (inventory: 800.0, injectWithdrawRange),
                (inventory: 1000.0, injectWithdrawRange),
            };

            var stepInjectWithdrawConstraint = new StepInjectWithdrawConstraint(injectWithdrawalRanges);

            const double nextPeriodMinInventory = 320.0;
            const double nextPeriodMaxInventory = 620.0;
            double thisPeriodMaxInventory = stepInjectWithdrawConstraint.InventorySpaceUpperBound(nextPeriodMinInventory, nextPeriodMaxInventory, minInventory, maxInventory, inventoryPercentLoss);

            const double expectedThisPeriodMaxInventory = (nextPeriodMaxInventory + maxWithdrawalRate) / (1 - inventoryPercentLoss);
            Assert.Equal(expectedThisPeriodMaxInventory, thisPeriodMaxInventory, 12);
        }

        [Fact]
        public void InventorySpaceLowerBound_ConstantInjectWithdrawRate_EqualsNextPeriodInventoryMinusMaxInjectRateAdjustedForLoss()
        {
            const double maxInjectionRate = 56.8;
            const double maxWithdrawalRate = 47.12;

            var injectWithdrawRange = new InjectWithdrawRange(-maxWithdrawalRate, maxInjectionRate);

            const double inventoryPercentLoss = 0.03;
            const double minInventory = 0.0;
            const double maxInventory = 1000.0;

            var injectWithdrawalRanges = new List<InjectWithdrawRangeByInventory>
            {
                (inventory: 0.0, injectWithdrawRange),
                (inventory: 100.0, injectWithdrawRange),
                (inventory: 300.0, injectWithdrawRange),
                (inventory: 600.0, injectWithdrawRange),
                (inventory: 800.0, injectWithdrawRange),
                (inventory: 1000.0, injectWithdrawRange),
            };

            var stepInjectWithdrawConstraint = new StepInjectWithdrawConstraint(injectWithdrawalRanges);

            const double nextPeriodMinInventory = 620.0;
            const double nextPeriodMaxInventory = 870.0;
            double thisPeriodMinInventory = stepInjectWithdrawConstraint.InventorySpaceLowerBound(nextPeriodMinInventory, nextPeriodMaxInventory, minInventory, maxInventory, inventoryPercentLoss);

            const double expectedThisPeriodMinInventory = (nextPeriodMinInventory - maxInjectionRate) / (1 - inventoryPercentLoss);
            Assert.Equal(expectedThisPeriodMinInventory, thisPeriodMinInventory, 12);
        }

        [Fact]
        public void InventorySpaceUpperBound_InventoryDependentInjectWithdrawRate_ConsistentWithGetInjectWithdrawRange()
        {
            const double inventoryPercentLoss = 0.03;
            const double minInventory = 0.0;
            const double maxInventory = 1000.0;

            const double nextPeriodMinInventory = 320.0;
            const double nextPeriodMaxInventory = 590.5;
            double thisPeriodMaxInventory = _stepConstraint.InventorySpaceUpperBound(nextPeriodMinInventory, nextPeriodMaxInventory, minInventory, maxInventory, inventoryPercentLoss);
            double thisPeriodMaxWithdrawalRateAtInventory = _stepConstraint.GetInjectWithdrawRange(thisPeriodMaxInventory).MinInjectWithdrawRate;

            double derivedNextPeriodMaxInventory = thisPeriodMaxInventory * (1 - inventoryPercentLoss) + thisPeriodMaxWithdrawalRateAtInventory;
            Assert.Equal(nextPeriodMaxInventory, derivedNextPeriodMaxInventory, 12);
        }

        [Fact]
        public void InventorySpaceLowerBound_InventoryDependentInjectWithdrawRate_ConsistentWithGetInjectWithdrawRange()
        {
            const double inventoryPercentLoss = 0.03;
            const double minInventory = 0.0;
            const double maxInventory = 1000.0;

            const double nextPeriodMinInventory = 552.0;
            const double nextPeriodMaxInventory = 734.0;
            double thisPeriodMinInventory = _stepConstraint.InventorySpaceLowerBound(nextPeriodMinInventory, nextPeriodMaxInventory, minInventory, maxInventory, inventoryPercentLoss);
            double thisPeriodMaxInjectRateAtInventory = _stepConstraint.GetInjectWithdrawRange(thisPeriodMinInventory).MaxInjectWithdrawRate;

            double derivedNextPeriodMinInventory = thisPeriodMinInventory * (1 - inventoryPercentLoss) + thisPeriodMaxInjectRateAtInventory;
            Assert.Equal(nextPeriodMinInventory, derivedNextPeriodMinInventory, 12);
        }

        [Theory]
        [InlineData(99.0, -44.85, 56.8)]
        [InlineData(610.85, -46.17, 51.9)]
        [InlineData(999.99, -46.99, 50.8)]
        public void GetInjectWithdrawRange_AsExpected(double inventory, double expectedMinInjectWithdraw, double expectedMaxInjectWithdraw)
        {
            (double minInjectWithdraw, double maxInjectWithdraw) = _stepConstraint.GetInjectWithdrawRange(inventory);
            Assert.Equal(expectedMinInjectWithdraw, minInjectWithdraw);
            Assert.Equal(expectedMaxInjectWithdraw, maxInjectWithdraw);
        }
        
        [Fact]
        public void GetInjectWithdrawRange_InventoryEqualsPiecewisePillars_EqualToInputsAtInventoryPillars()
        {
            foreach ((double inventoryPillar, InjectWithdrawRange inputInjectWithdrawRange) in _injectWithdrawRanges)
            {
                InjectWithdrawRange outputInjectWithdrawRange = _stepConstraint.GetInjectWithdrawRange(inventoryPillar);
                Assert.Equal(inputInjectWithdrawRange.MinInjectWithdrawRate, outputInjectWithdrawRange.MinInjectWithdrawRate);
                Assert.Equal(inputInjectWithdrawRange.MaxInjectWithdrawRate, outputInjectWithdrawRange.MaxInjectWithdrawRate);
            }
        }

        [Fact]
        public void GetInjectWithdrawRange_InventoryBelowMinInventory_ThrowsArgumentException()
        {
            Assert.Throws<ArgumentException>(() => _stepConstraint.GetInjectWithdrawRange(-0.01));
        }
        
        [Fact]
        public void GetInjectWithdrawRange_InventoryAboveMaxInventory_ThrowsArgumentException()
        {
            Assert.Throws<ArgumentException>(() => _stepConstraint.GetInjectWithdrawRange(1000.001));
        }

        [Fact]
        public void Constructor_InjectRatesIncreaseWithInventory_ThrowsArgumentException()
        {
            var injectWithdrawRanges = new List<InjectWithdrawRangeByInventory>
            {
                (inventory: 0.0, (minInjectWithdrawRate: -44.85, maxInjectWithdrawRate: 56.8)),
                (inventory: 100.0, (minInjectWithdrawRate: -45.01, maxInjectWithdrawRate: 54.5)),
                (inventory: 300.0, (minInjectWithdrawRate: -45.78, maxInjectWithdrawRate: 52.01)),
                (inventory: 600.0, (minInjectWithdrawRate: -46.17, maxInjectWithdrawRate: 52.01001)),
                (inventory: 800.0, (minInjectWithdrawRate: -46.99, maxInjectWithdrawRate: 50.8)),
                (inventory: 1000.0, (minInjectWithdrawRate: -46.99, maxInjectWithdrawRate: 50.8))
            };
            Assert.Throws<ArgumentException>(() => new StepInjectWithdrawConstraint(injectWithdrawRanges));
        }

        [Fact]
        public void Constructor_WithdrawRatesDecreaseWithInventory_ThrowsArgumentException()
        {
            var injectWithdrawRanges = new List<InjectWithdrawRangeByInventory>
            {
                (inventory: 0.0, (minInjectWithdrawRate: -44.85, maxInjectWithdrawRate: 56.8)),
                (inventory: 100.0, (minInjectWithdrawRate: -45.01, maxInjectWithdrawRate: 54.5)),
                (inventory: 300.0, (minInjectWithdrawRate: -45.78, maxInjectWithdrawRate: 52.01)),
                (inventory: 600.0, (minInjectWithdrawRate: -45.77, maxInjectWithdrawRate: 51.9)),
                (inventory: 800.0, (minInjectWithdrawRate: -46.99, maxInjectWithdrawRate: 50.8)),
                (inventory: 1000.0, (minInjectWithdrawRate: -46.99, maxInjectWithdrawRate: 50.8))
            };
            Assert.Throws<ArgumentException>(() => new StepInjectWithdrawConstraint(injectWithdrawRanges));
        }

    }
}
