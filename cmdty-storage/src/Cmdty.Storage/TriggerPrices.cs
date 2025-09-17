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

namespace Cmdty.Storage
{
    public sealed class TriggerPrices
    {
        public double? MaxInjectVolume { get; }
        public double? MaxInjectTriggerPrice { get; }
        public double? MaxWithdrawVolume { get; }
        public double? MaxWithdrawTriggerPrice { get; }

        public TriggerPrices(double? maxInjectVolume, double? maxInjectTriggerPrice, double? maxWithdrawVolume, double? maxWithdrawTriggerPrice)
        {
            MaxInjectVolume = maxInjectVolume;
            MaxInjectTriggerPrice = maxInjectTriggerPrice;
            MaxWithdrawVolume = maxWithdrawVolume;
            MaxWithdrawTriggerPrice = maxWithdrawTriggerPrice;
        }

        public bool HasInjectPrice => MaxInjectTriggerPrice.HasValue;

        public bool HasWithdrawPrice => MaxWithdrawTriggerPrice.HasValue;

        public override string ToString()
        {
            return $"{nameof(MaxInjectVolume)}: {MaxInjectVolume}, {nameof(MaxInjectTriggerPrice)}: {MaxInjectTriggerPrice}, " +
                   $"{nameof(MaxWithdrawVolume)}: {MaxWithdrawVolume}, {nameof(MaxWithdrawTriggerPrice)}: {MaxWithdrawTriggerPrice}";
        }

        public sealed class Builder
        {
            public double? MaxInjectVolume { get; set; }
            public double? MaxInjectTriggerPrice { get; set; }
            public double? MaxWithdrawVolume { get; set; }
            public double? MaxWithdrawTriggerPrice { get; set; }

            public TriggerPrices Build() => new TriggerPrices(MaxInjectVolume, MaxInjectTriggerPrice, MaxWithdrawVolume, MaxWithdrawTriggerPrice);

        }
    }
}
