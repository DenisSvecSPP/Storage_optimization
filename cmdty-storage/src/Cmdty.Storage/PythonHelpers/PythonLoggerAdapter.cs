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
using JetBrains.Annotations;
using Microsoft.Extensions.Logging;

namespace Cmdty.Storage.PythonHelpers
{
    public sealed class PythonLoggerAdapter<T> : ILogger<T> // TODO add to Cmdty.Core
    {
        private readonly Func<int, bool> _pythonIsEnabled;
        private readonly Action<int, string> _pythonLog;
        
        public PythonLoggerAdapter([NotNull] Func<int, bool> pythonIsEnabled, [NotNull] Action<int, string> pythonLog)
        {
            _pythonIsEnabled = pythonIsEnabled ?? throw new ArgumentNullException(nameof(pythonIsEnabled));
            _pythonLog = pythonLog ?? throw new ArgumentNullException(nameof(pythonLog));
        }

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception exception, Func<TState, Exception, string> formatter)
        {
            string message = state.ToString();
            if (exception != null)
                message = message + Environment.NewLine + exception;
            int pythonLogLevel = ConvertLogLevel(logLevel);
            _pythonLog(pythonLogLevel, message);
        }

        public bool IsEnabled(LogLevel logLevel)
        {
            int pythonLogLevel = ConvertLogLevel(logLevel);
            return _pythonIsEnabled(pythonLogLevel);
        }

        public IDisposable BeginScope<TState>(TState state)
        {
            throw new NotImplementedException("BeginScope not implemented for PythonLoggerAdapter.");
        }

        private static int ConvertLogLevel(LogLevel logLevel)
        {
            return logLevel switch
            {
                LogLevel.Critical => 50,
                LogLevel.Error => 40,
                LogLevel.Warning => 30,
                LogLevel.Information => 20,
                LogLevel.Debug => 10,
                LogLevel.Trace => 10, // No Trace level in Python logging so use Debug level
                LogLevel.None => 0,
                _ => throw new ArgumentException($"LogLevel value of {logLevel:G} not recognised.")
            };
        }

    }
}
