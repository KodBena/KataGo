#ifndef CORE_FANCYMATH_H_
#define CORE_FANCYMATH_H_

#include <cmath>

#include "../core/global.h"

namespace FancyMath {
  //Compute pow(x,exponent) where the exponent is the same value on every call in a hot loop -
  //typically a config field that is fixed for the lifetime of a search. Under that condition the
  //comparisons below are perfectly predicted, and the exponents that have an exact cheap closed
  //form skip the libm call entirely.
  //
  //Every substitution here is BIT-IDENTICAL to std::pow for every input, so a call site can be
  //switched to this function without changing any result it produces:
  //  exponent == 1: the result is x itself.
  //  exponent == 2: x*x is the correctly-rounded square, which is what pow(x,2.0) returns.
  //Exponents whose cheap closed form is NOT bit-identical to pow are deliberately absent -
  //notably 0.5 (sqrt is correctly rounded where pow is not, so the two differ by one ulp on a
  //small fraction of inputs) and 0.25 (sqrt(sqrt(x)) rounds twice, and differs from pow on
  //roughly an eighth of inputs). Either would make this a silent change to what the caller
  //computes rather than a pure speedup, so that tradeoff stays with the caller.
  //
  //Both substitutions are asserted bit-for-bit in runTests(), so a platform whose libm disagrees
  //fails the test suite instead of silently computing something else.
  //
  //Scope, so a reader does not expect more of this than it gives: any other exponent falls through
  //to std::pow at full cost, so a call site whose configured exponent is neither 1 nor 2 gains
  //nothing here beyond two predicted comparisons. The constant-exponent precondition is about speed
  //only, never correctness - passing a genuinely varying exponent still returns exactly what
  //std::pow would, it just mispredicts the comparisons.
  inline double powConstExponent(double x, double exponent) {
    if(exponent == 1.0)
      return x;
    if(exponent == 2.0)
      return x * x;
    return std::pow(x, exponent);
  }

  //For large or extreme values these might not be too accurate, use GSL or Boost for more accuracy
  double beta(double a, double b);
  double logbeta(double a, double b);
  double incompleteBeta(double x, double a, double b);
  double regularizedIncompleteBeta(double x, double a, double b);

  double evaluateContinuedFraction(const std::function<double(int)>& numer, const std::function<double(int)>& denom, double tolerance, int maxTerms);

  //For large or extreme values these might not be too accurate, use GSL or Boost for more accuracy
  double tdistpdf(double x, double degreesOfFreedom);
  double tdistcdf(double x, double degreesOfFreedom);
  double betapdf(double x, double a, double b);
  double betacdf(double x, double a, double b);

  //Given z, compute and return an approximation for the value t
  //such that the probability that a draw from StudentT(degreesOfFreedom) > t
  //is the same as a probability that a draw from StandardNormal() > z
  double normToTApprox(double z, double degreesOfFreedom);

  //predProb is scaled into the range [epsilon,1.0-epsilon].
  double binaryCrossEntropy(double predProb, double targetProb, double epsilon);

  void runTests();
}


#endif  // CORE_FANCYMATH_H_
