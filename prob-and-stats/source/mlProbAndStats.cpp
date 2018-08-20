// mlProbAndStats.cpp
// Probability and statistics related to machine learning
//
// Written by Bradley Denby
// Other contributors: None
//
// To the extent possible under law, the author(s) have dedicated all copyright
// and related and neighboring rights to this software to the public domain
// worldwide. This software is distributed without any warranty.
//
// You should have received a copy of the CC0 Public Domain Dedication with this
// software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.

// Standard libraries
#include <cstddef>   // size_t
#include <iostream>  // cout
#include <ostream>   // endl
#include <random>    // mt19937_64

// Stats
#include "stats.hpp" // Stats library

int main(int argc, char** argv) {

  std::mt19937_64 randeng(1);
  size_t numTrials  = 1000000;

  // Bernoulli: discrete probability distribution of a random variable that
  // takes the value 1 with probability p and the value 0 with probability 1-p
  double bern_p = 0.11;
  unsigned int bernRA = 0; // Bernoulli realization accumulator
  std::cout << "Bernoulli, p=" << bern_p
            << std::endl;
  for(size_t i=0; i<numTrials; i++) {
    bernRA += stats::rbern(bern_p, randeng);
  }
  std::cout << "  Mean outcome: "
            << static_cast<double>(bernRA)/static_cast<double>(numTrials)
            << std::endl
            << "  Expected value: " << bern_p
            << std::endl;

  // Binomial: discrete probability distribution of a random variable that takes
  // a value from 0 to n determined by n independent Bernoulli trials
  unsigned int binom_n = 12;
  double binom_p = 0.21;
  unsigned int binomRA = 0; // Binomial realization accumulator
  std::cout << "Binomial, n=" << binom_n << " p=" << binom_p
            << std::endl;
  for(size_t i=0; i<numTrials; i++) {
    binomRA += stats::rbinom(binom_n, binom_p, randeng);
  }
  std::cout << "  Mean outcome: "
            << static_cast<double>(binomRA)/static_cast<double>(numTrials)
            << std::endl
            << "  Expected value: " << binom_n*binom_p
            << std::endl;

  // Exponential: the probability distribution that describes the time between
  // events in a Poisson process
  double exp_rate = 0.79;
  double expRA = 0; // Exponential realization accumulator
  std::cout << "Exponential, \u03BB=" << exp_rate
            << std::endl;
  for(size_t i=0; i<numTrials; i++) {
    expRA += stats::rexp(exp_rate, randeng);
  }
  std::cout << "  Mean outcome: " << expRA/static_cast<double>(numTrials)
            << std::endl
            << "  Expected value: " << 1.0/exp_rate
            << std::endl;

  // Logistic: a continuous probability distribution that resembles a normal
  // distribution except with heavier tails
  double logis_u = 0.88;
  double logis_o = 0.32;
  double logisRA = 0; // Logistic realization accumulator
  std::cout << "Logistic, \u03BC=" << logis_u << " \u03C3=" << logis_o
            << std::endl;
  for(size_t i=0; i<numTrials; i++) {
    logisRA += stats::rlogis(logis_u, logis_o, randeng);
  }
  std::cout << "  Mean outcome: " << logisRA/static_cast<double>(numTrials)
            << std::endl
            << "  Expected value: " << logis_u
            << std::endl;

  // Uniform: a continuous probability distribution such that each realization
  // is equally likely
  double unif_a = 0.34;
  double unif_b = 0.69;
  double unifRA = 0; // Uniform realization accumulator
  std::cout << "Uniform, a=" << unif_a << " b=" << unif_b
            << std::endl;
  for(size_t i=0; i<numTrials; i++) {
    unifRA += stats::runif(unif_a, unif_b, randeng);
  }
  std::cout << "  Mean outcome: " << unifRA/static_cast<double>(numTrials)
            << std::endl
            << "  Expected value: " << (unif_a+unif_b)/2.0
            << std::endl;

  // Poisson: a discrete probability distribution of events occurring in a fixed
  // interval given that the events occur independently at a given rate
  unsigned int pois_rate = 4;
  unsigned int poisRA = 0; // Poisson realization accumulator
  std::cout << "Poisson, \u03BB=" << pois_rate
            << std::endl;
  for(size_t i=0; i<numTrials; i++) {
    poisRA += stats::rpois(pois_rate, randeng);
  }
  std::cout << "  Mean outcome: "
            << static_cast<double>(poisRA)/static_cast<double>(numTrials)
            << std::endl
            << "  Expected value: " << pois_rate
            << std::endl;

  // Gaussian distribution
  double norm_u = 0.60;
  double norm_o = 0.93;
  double normRA = 0; // Gaussian realization accumulator
  std::cout << "Gaussian, \u03BC=" << norm_u << " \u03C3=" << norm_o
            << std::endl;
  for(size_t i=0; i<numTrials; i++) {
    normRA += stats::rnorm(norm_u, norm_o, randeng);
  }
  std::cout << "  Mean outcome: " << normRA/static_cast<double>(numTrials)
            << std::endl
            << "  Expected value: " << norm_u
            << std::endl;

  // Multivariate Gaussian distribution
  // with Eigen

  return 0;
}
