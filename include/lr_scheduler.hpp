#include <cmath>
#include <omp.h>
using namespace std;

class LR{
 public:
  explicit LR(float initial_lr): initial_lr(initial_lr),step(0){};
  virtual float get_lr() = 0;

 protected:
  float initial_lr;
  int step;
};

class FixedLR : public LR{
  public:
    explicit FixedLR(float initial_lr) : LR(initial_lr){}
    float get_lr() override {
      return initial_lr;
  }
};

class ExponentialDecayLR : public LR {
public:
  ExponentialDecayLR(float initial_lr,float decay_rate) : LR(initial_lr),decay_rate(decay_rate){}
  float get_lr() override {
    return initial_lr * exp(-decay_rate * step++);
  }
private:
  float decay_rate;
};

class CosineAnnealingLR : public LR {
public:
  CosineAnnealingLR(float initial_lr,float min_lr,int T_max)
  : LR(initial_lr),min_lr(min_lr),T_max(T_max){}
  float get_lr() override {
    float cosine = cos(M_PI * step / T_max);
    float lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cosine);
    ++ step;
    return lr;
  }
private:
  double min_lr;
  int T_max; // Total period length
};

class StepDecayLR : public LR {
public:
    StepDecayLR(float initial_lr, float decay_factor, int step_size)
        : LR(initial_lr), decay_factor(decay_factor), step_size(step_size) {}

    float get_lr() override {
        int factor = step / step_size;
        ++step;
        return initial_lr * std::pow(decay_factor, factor);
    }

private:
    float decay_factor;
    int step_size;
};

class CustomLR : public LR {
public:
    explicit CustomLR(float initial_lr) : LR(initial_lr) {}

    float get_lr() override {
        if (step % 5 == 0) {
            initial_lr *= 0.8;
        }
        ++step;
        return initial_lr;
    }
};


