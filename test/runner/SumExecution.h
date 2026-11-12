#ifndef SCONEML_SUM_EXECUTION_H
#define SCONEML_SUM_EXECUTION_H
#include "test/runner/Pipelines.h"
namespace sconeml::test {
Output executeSum(const TestCase &, const Run &, Output &, bool disableMetal);
}
#endif
