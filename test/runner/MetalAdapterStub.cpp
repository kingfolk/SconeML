#include "test/runner/Pipelines.h"
#include <stdexcept>
namespace sconeml::test {
void compileMetal(const std::string &, const std::string &) {
  throw std::runtime_error("Metal unavailable on this platform");
}
} // namespace sconeml::test
